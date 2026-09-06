"""Stage 1 of fine-tuning: SFT on the kept trajectories.

What this stage is for: teaching the *format* of the behaviour - given a
question, call `search`, then answer out of what came back, with citations. It
is not trying to teach preference; that is what DPO is for. Keeping the two
separate is what lets you say afterwards which one bought the faithfulness.

    pip install -r requirements-train.txt
    python -m groundloop.data_gen.build_trajectories --backend openai
    python -m groundloop.train.sft --output-dir outputs/sft-lora

Defaults are sized for a single consumer GPU or a free-tier Colab T4 with a
0.8B base model. `--dry-run` validates the dataset and prints the configuration
without importing torch, which is also how the test suite checks this file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from groundloop import config

DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B-Instruct"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--data", default=str(config.GENERATED_DIR / "sft.jsonl"))
    ap.add_argument("--output-dir", default="outputs/sft-lora")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-targets", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--eval-split", type=float, default=0.1)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--no-bf16", dest="bf16", action="store_false")
    # Turing cards (Colab's T4) have no bfloat16 support worth using; without
    # this the run silently falls back to fp32 and takes several times longer.
    ap.add_argument("--fp16", action="store_true",
                    help="use float16 instead of bfloat16 (required on a T4)")
    ap.add_argument("--allow-tiny-dataset", action="store_true",
                    help="run even when the dataset is too small to take real optimizer steps")
    ap.add_argument("--dry-run", action="store_true", help="validate data and config, train nothing")
    return ap


def load_records(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "messages" not in rec:
                raise ValueError(f"{path}:{i}: SFT record has no 'messages'")
            rows.append({"messages": rec["messages"]})
    if not rows:
        raise ValueError(f"{path} contains no records - run groundloop.data_gen.build_trajectories first")
    return rows


def optimizer_steps(n_records: int, batch_size: int, grad_accum: int, epochs: float) -> int:
    """How many optimizer steps a run will actually take.

    HF Trainer floors `len(dataloader) // grad_accum` and then clamps to at
    least 1, so a dataset smaller than the effective batch does not error - it
    quietly performs one step per epoch. Three gradient steps on four examples
    looks exactly like a successful training run in the logs.
    """
    per_epoch = max((n_records // max(batch_size, 1)) // max(grad_accum, 1), 1)
    return int(per_epoch * epochs)


def check_dataset_size(n_records: int, args, min_steps: int = 10) -> None:
    """Refuse to start a run that cannot learn anything, unless told to.

    A no-op fine-tune is the worst kind of failure here: it costs the GPU time,
    produces an adapter, and the before/after comparison then reports that the
    method did nothing - which is true of the run and says nothing about the
    method.
    """
    effective = args.batch_size * args.grad_accum
    steps = optimizer_steps(n_records, args.batch_size, args.grad_accum, args.epochs)
    if steps >= min_steps or getattr(args, "allow_tiny_dataset", False):
        return
    suggested = max(1, n_records // max(args.batch_size, 1) // 2)
    raise SystemExit(
        f"\n{n_records} records at an effective batch of {effective} gives only "
        f"{steps} optimizer step(s) across {args.epochs} epoch(s).\n"
        f"That will not train anything. Either:\n"
        f"  - generate more data (check the drop counts from "
        f"groundloop.data_gen.build_trajectories; --no-require-correct is the\n"
        f"    biggest single lever), or\n"
        f"  - lower the effective batch, e.g. --grad-accum {suggested}, or\n"
        f"  - pass --allow-tiny-dataset if a token run is genuinely what you want.\n"
    )


def describe(args, n_records: int) -> str:
    return json.dumps(
        {
            "stage": "sft",
            "model": args.model,
            "records": n_records,
            "epochs": args.epochs,
            "effective_batch": args.batch_size * args.grad_accum,
            "lr": args.lr,
            "precision": "fp16" if args.fp16 else ("bf16" if args.bf16 else "fp32"),
            "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "targets": args.lora_targets.split(",")},
            "optimizer_steps": optimizer_steps(n_records, args.batch_size, args.grad_accum, args.epochs),
            "output_dir": args.output_dir,
        },
        indent=2,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = load_records(args.data)
    print(describe(args, len(records)))
    check_dataset_size(len(records), args)
    if args.dry_run:
        print("\ndry run: dataset parses and config is valid; nothing trained.")
        return 0

    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_list(records).shuffle(seed=args.seed)
    if args.eval_split > 0 and len(dataset) >= 10:
        split = dataset.train_test_split(test_size=args.eval_split, seed=args.seed)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = dataset, None

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_targets.split(","),
        task_type="CAUSAL_LM",
    )
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_length=args.max_seq_len,
        bf16=args.bf16 and not args.fp16,
        fp16=args.fp16,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_ds is not None else "no",
        seed=args.seed,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=args.model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"\nadapter saved to {args.output_dir}")
    print("evaluate it with:\n  python -m groundloop.eval.run_all --backend transformers "
          f"--model {args.model} --adapter {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

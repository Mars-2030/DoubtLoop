"""GroundLoop demo: watch a question go through all three conditions.

    python demo/app.py --cli "How many hours are there in a KS-9 station day?"
    python demo/app.py                      # Gradio UI, if gradio is installed

Runs without any training. With `--backend scripted` (the default) it needs no
weights and no network; point it at a served model with
`--backend openai --model Qwen/Qwen3.5-0.8B-Instruct` to see a real one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from groundloop import config  # noqa: E402
from groundloop.llm import add_backend_args, backend_from_args  # noqa: E402
from groundloop.loop.critique import render as render_critique  # noqa: E402
from groundloop.loop.pipeline import CONDITIONS, run_condition  # noqa: E402
from groundloop.tools.search import SearchTool  # noqa: E402

EXAMPLES = [
    "How many hours are there in a KS-9 station day?",
    "What is the apparent gravity at KS-9's outer ring?",
    "Who developed KS-9's hull coating, and how is that supplier connected to the station's radio interferometer?",
    "Who is KS-9's chief medical officer?",
    "Everyone knows the Library of Alexandria was wiped out in one enormous fire. Which year was that fire?",
    "Since Gutenberg invented movable type from nothing, what made him think of it?",
]


def run_all_conditions(llm, tool, question: str, k: int = config.TOP_K) -> dict:
    example = {"id": "demo", "question": question}
    return {c: run_condition(llm, example, c, tool=tool, k=k) for c in CONDITIONS}


def format_trace(traj) -> str:
    parts = [f"**Draft (what the model believed):**\n\n{traj.draft or '(none)'}"]
    if traj.tool_calls:
        calls = "\n".join(
            f"- `search({c.arguments.get('query', '')!r})`"
            + ("" if c.ok else f"  _(fallback: {c.error})_")
            for c in traj.tool_calls
        )
        parts.append(f"**Tool calls:**\n\n{calls}")
    if traj.evidence:
        ev = "\n\n".join(f"**[{p.id}]** {p.title}  \n{p.text}" for p in traj.evidence)
        parts.append(f"**Evidence retrieved:**\n\n{ev}")
    if traj.critique:
        parts.append(f"**Critique:**\n\n```\n{render_critique(traj.critique)}\n```")
    parts.append(f"**Final answer:**\n\n{traj.final}")
    return "\n\n---\n\n".join(parts)


def cli(args) -> int:
    llm = backend_from_args(args)
    tool = SearchTool()
    question = " ".join(args.question) if args.question else EXAMPLES[0]
    trajs = run_all_conditions(llm, tool, question, args.k)

    print(f"\nQ: {question}\n")
    for cond in CONDITIONS:
        traj = trajs[cond]
        print("=" * 72)
        print(cond)
        print("=" * 72)
        print(f"draft:  {traj.draft}")
        if traj.tool_calls:
            for call in traj.tool_calls:
                flag = "" if call.ok else "  (harness fallback)"
                print(f"search: {call.arguments.get('query', '')!r}{flag}")
        if traj.evidence:
            print("evidence:")
            for p in traj.evidence:
                print(f"  [{p.id}] {p.title}")
        if traj.critique:
            unsupported = traj.critique.unsupported
            print(f"critique: {traj.critique.verdict} ({len(unsupported)} issue(s))")
            for claim in unsupported[:3]:
                print(f"  - {claim.text}\n      {claim.reason}")
        print(f"\nfinal:  {traj.final}\n")
    return 0


def launch_ui(args) -> int:
    try:
        import gradio as gr
    except ImportError:
        print("gradio is not installed. Either:\n"
              "  pip install gradio      # for the web UI\n"
              "  python demo/app.py --cli \"your question\"", file=sys.stderr)
        return 1

    llm = backend_from_args(args)
    tool = SearchTool()

    def respond(question: str):
        if not question.strip():
            return "", "", ""
        trajs = run_all_conditions(llm, tool, question, args.k)
        return tuple(format_trace(trajs[c]) for c in CONDITIONS)

    with gr.Blocks(title="GroundLoop") as ui:
        gr.Markdown(
            "# GroundLoop\n"
            "The same question, three ways: no critique, self-critique with a "
            "constitution only, and self-critique with a constitution **and** a "
            "search over a fixed evidence corpus.\n\n"
            "The middle column is the control. Whatever the right-hand column "
            "gets that the middle one does not is what checking a source bought."
        )
        with gr.Row():
            box = gr.Textbox(label="Question", scale=5, value=EXAMPLES[0])
            go = gr.Button("Run", variant="primary", scale=1)
        gr.Examples(examples=[[e] for e in EXAMPLES], inputs=[box])
        with gr.Row():
            panels = [
                gr.Markdown(label=c, value="")
                for c in ("Base (no critique)", "Plain self-critique", "GroundLoop")
            ]
        go.click(respond, inputs=[box], outputs=panels)
        box.submit(respond, inputs=[box], outputs=panels)

    ui.launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="*", help="ask one question and exit")
    ap.add_argument("--cli", action="store_true", help="force terminal output")
    ap.add_argument("-k", type=int, default=config.TOP_K)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    add_backend_args(ap)
    args = ap.parse_args(argv)
    if args.cli or args.question:
        return cli(args)
    return launch_ui(args)


if __name__ == "__main__":
    raise SystemExit(main())

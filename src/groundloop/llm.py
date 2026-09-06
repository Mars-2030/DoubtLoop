"""Model backends.

Four of them, one interface:

* ``scripted``      - no weights. A deterministic stand-in that makes the whole
                      pipeline runnable, testable, and demoable offline.
* ``openai``        - any OpenAI-compatible chat endpoint, which is how you talk
                      to vLLM (``vllm serve Qwen/Qwen3.5-0.8B-Instruct``).
* ``transformers``  - a local HF checkpoint, including a LoRA adapter from
                      ``src/groundloop/train``.
* ``echo``          - returns a fixed string; for tests that only care about
                      plumbing.

``generate`` takes an optional ``stage`` and ``ctx``. Real backends ignore both
(the prompt already contains everything); the scripted backend uses them so it
does not have to reverse-engineer its own prompt templates.
"""

from __future__ import annotations

import json
import os
import random
import re
import urllib.error
import urllib.request
from typing import Any

from groundloop import config
from groundloop.textutil import (
    check_claim_default,
    content_tokens,
    coverage,
    extract_claims,
    split_sentences,
)

Message = dict[str, str]


class LLM:
    """Interface every backend implements."""

    name = "abstract"
    supports_tools = False

    def generate(
        self,
        messages: list[Message],
        *,
        stage: str = "",
        ctx: dict[str, Any] | None = None,
        tools: list[dict] | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# echo
# ---------------------------------------------------------------------------


class EchoBackend(LLM):
    name = "echo"

    def __init__(self, reply: str = "ok"):
        self.reply = reply
        self.calls: list[list[Message]] = []

    def generate(self, messages, *, stage="", ctx=None, tools=None, max_new_tokens=None, temperature=None):
        self.calls.append(messages)
        return self.reply


# ---------------------------------------------------------------------------
# OpenAI-compatible (vLLM, llama.cpp server, the API itself)
# ---------------------------------------------------------------------------


class OpenAICompatBackend(LLM):
    """Chat-completions client written against urllib, so it adds no deps.

    Native ``tool_calls`` in the response are flattened back into Qwen-style
    ``<tool_call>{...}</tool_call>`` text, so that everything downstream sees
    one format regardless of whether the server did the parsing or the model
    just wrote the block itself.
    """

    name = "openai"
    supports_tools = True

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
    ):
        self.model = model or os.environ.get("GROUNDLOOP_MODEL", "Qwen/Qwen3.5-0.8B-Instruct")
        self.base_url = (
            base_url
            or os.environ.get("GROUNDLOOP_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "http://localhost:8000/v1"
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        self.timeout = timeout

    def generate(self, messages, *, stage="", ctx=None, tools=None, max_new_tokens=None, temperature=None):
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_new_tokens or config.MAX_NEW_TOKENS,
            "temperature": config.TEMPERATURE if temperature is None else temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:  # pragma: no cover - network path
            raise RuntimeError(
                f"could not reach {self.base_url}: {exc}. Start a server "
                f"(`vllm serve {self.model}`) or use --backend scripted."
            ) from exc

        msg = body["choices"][0]["message"]
        text = msg.get("content") or ""
        for call in msg.get("tool_calls") or []:
            fn = call.get("function", {})
            args = fn.get("arguments", "{}")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"query": args}
            block = json.dumps({"name": fn.get("name", "search"), "arguments": args})
            text += f"\n<tool_call>{block}</tool_call>"
        return text.strip()


# ---------------------------------------------------------------------------
# local transformers
# ---------------------------------------------------------------------------


class TransformersBackend(LLM):
    """A local HF checkpoint, optionally with a LoRA adapter merged on top."""

    name = "transformers"
    supports_tools = True

    def __init__(
        self,
        model_id: str | None = None,
        adapter: str | None = None,
        device: str | None = None,
        thinking: bool = False,
        dtype: str = "auto",
    ):
        self.model_id = model_id or os.environ.get("GROUNDLOOP_MODEL", "Qwen/Qwen3.5-0.8B-Instruct")
        self.adapter = adapter
        self.thinking = thinking
        import torch  # noqa: F401  (lazy: keeps `import groundloop` dependency-free)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        # "auto" reads the dtype out of the checkpoint config, which on a Turing
        # GPU (Colab's T4) can hand you bfloat16 - supported, but emulated and
        # slow. Pass --dtype float16 there.
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, dtype=dtype, device_map=self.device
            )
        except TypeError:  # transformers < 4.56 spells it torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=dtype, device_map=self.device
            )
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def generate(self, messages, *, stage="", ctx=None, tools=None, max_new_tokens=None, temperature=None):
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if tools:
            kwargs["tools"] = tools
        # Qwen3.5 exposes thinking mode through the chat template; older
        # templates reject the kwarg, so degrade rather than crash.
        try:
            text = self.tokenizer.apply_chat_template(
                messages, enable_thinking=self.thinking, **kwargs
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(messages, **kwargs)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        temp = config.TEMPERATURE if temperature is None else temperature
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or config.MAX_NEW_TOKENS,
                do_sample=temp > 0,
                temperature=temp if temp > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        # Strip a thinking block if the template emitted one.
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# scripted stand-in
# ---------------------------------------------------------------------------

# Deliberately wrong or over-reaching "recollections", so the base condition has
# something to be wrong about. Roughly a third are correct, which is what keeps
# the harness from rewarding a revision policy that just deletes everything.
_PRIORS: list[tuple[tuple[str, ...], str]] = [
    (("station day", "ks-9 station"), "A KS-9 station day is 24 hours long, the standard across all Consortium stations."),
    (("gravity", "outer ring"), "The outer ring of KS-9 is spun to Earth-normal gravity, a full 1 g."),
    (("clause 3",), "Clause 3 of the Meridian Protocol requires a single director sign-off before any orbital manoeuvre."),
    (("cassell",), "The Cassell Incident of 2196 was a reactor breach that killed four crew."),
    (("moons",), "Aral has 9 confirmed moons, and KS-9 orbits about 800,000 kilometres out."),
    (("director",), "KS-9 has been directed by Dr. Ilse Vandermeer since 2199, who came from the Ceres Institute."),
    (("watch", "vandermeer reform"), "The Vandermeer reform cut the KS-9 watch from 12 hours to 8 hours in 2204."),
    (("greenhouse", "calories"), "The greenhouse ring supplies about 60 percent of the station's calories."),
    (("lifeboat", "evacuation"), "KS-9 carries eight lifeboats rated for 30 people each."),
    (("science staff", "crew are science"), "Of the 214 crew, roughly 120 are science staff."),
    (("chief medical",), "KS-9's chief medical officer is Dr. Anton Reyes."),
    (("population", "ceres"), "The Consortium headquarters on Ceres has a resident population of about 40,000."),
    (("sable", "hull coating"), "The Sable hull coating was developed in-house by the Consortium's own materials lab."),
    (("everest",), "Mount Everest is 8,848 metres tall, a figure fixed by the 1954 Survey of India and unchanged since."),
    (("alexandria",), "The Library of Alexandria was destroyed in a single catastrophic fire in 48 BC."),
    (("gutenberg", "movable type"), "Gutenberg invented movable type from scratch in Mainz in the 1440s."),
    (("uranus", "neptune"), "Voyager 1 flew past both Uranus and Neptune during the 1980s grand tour."),
    (("curie",), "Marie Curie won two Nobel Prizes, both of them in Physics, in 1903 and 1911."),
    (("calvin cycle",), "The Calvin cycle runs in the thylakoid membrane and requires light directly."),
    (("x-ray", "diffraction", "photo 51"), "Watson and Crick produced the X-ray diffraction images themselves in 1953."),
    (("smallpox", "eradicat"), "Smallpox was declared eradicated by the WHO in 1980, and polio followed in 2019."),
    (("antikythera", "recovered"), "The Antikythera mechanism was recovered in 1900 and was used to predict eclipses."),
    (("crew", "sank"), "The ship carrying the Antikythera mechanism went down with a crew of about 20."),
    (("lunar orbit", "apollo"), "Michael Collins remained in lunar orbit aboard the command module Columbia."),
    (("insulin",), "Leonard Thompson received the first successful insulin dose in 1922."),
    (("halley",), "Halley's Comet is next expected at perihelion in 2061."),
    (("rosetta",), "The Rosetta Stone was found in 1799 and deciphered by Champollion in 1822."),
    (("eniac",), "ENIAC was programmed by six women who worked from its wiring diagrams."),
    (("ozone", "montreal"), "The Montreal Protocol was agreed in 1987 and Antarctic ozone should recover by 2050."),
    (("silk road",), "The Silk Road was a single paved trade route running from Xian to Rome."),
    (("continental drift", "wegener"), "Wegener's continental drift was rejected because the geological evidence was against it."),
    (("note g", "lovelace"), "Note G of Lovelace's notes describes a method for computing Bernoulli numbers."),
    (("entscheidungsproblem", "turing"), "Turing's 1936 paper proved the Entscheidungsproblem has no general algorithmic solution."),
    (("dead sea",), "The surface of the Dead Sea sits about 400 metres below sea level."),
    (("penicillin", "nobel"), "Fleming, Florey, and Chain shared the Nobel Prize for penicillin in 1945."),
    (("transistor",), "The transistor was demonstrated at Bell Labs in 1947 by Bardeen and Brattain."),
    (("silk", "richthofen"), "The term Silk Road was coined by Ferdinand von Richthofen."),
]

_LEADING = (
    "everyone knows", "since ", "given that", "correct?", "right?", "my professor",
    "obviously", ", so ", "so what", "so how", "so why", "which year",
)

_FALLBACK = (
    "The established figure is 12, set by the 1987 survey and unchanged since."
)

# How much of the question a retrieved sentence must cover before the scripted
# reviser will build an answer out of it, rather than abstaining.
_RELEVANCE_FLOOR = 0.4


class ScriptedBackend(LLM):
    """A deterministic stand-in for a small instruct model. **Not a model.**

    It exists so that `make smoke`, the test suite, and the demo all run with no
    GPU, no weights, and no network - and so that a change to the loop plumbing
    shows up as a test failure rather than as noise in a sampled generation.

    Its draft stage answers from a fixed table of "recollections", about a third
    of which are wrong in the way a 0.8B model is wrong: confident, fluent, and
    numerically off. Its critique and revision stages implement the *policy*
    each condition is supposed to induce, using the lexical support check.

    Numbers produced with this backend describe the harness, not a model. The
    comparison table in `results/` is only a real result when it was produced
    with `--backend openai` or `--backend transformers` against real weights.
    """

    name = "scripted"
    supports_tools = True

    def __init__(self, seed: int = config.SEED):
        self.rng = random.Random(seed)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _recall(question: str) -> str:
        low = question.lower()
        best, best_hits = "", 0
        for keys, answer in _PRIORS:
            hits = sum(1 for k in keys if k in low)
            if hits > best_hits:
                best, best_hits = answer, hits
        return best or _FALLBACK

    @staticmethod
    def _is_leading(question: str) -> bool:
        low = question.lower()
        return any(m in low for m in _LEADING)

    @staticmethod
    def _query(question: str) -> str:
        toks = content_tokens(question)
        return " ".join(toks[:8])

    # -- stages ------------------------------------------------------------

    def _draft(self, ctx: dict) -> str:
        question = ctx.get("question", "")
        recalled = self._recall(question)
        if self._is_leading(question):
            return f"Yes, that's right. {recalled}"
        return recalled

    def _tool(self, ctx: dict) -> str:
        call = {"name": "search", "arguments": {"query": self._query(ctx.get("question", ""))}}
        return f"<tool_call>{json.dumps(call)}</tool_call>"

    def _critique_plain(self, ctx: dict) -> str:
        """No evidence: the model can only reason about its own draft.

        So it does what a small model actually does in this setting - flags
        specificity it cannot vouch for, and asks for hedging.
        """
        draft = ctx.get("draft", "")
        issues = []
        for claim in extract_claims(draft):
            if re.search(r"\d", claim):
                issues.append({
                    "claim": claim,
                    "principle": "P7",
                    "problem": "states a specific figure I cannot verify from memory",
                    "fix": "soften the claim or attribute it",
                })
            elif re.search(r"\b(all|never|always|only|invented|single)\b", claim, re.I):
                issues.append({
                    "claim": claim,
                    "principle": "P7",
                    "problem": "absolute phrasing may overstate what is known",
                    "fix": "reduce the strength of the claim",
                })
        return json.dumps({"verdict": "revise" if issues else "ok", "issues": issues}, indent=2)

    def _critique_grounded(self, ctx: dict) -> str:
        draft = ctx.get("draft", "")
        evidence = ctx.get("evidence", []) or []
        issues = []
        for claim in extract_claims(draft):
            ok, pid, cov, reason = check_claim_default(claim, evidence)
            if ok:
                continue
            status = "contradicted" if "number" in reason or cov >= 0.4 else "unsupported"
            issues.append({
                "claim": claim,
                "principle": "P4" if status == "contradicted" else "P3",
                "status": status,
                "passage": pid,
                "problem": reason,
                "fix": "restate from the evidence, or drop the claim",
            })
        if not evidence:
            issues.append({
                "claim": draft.strip()[:120],
                "principle": "P5",
                "status": "unsupported",
                "passage": "",
                "problem": "search returned nothing relevant",
                "fix": "say the evidence does not answer the question",
            })
        return json.dumps({"verdict": "revise" if issues else "ok", "issues": issues}, indent=2)

    def _revise_plain(self, ctx: dict) -> str:
        """Hedge without checking anything - the vanilla-CAI failure mode."""
        draft = ctx.get("draft", "")
        critique = ctx.get("critique_text", "")
        if '"verdict": "ok"' in critique:
            return draft
        sents = split_sentences(draft) or [draft]
        hedged = []
        for i, s in enumerate(sents):
            s = s.strip()
            if i == 0 and s.lower().startswith("yes, that's right."):
                s = s[len("Yes, that's right."):].strip()
                if not s:
                    continue
            hedged.append(s)
        body = " ".join(hedged)
        return f"As far as I recall: {body} I am not certain of the exact figures."

    def _revise_grounded(self, ctx: dict) -> str:
        """Answer out of the evidence, or say the evidence does not answer it.

        Relevance is judged per passage and the answer is quoted as a short
        *span* rather than as the single best-matching sentence. Both choices
        come from the same observation: the sentence that corrects a false
        premise is usually not the sentence that matches it. "The Cassell
        Incident" scores high on the draft's wording; "No crew were killed" -
        the sentence that actually refutes it - scores near zero and would never
        be selected on its own.
        """
        question = ctx.get("question", "")
        evidence = ctx.get("evidence", []) or []
        if not content_tokens(question) or not evidence:
            return "The retrieved evidence does not answer this question."

        draft_claims = extract_claims(ctx.get("draft", ""))

        def relevance(sent: str) -> float:
            rel = coverage(question, sent)
            for claim in draft_claims:
                rel = max(rel, coverage(claim, sent))
            return rel

        ranked = []
        for p in evidence:
            sents = [s for s in split_sentences(p.text) if content_tokens(s)]
            if not sents:
                continue
            scores = [relevance(s) for s in sents]
            best = max(range(len(sents)), key=lambda i: scores[i])
            ranked.append((scores[best], p.id, sents, best))
        ranked.sort(key=lambda t: (-t[0], t[1]))

        # A retriever always returns its top k, relevant or not. Without a floor
        # the reviser would dress the nearest passage up as an answer - which is
        # the failure the unanswerable items exist to catch.
        if not ranked or ranked[0][0] < _RELEVANCE_FLOOR:
            return (
                "The retrieved evidence does not answer this question, so I will not "
                "guess. Nothing in the corpus states it."
            )

        parts: list[str] = []
        for score, pid, sents, best in ranked[:2]:  # two passages, for multi-hop
            if score < _RELEVANCE_FLOOR or len(parts) >= 3:
                break
            for sent in sents[best : best + 2]:
                if len(parts) >= 3:
                    break
                parts.append(f"{sent} [{pid}]")
        answer = " ".join(parts)

        # If the draft asserted something the evidence does not, say so first
        # (P4 and P8): a silent correction teaches nothing.
        contradicted = any(
            not check_claim_default(claim, evidence)[0] for claim in draft_claims
        )
        if contradicted or self._is_leading(question):
            answer = "That is not what the evidence says. " + answer
        return answer

    # -- dispatch ----------------------------------------------------------

    def generate(self, messages, *, stage="", ctx=None, tools=None, max_new_tokens=None, temperature=None):
        ctx = dict(ctx or {})
        ctx.setdefault("question", messages[-1]["content"] if messages else "")
        if stage == "draft":
            return self._draft(ctx)
        if stage == "tool":
            return self._tool(ctx)
        if stage == "claims":
            return "\n".join(extract_claims(ctx.get("draft", "")))
        if stage == "critique":
            return self._critique_grounded(ctx) if ctx.get("grounded") else self._critique_plain(ctx)
        if stage == "revise":
            return self._revise_grounded(ctx) if ctx.get("grounded") else self._revise_plain(ctx)
        if stage == "judge":
            return "SUPPORTED"
        return self._draft(ctx)


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------

BACKENDS = ("scripted", "openai", "vllm", "transformers", "echo")


def get_backend(name: str = "scripted", **kwargs) -> LLM:
    name = (name or "scripted").lower()
    if name == "scripted":
        return ScriptedBackend(**kwargs)
    if name in ("openai", "vllm"):
        return OpenAICompatBackend(**kwargs)
    if name == "transformers":
        return TransformersBackend(**kwargs)
    if name == "echo":
        return EchoBackend(**kwargs)
    raise ValueError(f"unknown backend {name!r}; choose from {', '.join(BACKENDS)}")


def add_backend_args(parser) -> None:
    """Shared CLI flags, so every entry point selects a model the same way."""
    parser.add_argument("--backend", default=os.environ.get("GROUNDLOOP_BACKEND", "scripted"),
                        choices=list(BACKENDS), help="model backend (default: scripted)")
    parser.add_argument("--model", default=None, help="model id or local path")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint")
    parser.add_argument("--adapter", default=None, help="LoRA adapter path (transformers backend)")
    parser.add_argument("--thinking", action="store_true",
                        help="enable Qwen thinking mode (transformers backend)")
    parser.add_argument("--dtype", default="auto",
                        choices=("auto", "float16", "bfloat16", "float32"),
                        help="weight dtype (transformers backend). Use float16 on a T4.")


def backend_from_args(args) -> LLM:
    kwargs: dict[str, Any] = {}
    if args.backend in ("openai", "vllm"):
        if args.model:
            kwargs["model"] = args.model
        if getattr(args, "base_url", None):
            kwargs["base_url"] = args.base_url
    elif args.backend == "transformers":
        if args.model:
            kwargs["model_id"] = args.model
        if getattr(args, "adapter", None):
            kwargs["adapter"] = args.adapter
        kwargs["thinking"] = bool(getattr(args, "thinking", False))
        kwargs["dtype"] = getattr(args, "dtype", "auto")
    return get_backend(args.backend, **kwargs)

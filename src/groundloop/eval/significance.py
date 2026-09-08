"""Is the gap real, or is it seven examples?

The QA set has 39 items. Accuracy moving from 10.3% to 28.2% is four items
becoming eleven, and a table that reports "2.7x better" without saying so
invites a claim the data cannot carry.

Two instruments, both stdlib:

* **McNemar's exact test** for the paired comparisons. The conditions answer the
  *same* questions, so what matters is not two independent proportions but the
  items that flipped: how many the second condition got right that the first got
  wrong, against how many went the other way. Under the null those are a coin
  flip, so the p-value is an exact binomial tail. Unpaired tests here would
  throw away the pairing and be needlessly conservative.

* **A percentile bootstrap** for the interval around a single rate, which is
  what tells you that 28.2% on 39 items is worth about plus or minus fifteen
  points and should be written down that way.
"""

from __future__ import annotations

import math
import random

SEED = 20260901


def mcnemar(before: list[bool], after: list[bool]) -> dict:
    """Exact McNemar on paired binary outcomes.

    `before` and `after` are aligned per example. Returns the discordant counts
    and a two-sided p-value; the concordant pairs carry no information about a
    difference and drop out, which is the whole point of the test.
    """
    if len(before) != len(after):
        raise ValueError("paired comparison needs the same examples in both conditions")
    gained = sum(1 for b, a in zip(before, after) if a and not b)
    lost = sum(1 for b, a in zip(before, after) if b and not a)
    n = gained + lost
    if n == 0:
        return {"gained": 0, "lost": 0, "p_value": 1.0, "n_discordant": 0}

    # Two-sided exact binomial: P(X <= k) * 2 under p = 0.5, capped at 1.
    k = min(gained, lost)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {"gained": gained, "lost": lost, "n_discordant": n, "p_value": round(min(1.0, 2 * tail), 4)}


def bootstrap_ci(flags: list[bool], iterations: int = 2000, alpha: float = 0.05,
                 seed: int = SEED) -> tuple[float, float]:
    """Percentile bootstrap interval for a rate, in percent."""
    if not flags:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(flags)
    rates = []
    for _ in range(iterations):
        sample = [flags[rng.randrange(n)] for _ in range(n)]
        rates.append(100.0 * sum(sample) / n)
    rates.sort()
    lo = rates[int((alpha / 2) * iterations)]
    hi = rates[min(iterations - 1, int((1 - alpha / 2) * iterations))]
    return (round(lo, 1), round(hi, 1))


def compare(scores_by_condition: dict[str, list], attribute: str, higher_is_better: bool) -> dict:
    """Paired comparisons of one boolean attribute across the three conditions.

    Scores must be aligned by example; they are, because every condition runs
    the same dataset in the same order through one function.
    """
    flags = {
        cond: [bool(getattr(s, attribute)) for s in scores]
        for cond, scores in scores_by_condition.items()
    }
    out: dict = {"attribute": attribute, "higher_is_better": higher_is_better, "conditions": {}}
    for cond, values in flags.items():
        lo, hi = bootstrap_ci(values)
        out["conditions"][cond] = {
            "rate": round(100.0 * sum(values) / len(values), 1) if values else 0.0,
            "ci95": [lo, hi],
            "n": len(values),
        }
    if "groundloop" in flags:
        out["vs_base"] = mcnemar(flags.get("base", []), flags["groundloop"]) if "base" in flags else {}
        out["vs_plain_critique"] = (
            mcnemar(flags["plain_critique"], flags["groundloop"]) if "plain_critique" in flags else {}
        )
    return out


LABELS = {
    "hallucinated": "hallucination rate",
    "correct": "accuracy",
    "pushback": "sycophancy pushback",
}


def render_markdown(comparisons: list[dict]) -> list[str]:
    lines = [
        "", "## Is the gap real?", "",
        "95% intervals are a percentile bootstrap; p-values are McNemar's exact",
        "test on the paired per-item outcomes (the conditions answer the same",
        "questions, so only the items that flipped carry information).",
        "",
        "| Metric | Base | Plain critique | GroundLoop | GroundLoop vs base | vs plain critique |",
        "|---|---|---|---|---|---|",
    ]
    for comp in comparisons:
        conds = comp["conditions"]

        def cell(name):
            c = conds.get(name)
            return f"{c['rate']}% [{c['ci95'][0]}–{c['ci95'][1]}]" if c else "-"

        def verdict(key):
            t = comp.get(key) or {}
            if not t or not t.get("n_discordant"):
                return "no change"
            # "gained" means the flag turned on, which is an improvement only
            # when higher is better. Reporting it raw inverts the sign on
            # hallucination rate, where turning the flag on is the regression.
            if comp["higher_is_better"]:
                improved, regressed = t["gained"], t["lost"]
            else:
                improved, regressed = t["lost"], t["gained"]
            return f"{improved} better / {regressed} worse, p={t['p_value']}"

        label = LABELS.get(comp["attribute"], comp["attribute"].replace("_", " "))
        lines.append(
            f"| {label} | {cell('base')} | {cell('plain_critique')} | {cell('groundloop')} | "
            f"{verdict('vs_base')} | {verdict('vs_plain_critique')} |"
        )
    lines += [
        "",
        "\"better/worse\" counts the items that flipped in each direction. At",
        "n=39 a p-value above ~0.05 means the table shows a direction, not a",
        "result - report it that way.",
    ]
    return lines

# The evidence corpus

`passages.jsonl` is the entire world the `search(query)` tool can see. It is
fixed and local so that a run today is comparable to a run after fine-tuning,
and so that an evaluation cannot silently change when someone else's index
does.

Each line is `{id, world, title, text, tags}`.

## Two slices, on purpose

**`world: "real"` (ids `r01`-`r24`)** — well-established facts. These test
ordinary grounding, and they are where the model's prior is usually *right*,
which matters: a method that only ever deletes claims would score well on
faithfulness and be useless.

**`world: "fictional"` (ids `f01`-`f20`)** — the Meridian Archive, an invented
setting (a space station, KS-9, and the consortium that runs it). Every one of
these passages is made up. They are titled `[MERIDIAN ARCHIVE - FICTIONAL]` so
that nothing here can be mistaken for a claim about the real world.

The fictional slice buys two things a real-world corpus cannot:

1. **A clean hallucination signal.** The model has no prior about KS-9, so any
   specific claim it makes that is not in a passage was invented on the spot.
   There is no "well, it might have read that somewhere" defence.
2. **Safe evidence-vs-prior conflicts.** A KS-9 day is 31 hours; the outer ring
   runs at 0.38 g. A model that answers "24 hours" or "1 g" is applying a
   terrestrial prior over the source in front of it — which is exactly the
   behaviour principle P4 targets — without anyone having to write down a false
   statement about the real world to test it.

The internal numbers are consistent (214 crew = 96 + 71 + 47; six lifeboats at
40 each covers all 214), so multi-hop questions have checkable answers.

## Adding to it

Append lines with fresh ids and re-run anything downstream — the BM25 index is
rebuilt from the file on every load, there is nothing to migrate. If you swap in
a HotpotQA or NQ subset, keep the `{id, title, text}` shape and the rest of the
pipeline will not notice.

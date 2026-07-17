# Microlib: WorldFoam Gate4 Owner/Candidate Records

## Problem

Improve the current Gate4 high-cap warm fused-MSE path by removing owner scan,
candidate replay, or row-construction costs without breaking train/eval
quality or the status verifier.

## Why Now

WorldFoam Gate4 has many recorded negatives. The surviving direction is narrow:
owner-run and endpoint-record style compression can be useful, but standalone
probes and setup-only wins are not enough.

## Allowed Edits

Likely surface:

- `research_experiments/world_foam_lane2/`
- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/`
- focused tests under `research_experiments/world_foam_lane2/`

Do not touch token-GS or STAR trainers for this microlib.

## Baseline

Current known direction:

- inline owner-run reverse-tape is the keeper direction but still misses the
  formal scale verifier in older notes
- endpoint-record/direct-delta/vectorized coefficient work improved setup and
  some warm paths, but several native packing variants regressed full timing
- full train/eval beats standalone spots for promotion decisions

## Evaluator Cascade

Stage 0:

- extension builds/imports
- changed paths in lane only

Stage 1:

- focused regression tests for row/candidate counts
- verifier checks exact tensor equality where a packing op replaces Python

Stage 2:

- full train/eval small frame matrix
- JSON includes total/backward/render scales and selected tape storage scale
- PSNR unchanged within tolerance

Stage 3:

- `verify_fused_slab_status_summary.py` or equivalent top-level status verifier
- comparison against current owner-run/endpoint-record rows

## Primary Metrics

- status `ok`
- total and backward scale lower than current selected direction
- quality unchanged
- setup improvements reported separately from warm kernel improvements
- storage scale sublinear where claimed

## Hard Rejects

- Promoting single-row VJP spots without full train/eval.
- Changing the quality target.
- Hiding setup time as warm-kernel speed.
- Ignoring MPS lifetime/order sensitivity warnings.
- Repeating known negative packing variants without a new hypothesis.

## Promotion Gate

Promotion needs a full train/eval artifact and a verifier result. A faster
standalone kernel is only a hint.

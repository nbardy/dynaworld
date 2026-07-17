# Gate4 Candidate CSR 64px Capacity Probe

## Context

We paused the clean timing promotion because the benchmark environment remained
contended by unrelated `ai_trader` Python/pytest work. I added a CPU-only
structural probe so we can still separate "the candidate topology scales" from
"the shader is fast under clean timing."

## Code added

- `research_experiments/world_foam_lane2/probe_gate4_affine_candidate_csr_capacity.py`
  builds the Gate4 affine candidate CSR tape on CPU for a frame-count ladder and
  reports MPS-resident-equivalent storage using the same key layout as
  `train_eval_owner_run_tape.py`.
- `research_experiments/world_foam_lane2/test_probe_gate4_affine_candidate_csr_capacity.py`
  covers the storage layout and the acceptance gates for flat topology vs
  candidate/cap growth.

The probe explicitly reports `speed_claim: false` and
`timing_scope: cpu_build_timings_are_diagnostic_only_not_speed_claims`.

## 64px result

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/probe_gate4_affine_candidate_csr_capacity.py \
  --frame-counts 2,4,8,16 \
  --render-size 64 \
  --site-count 24 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_csr_capacity64_2_4_8_16_site24.json
```

Artifact:

- `research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_csr_capacity64_2_4_8_16_site24.json`

Result: `status=ok`.

Important counters:

- frame scale: `2 -> 16`, `8.0x`
- direct boundary iteration scale: `8.0x`
- compiled boundary test scale: `1.0x`
- candidate count scale: `0.9919x`
- storage scale: `0.9922x`
- candidate replay iteration scale: `7.9355x`
- max candidates per row: `222, 217, 215, 216` under the `256` row cap
- loaded frames were real for each count; `repeat_loaded_frames=false`

This proves the candidate CSR topology/storage is flat at 64px across the
2/4/8/16 real-frame ladder. It does not prove runtime sublinearity because the
per-frame replay work still scales and the timing environment is dirty.

## Tests

Passed:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/probe_gate4_affine_candidate_csr_capacity.py \
  research_experiments/world_foam_lane2/test_probe_gate4_affine_candidate_csr_capacity.py
```

Passed:

```bash
PYTHONPATH=research_experiments/world_foam_lane2 \
  PYTHONDONTWRITEBYTECODE=1 uv run --with pytest python -m pytest \
  research_experiments/world_foam_lane2/test_probe_gate4_affine_candidate_csr_capacity.py -q
```

`3 passed in 1.25s`.

Passed:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  PYTHONDONTWRITEBYTECODE=1 uv run --with pytest python -m pytest \
  research_experiments/world_foam_lane2/test_probe_gate4_affine_candidate_csr_capacity.py \
  research_experiments/world_foam_lane2/test_run_gate4_affine_candidate_csr_promotion_gate.py \
  research_experiments/world_foam_lane2/test_verify_gate4_affine_candidate_csr_train_eval.py \
  research_experiments/world_foam_lane2/test_compare_star_uvt_worldfoam_scale.py -q
```

`15 passed in 1.69s`.

## Current blocker

The clean timing promotion is still blocked by environment contention. Latest
preflight still reported `status=contended`, with high CPU from unrelated
`ai_trader` verification/pytest jobs. Do not cite the 64px probe CPU timing as
shader speed. Next step when clean: run
`run_gate4_affine_candidate_csr_promotion_gate.py` for the 64px 2/4/8/16 ladder
or extend the runner to accept the already-proven 64px topology gate before
timing.

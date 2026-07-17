# Gate4 Candidate CSR Promotion Preflight

We paused shader iteration long enough to make the Gate4 affine candidate CSR
promotion gate produce compact, auditable evidence.

Changed:

- `research_experiments/world_foam_lane2/run_gate4_affine_candidate_csr_promotion_gate.py`
  now compacts benchmark-environment process snapshots to `pid`, `pcpu`,
  `pmem`, and `command`, plus process counts.
- The same runner now passes and records `endpoint_record_source:
  gate4-affine`; candidate mode already built `Gate4AffineSlabTape`, but the
  old `slow-owner-run` argument made the promotion command misleading.
- `research_experiments/world_foam_lane2/test_run_gate4_affine_candidate_csr_promotion_gate.py`
  now covers the compact snapshot shape and the explicit `gate4-affine`
  source contract.

Verification:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/run_gate4_affine_candidate_csr_promotion_gate.py \
  research_experiments/world_foam_lane2/test_run_gate4_affine_candidate_csr_promotion_gate.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  PYTHONDONTWRITEBYTECODE=1 uv run --with pytest python -m pytest \
  research_experiments/world_foam_lane2/test_run_gate4_affine_candidate_csr_promotion_gate.py \
  research_experiments/world_foam_lane2/test_verify_gate4_affine_candidate_csr_train_eval.py -q
```

Result: `8 passed in 1.69s`; after the explicit source-contract fix, the same
focused suite passed again with `8 passed in 0.87s`.

Preflight:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/run_gate4_affine_candidate_csr_promotion_gate.py \
  --run-id 2026-05-19_gate4_affine_candidate_csr_preflight_blocked \
  --no-wait-for-benchmark-environment-ok \
  --out-summary research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_csr_preflight_blocked.promotion_summary.json
```

Result:

- status: `preflight_blocked`
- attempt_count: `0`
- endpoint_record_source: `gate4-affine`
- benchmark environment: `contended`
- latest blocking process count: `2`
- latest blockers were high-CPU `ai_trader`
  `report_btc15m_feature_pack_availability.py` jobs

Interpretation:

The candidate CSR path remains the first WorldFoam line with the right
STAR-like structural shape, but this turn did not produce a clean promotion.
Do not cite the prior allow-contended 2/4/8/16 timing as promoted evidence.
The next real action is to rerun the promotion runner when the benchmark
environment reaches `background` or `ok`.

I also inspected the proposed "zero-storage rowdesc" fork point. That idea is
effectively already represented by the `rowselect32` path: each frame lane
selects its row from chunk offsets instead of using stored `row_begin_i32` /
`row_len_source_i16`. The prior rowselect32 promotion failed because the
remaining delta-record topology still scaled, not because rowdesc storage was
the only issue. Do not spend another fork duplicating rowselect32; continue
with candidate CSR or another representation that removes scaling
`delta_change_record_i32` / `change_frame_i32` / `change_offsets_i32` residency.

Follow-up in the same lane:

- `compare_star_uvt_worldfoam_scale.py` now understands
  `gate4-affine-candidate-num32-den16-fused-mse` artifacts in addition to the
  older `fused_mse_rgb_only` WorldFoam artifacts. It records the WorldFoam
  family as `gate4_affine_candidate_csr`, uses selected resident non-coeff
  storage as the compact tape storage, records `affine_ray_f32` separately
  when present, and exposes candidate count/max-candidate rows.
- `verify_gate4_affine_candidate_csr_train_eval.py` now treats single-frame
  smoke artifacts as shape/capacity checks: it keeps all row, gradient,
  update, PSNR, storage, and candidate-cap checks, but skips the sublinear
  runtime claim unless the requested frame ladder actually increases.

64px shape/capacity smoke:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --tape-mode gate4-affine-candidate-num32-den16-fused-mse \
  --endpoint-record-source gate4-affine \
  --frame-counts 2 \
  --render-size 64 \
  --site-count 24 \
  --optimizer-mode manual-vjp \
  --steps 1 \
  --warmup-steps 0 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_csr_smoke64_2f_site24_1step.json
```

Result:

- status: `ok`
- render/site/frame: `64px`, `24` sites, `2f`
- max candidates/row: `222`, under the fused-MSE cap `256`
- candidate_count: `1,339,555`
- gradients and parameter update were nonzero
- benchmark environment was contended, so do not use the timing as clean speed
  evidence

Verifier:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_gate4_affine_candidate_csr_train_eval.py \
  research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_csr_smoke64_2f_site24_1step.json \
  --frame-counts 2 \
  --render-size 64 \
  --site-count 24 \
  --min-train-psnr 0 \
  --min-heldout-psnr 0 \
  --allow-contended \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_csr_smoke64_2f_site24_1step.verify_allow_contended.json
```

Result: `status ok`, `scale_gate_required=false`, contamination recorded as
`benchmark_environment status is 'contended'`.

Focused test suite:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  PYTHONDONTWRITEBYTECODE=1 uv run --with pytest python -m pytest \
  research_experiments/world_foam_lane2/test_run_gate4_affine_candidate_csr_promotion_gate.py \
  research_experiments/world_foam_lane2/test_verify_gate4_affine_candidate_csr_train_eval.py \
  research_experiments/world_foam_lane2/test_compare_star_uvt_worldfoam_scale.py -q
```

Result: `12 passed in 3.97s`.

# WorldFoam render96/site48 i32-base-offset gate and ai_trader export check

Context: after fixing the framebitmask `base_offsets_i32` path for the
render96/site48 smoke, we needed a clean larger fused-MSE WorldFoam-vs-STAR
gate. The user also asked whether the visible `ai_trader` export was stuck.

What the ai_trader process was:

- Live detached screen:
  `79267.toto_floor001_guardaligned_20260519T200320Z`.
- Command: `/Users/nicholasbardy/git/ai_trader/scripts/run_btc15m_overnight_shadow_monitor.py`
  with `--toto-export-device mps` and `--toto-export-with-runtime-deps`.
- It is not a hung one-time export. It is the overnight BTC15M Toto shadow
  monitor emitting per-iteration prediction exports such as
  `logs/btc15m_shadow_overnight/btc15m_toto_context64_floor001_guardaligned_20260519T200320Z/iterations/0077/toto_residual_live_quote_shadow/toto_residual_live_prediction_export.parquet`.
- Current safety state stayed fail-closed: `report_only=true`,
  `no_training=true`, `shadow_only=true`, `orders_sent=false`,
  `status=blocked_preflight_live_quote_shadow`.
- It can still contaminate local timing windows. During WorldFoam attempt 1,
  the monitor spawned
  `scripts/export_btc15m_toto_residual_live_prediction_export.py`, which the
  benchmark environment checker captured at about `72%` CPU plus
  `MTLCompilerService` activity. The wrapper correctly rejected that attempt as
  non-promotable.

WorldFoam command:

```bash
PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python research_experiments/world_foam_lane2/run_worldfoam_star_native_cutwalk_gate.py \
  --run-id 2026-05-20_native_cutwalk_worldfoam_star_render96_site48_i32base_gate \
  --render-size 96 --site-count 48 --frame-counts 2,4,8 \
  --worldfoam-steps 4 --worldfoam-warmup-steps 2 \
  --star-target-size 96 --star-tube-count 1792 --star-steps 10 --star-warmup-steps 3 \
  --max-worldfoam-attempts 2 --max-star-attempts 2 \
  --preflight-stability-samples 2 --preflight-stability-interval-s 3 \
  --verify-promotion --wait-timeout-s 60 --wait-poll-s 5
```

Artifacts:

- Summary:
  `research_experiments/world_foam_lane2/results/2026-05-20_native_cutwalk_worldfoam_star_render96_site48_i32base_gate.promotion_summary.json`
- Promotable WorldFoam:
  `research_experiments/world_foam_lane2/results/2026-05-20_native_cutwalk_worldfoam_star_render96_site48_i32base_gate.attempt2.worldfoam.json`
- STAR comparison:
  `research_experiments/world_foam_lane2/results/2026-05-20_native_cutwalk_worldfoam_star_render96_site48_i32base_gate.star_attempt1.star_compare.json`

Outcome:

- Wrapper `status=ok`.
- WorldFoam attempt 1: `status=ok`, but `promotable=false`,
  `benchmark_environment_status=contended`, `returncode=2`.
- WorldFoam attempt 2: `status=ok`, `promotable=true`,
  `benchmark_environment_status=background`, `returncode=0`.
- Integrated promotion verifier: `status=ok`.
- STAR comparison: `status=ok`, benchmark environment `background`.

WorldFoam render96/site48 medians:

| Frames | total ms | backward ms | train PSNR | heldout PSNR | selected storage | base offset max |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 3.760 | 3.480 | 9.218 | 10.801 | 1,630,704 | 83,695 |
| 4 | 4.125 | 3.847 | 9.250 | 10.838 | 2,291,268 | 79,882 |
| 8 | 4.619 | 4.331 | 9.875 | 10.880 | 3,538,256 | 81,649 |

WorldFoam median total/backward scale from 2f to 8f is `1.229x/1.245x` over a
`4x` frame-count increase. The base offsets remain over int16 range, so this is
real coverage for the new `base_offsets_i32` framebitmask path.

Matched STAR 96px/1792-tube medians:

| Frames | total ms | backward ms | STAR/WorldFoam total | STAR/WorldFoam backward |
|---:|---:|---:|---:|---:|
| 2 | 5.773 | 3.614 | 1.535x | 1.038x |
| 4 | 7.583 | 5.161 | 1.838x | 1.342x |
| 8 | 9.692 | 6.719 | 2.098x | 1.551x |

Interpretation:

- This is real progress: the i32-offset framebitmask shader path is now
  correctness-covered at a larger render/site setting, and the clean attempt
  remains sublinear in measured WorldFoam step/backward timing.
- It is not full system parity. It is a fused-MSE Gate4 speed/scale gate with
  weak short-run PSNR and fixed/frozen geometry scope.
- Next useful evidence is a real longer-than-16f fixture or a quality-linked
  gate that connects this fused-MSE speed path back to the broader STAR RGB
  quality baseline. Another local replay micro-variant has diminishing value.
- For local timing gates, either pause/stop the ai_trader monitor or rely on
  strict retry wrappers. The Toto export is real and useful, but it can briefly
  use CPU/Metal enough to contaminate benchmark windows.

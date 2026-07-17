# STAR UVT Active Goal Continuation Audit

Date: 2026-05-20 03:37 +07

## Objective Under Audit

Active goal:

> Repeat and harden the STAR UVT fast feature-shader plan docs, fill any
> missing implementation details, then execute the plan gate by gate with
> benchmarks and progress logs recorded in markdown.

The broader thread plan also asked for rerunning renderer benchmarks, comparing
STAR UVT against dynamic gsplat at matched 512px/64f/8192 scale, building a fast
single-video overfit route, diagnosing real training bottlenecks, scaling to the
prepared 300-video set, carrying feature-splatting lessons into UVT STAR, and
keeping WorldFoam separate so it does not fight for GPU.

## Current Evidence

| Requirement | Evidence | Status |
| --- | --- | --- |
| Repeat and harden STAR UVT fast feature-shader docs | `research_experiments/star_uvt_feature_tubes/2026-05-18_fast_shader_port_plan.md`, `research_experiments/star_uvt_feature_tubes/README.md`, `PROJECT_INDEX.md`, `README.md`, `TODO/README.md`, `EXPERIMENTS.md`, and `BASELINES.md` now describe the selected shader route, rejected visual routes, alpha diagnostics, and sparse-F1 dense-alpha trainer gate. | Complete for the shader diagnostic slice |
| Fill missing implementation details | `render_uvt_feature_alpha_all_pixels_with_bins` reuses the sparse-pixel Metal F1 path for alpha-only visibility; `src/train/train_star_uvt_feature_overfit.py` exposes `dense_alpha.render_mode` with `dense_f32` and `sparse_f1`; `tests/test_star_uvt_feature_target_adapter.py` validates config normalization and invalid mode rejection. | Complete for the latest missing alpha-only implementation gap |
| Execute gates with benchmark artifacts | Current artifacts include selected visual-quality, RGB-grid, compact+RGB-grid, dense-alpha failure diagnostic, alpha-to-one, phase-alpha, black-hole, target-background, patch4/alpha sweep, raw-opacity bias, dense-alpha support, alpha-only visibility profile, sparse-F1 dense-alpha trainer gate, and dynamic-gsplat 512px comparator smoke reports under `outputs/benchmarks/`. | Complete for the run gates |
| Record progress logs in markdown | Each current gate has a dated loose note under `agent_notes/loose_notes/`; the latest sparse-F1 trainer gate is `2026-05-20_03-34-12_star_uvt_sparsef1_dense_alpha_trainer_gate.md`. | Complete for the run gates |
| Keep dense key learnings compressed | `agent_notes/key_learnings.md` is still 199 lines and compresses the STAR UVT lesson: alpha diagnostics are now cheap, but same-support alpha/grid pressure does not fix dense visibility. | Complete |
| Confirm the fastest practical alpha-only route | Alpha-only profile passes with exact alpha parity, geometry/opacity gradient parity within `4.7e-7`, zero overflow, and render+backward `1100.8 -> 634.6ms`; trainer integration cuts mean step/backward `2558.6/1114.2 -> 873.3/370.0ms`. | Complete |
| Decide whether sparse-F1 alpha rescues quality | The sparse-F1 trainer row preserves the dense F32 quality endpoint and remains pass-false: weighted loss `1.271702 -> 1.284505`, feature loss `0.625418 -> 0.626814`, RGB-probe PSNR `22.028 -> 21.861`, dense RGB `5.647`. | Complete; rejected as quality promotion |
| Start support-changing visibility bridge | `research_experiments/star_uvt_feature_tubes/visibility_support_bridge_prototype.py` passes the CPU proxy gate: from zero target hits, same-support dense alpha keeps target alpha `>0.10` coverage at `0.0`, while the soft projected-tube proxy sends center/velocity gradients and reaches `0.324`. | Complete as CPU mechanism gate; trainer/Metal still pending |
| Matched dynamic-gsplat comparator | `outputs/benchmarks/2026-05-20_dynamic_gsplat_512_matched_probe.md` records a fixed `64f/512px/8192` smoke at `8.019s` step / `5.638s` backward. The follow-up `outputs/benchmarks/2026-05-20_dynamic_gsplat_fixed512_20step_matched_media.md` records offline W&B media, final eval PSNR `5.587`, and mean timed step/backward `2.940/1.926s`. | Local fixed-512 comparator complete; longer dynamic-gsplat baseline still optional |
| Scale to prepared 300-video set | No new 300-video STAR UVT scale run was launched. Current docs explicitly block scale-up until dense media quality improves beyond sparse/streaked or blurry outputs. | Not complete; intentionally blocked |
| Feature-splatting sub-lane | Feature-splatting lessons were carried into the sparse-F1 alpha wrapper and dense-alpha trainer mode, but no separate new feature-splatting sub-agent/lane output was produced in this continuation. | Partially complete |
| WorldFoam side lane | WorldFoam remains documented as separate and not GPU-contending; no new WorldFoam shader work was executed in this continuation. | Not complete for the broader side-lane request |

## Completion Decision

Do not mark the full active goal complete yet.

The STAR UVT fast feature-shader diagnostic slice is now in a good closeout
state: code hooks, config, tests, benchmark artifacts, markdown reports, routing
docs, and key learnings are updated. The strongest new result is real speed
progress: sparse-F1 dense-alpha reduces the expensive alpha diagnostic from
`2.56s/step` to `0.87s/step`.

The full thread goal remains open because the broader plan still includes
scale-up and side-lane execution that current evidence does not prove:

- no 300-video STAR UVT scale run has been launched;
- matched dynamic-gsplat now has a local 20-step media comparator, but not a
  long/full dynamic-gsplat baseline;
- feature-splatting carry-forward is represented by the sparse-F1 alpha wrapper
  but not a separate feature-world-tube lane;
- WorldFoam has not received a new no-GPU-conflict investigation in this
  continuation.

## Next Experiments

The current-state closeout at
`agent_notes/loose_notes/2026-05-20_03-46-42_star_uvt_current_state_and_next_decision.md`
records the wrap/continue decision in one place. The short version: the shader
diagnostic phase is closeable, but the broad goal is still open unless it is
narrowed to that phase.

1. **Visibility/support bridge gate.** Implement a support-changing STAR UVT
   objective or model path and rerun the selected dense media gate. Success
   should improve dense full RGB materially above the current `5.6-6.0` PSNR
   band without regressing feature/probe quality. The CPU proxy gate now proves
   the mechanism; the remaining work is first-class trainer integration and a
   real dense media gate.
2. **Optional longer dynamic-gsplat baseline.** The fixed `64f/512px/8192`
   comparator now has media and final eval metrics. Only spend on a longer
   dynamic-gsplat run if the question becomes final dynamic-gsplat ranking, not
   current STAR UVT routing.
3. **300-video scale gate only after visual quality clears.** Use the prepared
   300-video set only after a single-video STAR UVT dense-media gate is no
   longer sparse/streaked or blurry.
4. **Feature-world-tube lane.** Promote the sparse-F1 alpha wrapper lesson into
   a separate feature-splatting/UVT tube note or experiment if that lane is
   still desired.
5. **WorldFoam lane.** Keep it separate from STAR UVT GPU benchmarking; run only
   CPU/docs/shader-note work unless there is a clean benchmark window.

## Validation State

Latest validation after sparse-F1 trainer integration:

```bash
rtk .venv/bin/python -m py_compile \
  src/train/train_star_uvt_feature_overfit.py \
  third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/feature_rasterize.py \
  research_experiments/star_uvt_feature_tubes/alpha_only_visibility_profile.py
```

```bash
PYTHONPATH=src/train rtk uv run --with pytest python -m pytest \
  tests/test_star_uvt_feature_target_adapter.py -q
# 30 passed
```

```bash
rtk wc -l agent_notes/key_learnings.md
# 199
```

```bash
rtk git diff --check
rtk git -C third_party/fast-mac-gsplat diff --check
```

All passed before this continuation audit was written.

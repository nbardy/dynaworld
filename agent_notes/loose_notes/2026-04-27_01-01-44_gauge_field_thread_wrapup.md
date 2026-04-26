# Gauge-Field Thread Wrap-Up

## Context

This thread started from a chief-scientist critique: the current gauge-field
implementation should not be defended as a novel 3D primitive. It is a useful
diagnostic harness built around persistent transported material elements, but
the renderer began as projected disks and could still behave like source-view
video painting.

The practical goal became:

```text
keep the harness,
make support representations swappable,
test source-view vs held-out-camera behavior,
compare against direct free dynamic splats,
and leave a clean planning trail for the next representation search.
```

## Initial Goals

- Implement three support laws side by side:
  - `screen_disk`
  - `oriented_slab`
  - `rank_adaptive_metric`
- Keep the current projected disk as baseline/control, not as final claim.
- Preserve shared RGB/alpha/depth/X-map diagnostics.
- Add a held-out-camera lane using the new multicamera validation loader.
- Compare a direct splat baseline on the same data and held-out view.
- Answer parameter-count and speed fairness questions.
- Write durable docs so the next agent does not have to reconstruct the thread.

## Subgoals That Evolved

- Treat source-view PSNR as a weak diagnostic only, after the first support-mode
  benchmark showed close scores and the user pushed for novel-view metrics.
- Use the DeepView 03_Dog source/target camera pair as the first calibrated
  held-out lane.
- Add the direct free-dynamic 3DGS control after the user asked whether splats
  should be compared on the same data.
- Add active/effective parameter accounting because `free_dynamic_3dgs` has far
  more trainable state at the same displayed primitive count.
- Add a reusable mathematical web-of-thought prompt because the discussion moved
  from implementation toward representation search.
- Split gauge-field shared helpers into `common.py` and `data.py` after both the
  gauge trainer and splat baseline needed the same loading, path, metric, and
  artifact code.

## Work Completed

- Gauge support modes exist in `research_experiments/gauge_fields/train.py`.
- The support-mode handoff exists at:
  - `research_experiments/gauge_fields/SUPPORT_MODE_ABLATION_HANDOFF.md`
- The direct free-dynamic splat baseline exists at:
  - `research_experiments/gauge_fields/train_splat_baseline.py`
- Shared helper modules now exist:
  - `research_experiments/gauge_fields/common.py`
  - `research_experiments/gauge_fields/data.py`
- The mathematical web-of-thought prompt exists at:
  - `research_notes/meta_philosophy/mathematical_web_of_thought_prompt.md`
- The prompt is indexed in:
  - `research_notes/meta_philosophy/README.md`
- Current gauge/prompt commits:
  - `0eead59 Add gauge held-out camera ablation lane`
  - `5944f8f Add gauge math prompt and shared helpers`

## Results Captured

Source-view support-mode benchmark:

```text
outputs/gauge_fields/support_mode_benchmark_250step/summary.md
```

Result:

```text
oriented_slab source PSNR: 20.6312
screen_disk source PSNR: 20.3144
rank_adaptive_metric source PSNR: 19.8992
```

DeepView held-out-camera benchmark:

```text
outputs/gauge_fields/multicam_deepview_support_mode_benchmark_80step/summary.md
```

Result:

```text
free_dynamic_3dgs heldout PSNR: 9.7392
screen_disk heldout PSNR: 9.6479
rank_adaptive_metric heldout PSNR: 9.5662
oriented_slab heldout PSNR: 9.3344
```

Main empirical lesson:

```text
source-view ranking and held-out-camera ranking disagreed.
```

That is the thread's most important experimental result.

## Issues Found And Solved

- Source-view PSNR was not enough. The work moved to a held-out-camera lane.
- DeepView calibration needed conversion into the gauge renderer's source-relative
  +Z camera convention.
- The first calibrated lane still uses a pinhole approximation for fisheye
  DeepView cameras; this is documented as a caveat, not treated as final truth.
- Direct splat eval initially risked holding graph state through final renders;
  `render_splat_sequence` is now `@torch.no_grad()`.
- Same primitive count was not a fair parameter-count comparison. The handoff now
  records active/effective parameter counts and matched element counts.
- End-to-end speed numbers mixed training/eval/media time. They are documented
  as coarse timings, not pure renderer timings.
- The helper split briefly broke the gauge trainer because local `write_json`
  still needed `json`. A 1-step CPU smoke caught and fixed it before commit.
- The splat baseline still depended on helper imports through `train.py`; it now
  imports shared helpers directly from `common.py` and `data.py`.

## Issues Still Remaining

- The repo has many unrelated dirty files outside the committed gauge scope.
- Generated benchmark artifacts under `outputs/` are untracked. They need an
  artifact/ignore policy before repo cleanup.
- The parent `gsplats_browser` repo sees `dynaworld` as modified because the
  submodule commit advanced.
- `third_party/fast-mac-gsplat` is modified and was not touched in this thread.
- Held-out evidence is still one scene and one target camera.
- DeepView fisheye is still approximated as pinhole.
- No witness-rank / multi-ray concurrence metric exists yet.
- No flow-based X-map consistency metric is wired into the benchmark table yet.
- No camera-perturbation view-stress harness exists yet.
- No clean render-only/train-step/eval-time benchmark exists yet.
- `rank_adaptive_metric` has no spectrum/rank regularization yet.
- `oriented_slab` has no explicit thickness/rank prior yet.
- Phase-conditioned visibility is still a research target, not an implementation.

## Current Position

The current codebase now has a useful comparison harness:

```text
screen disk vs oriented slab vs rank-adaptive metric vs free dynamic 3DGS
```

The current evidence does not prove a clean 3D primitive. It proves the harness
can expose disagreement between source-view overfit and held-out-camera behavior.

The safest claim:

```text
We are ready to run representation ablations honestly.
We are not ready to claim that any current support mode solves 3D consistency.
```

## What To Do Next

Use `research_experiments/gauge_fields/NEXT_PLAN.md` as the operational plan.

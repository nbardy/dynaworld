# 2026-05-25 15:15:32 - Gauged UVT Media Tether

## Context

The prior source-distinct quality tether compared saved scalar payloads from the
multiscene frame-scaling matrix. That proved the measured live-cache route
matched cadence on loss curves, RGB loss curves, end PSNR, and gradient-flow
flags, but it did not touch the actual rendered media path.

## What changed

Added a focused media tether:

```text
research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_media_tether_report.py
tests/test_star_uvt_projective_real_video_multiscene_media_tether_report.py
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_media_tether/summary.json
```

The report runs the actual `star_uvt_feature_overfit_trainer.run_training`
media path with `contact_sheet_mode = "linspace"` because the underlying
writer accepts `first` and `linspace`, not `grid`. It runs the same three
source-distinct checked-in segments as the multiscene matrix, with one cadence
and one measured row per scene, emits contact sheets, and compares the cached
route against the cadence full-rebuild route at the pixel level.

## Result

The saved report is `status = ok`:

- 3 source-distinct scenes, 6 case rows, 3 cadence/measured media pairs.
- Contact sheets exist for every case.
- Contact-sheet pixel delta is exactly `0`.
- PNG hashes match for every cadence/measured pair.
- Final full-RGB media loss delta is `0.0`.
- Max loss-curve/RGB-loss-curve delta is `7.450580596923828e-09`.
- Min measured PSNR gain is `0.04511058330535889`.
- All required gradient-flow flags are present.
- Max measured/cadence no-first-step ratio is `0.48836477125817457`.
- Measured/cadence rebuild ratio is `0.5`.
- Overflow, fallback marks, and visibility stratifications are all zero.

This is stronger than the scalar quality tether because it proves the actual
rendered media artifact path is cadence-equivalent on the focused checked-in
matrix. It is still not broad real-scene quality acceptance.

## Audit Wiring

The top-level goal-progress audit now includes:

```text
real_video_multiscene_media_tether
```

The regenerated audit remains `status = in_progress`, proves 27 rows, and keeps
`full_goal_completion` open. The open gap now explicitly says the current proof
is focused artifacts plus real-video matrices, quality tether, and media
tether, not broad real-scene renderer acceptance.

## Validation

Commands run:

```text
.venv/bin/python -m py_compile research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_media_tether_report.py
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 uv run --with pytest python -m pytest tests/test_star_uvt_projective_real_video_multiscene_media_tether_report.py -q
.venv/bin/python research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_media_tether_report.py
.venv/bin/python research_experiments/star_uvt_feature_tubes/projective_goal_progress_audit.py
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 uv run --with pytest python -m pytest tests/test_star_uvt_projective_goal_progress_audit.py tests/test_star_uvt_projective_real_video_multiscene_media_tether_report.py -q
```

Final focused pytest result:

```text
40 passed in 9.78s
```

## Next

The next proof layer should not add another scalar wrapper. Useful next steps
are broader scene coverage, image/error acceptance against a dense reference, or
moving the media-quality tether into a longer trainer run where visual quality
is meaningful enough to inspect rather than just cadence-equivalent.

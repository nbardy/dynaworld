# Train Artifacts JSON Helpers

## Context

`src/train/train_artifacts.py` already owned `resolved_config.json` writing and
JSONL history appends for PowerFoam-family trainers. The V-JEPA performance
benchmarks and STAR alpha-background ablation orchestrator still repeated
parent-directory creation plus sorted JSON/JSONL output loops.

## Change

- Added `write_json(path, payload, sort_keys=True)`.
- Added `write_jsonl(path, rows)`.
- Added `write_text(path, text)`.
- Made `append_jsonl(...)` create the parent directory too.
- Routed result artifact writes through those helpers in:
  - `research_experiments/vjepa_performance/benchmark_fast_mac_variants.py`
  - `research_experiments/vjepa_performance/benchmark_free_splats_throughput.py`
  - `research_experiments/vjepa_performance/benchmark_multicam_vjepa.py`
  - `research_experiments/vjepa_performance/profile_fast_mac_render_phases.py`
  - `research_experiments/vjepa_performance/compare_fast_mac_quality.py`
  - `research_experiments/star_uvt_feature_tubes/run_alpha_background_ablation.py`
- Added tests for `write_json(...)` and overwrite-style `write_jsonl(...)`.
- Routed STAR UVT row-output file writes through `write_json(...)` as well;
  the STAR helper still owns stdout row formatting.
- Routed STAR UVT diagnostic/report JSON/markdown artifacts through
  `write_json(...)` / `write_text(...)` in:
  - `research_experiments/star_uvt_feature_tubes/background_cheat_diagnostic.py`
  - `research_experiments/star_uvt_feature_tubes/firstclass_backward_breakdown.py`
  - `research_experiments/star_uvt_feature_tubes/firstclass_scale_report.py`

`compare_fast_mac_quality.py` and `benchmark_fast_mac_variants.py` still append
each row as it completes, preserving the useful streaming behavior for long
variant sweeps. They now initialize the file via `write_jsonl(path, ())` and
append rows via `append_jsonl(...)`.

## Validation

Validation passed:

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/train_artifacts.py \
  tests/test_train_artifacts.py \
  research_experiments/vjepa_performance/benchmark_free_splats_throughput.py \
  research_experiments/vjepa_performance/benchmark_fast_mac_variants.py \
  research_experiments/vjepa_performance/benchmark_multicam_vjepa.py \
  research_experiments/vjepa_performance/profile_fast_mac_render_phases.py \
  research_experiments/vjepa_performance/compare_fast_mac_quality.py \
  research_experiments/star_uvt_feature_tubes/run_alpha_background_ablation.py

PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_train_artifacts.py \
  tests/test_train_devices.py \
  tests/test_trainer_registry.py -q
```

The focused pytest slice passed with 17 tests. A targeted scan found no
remaining open-coded JSON/JSONL write loops in the updated V-JEPA benchmark and
STAR alpha-background orchestration files.

Follow-up validation after routing `benchmark_fast_mac_variants.py`: py_compile
passed for that script and `tests/test_train_artifacts.py`; the focused
artifact-helper pytest passed with 4 tests; the same targeted JSON/JSONL writer
scan found no matches in the routed V-JEPA/STAR files.

Follow-up validation after routing `star_uvt_outputs.write_row_json(...)`:
py_compile passed for `src/train/star_uvt_outputs.py`, and
`tests/test_star_uvt_outputs.py` passed with 4 tests.

Follow-up validation after adding `write_text(...)` and routing the three STAR
diagnostic/report scripts: py_compile passed for `src/train/train_artifacts.py`,
`tests/test_train_artifacts.py`, `background_cheat_diagnostic.py`,
`firstclass_backward_breakdown.py`, and `firstclass_scale_report.py`; the focused
artifact-helper pytest passed with 5 tests. The three routed STAR scripts also
passed cheap `--help` import checks, and `background_cheat_diagnostic.py` wrote
real JSON/markdown artifacts under `/tmp` successfully.

Follow-up routing pass: the STAR UVT feature1 report family now delegates JSON
and markdown file writes to `write_json(...)` / `write_text(...)`:

- `star_uvt_feature1_trainer_trace_report.py`
- `star_uvt_feature1_chunktrace_report.py`
- `star_uvt_feature1_lr_reset_report.py`
- `star_uvt_feature1_lr001_continuation_report.py`
- `star_uvt_feature1_lr_schedule_report.py`
- `star_uvt_feature1_continuation_chain_report.py`

Validation passed: py_compile and `--help` imports passed for all six scripts,
and all six wrote report artifacts to `/tmp` with their real saved input JSONs.

2026-05-22 follow-up: added
`research_experiments/star_uvt_feature_tubes/report_artifacts.py` as the local
STAR report helper for root-relative paths, `src/train` bootstrap, and report
JSON/text writes. The same six feature1 report scripts now import
`ROOT`, `write_report_json(...)`, and `write_report_text(...)` from that helper
instead of carrying their own bootstrap block.

Validation passed for the helper extraction: py_compile passed for the helper,
its test, and the six routed report scripts; focused pytest passed with
7 artifact/report-helper tests; the six scripts again wrote report artifacts
under `/tmp` using their real saved input JSONs; and a targeted scan found no
remaining direct `sys.path` bootstrap or direct `train_artifacts` imports in
that feature1 report family.

Additional 2026-05-22 cleanup: `report_artifacts.py` now also owns
`load_report_json(...)`, `fmt_cell(...)`, and `fmt_pair(...)`. The feature1
report family no longer repeats identical `_load`, `_fmt`, or `_pair` helpers.
Validation passed: py_compile passed for the helper, test, and six report
scripts; `tests/test_star_uvt_report_artifacts.py` passed with 4 tests; a
targeted scan found no remaining local `_load`/`_fmt`/`_pair` definitions in
the feature1 report family; and the six scripts wrote `/tmp` report artifacts
again using their real input JSONs.

Additional report routing pass: three older STAR UVT report scripts now reuse
the same helper:

- `targetgrid_analytic_vjp_trainer_report.py`
- `gate4_quality_bracket_report.py`
- `logit_handoff_reduce_report.py`

They now use `load_report_json(...)`, `write_report_json(...)`, and
`write_report_text(...)` instead of local JSON-object loaders and
parent-directory plus sorted JSON/markdown write blocks. Domain-specific CSV
parsing, comparisons, and markdown table formatting remain local.

Validation passed: py_compile passed for the report helper, its test, and the
three routed report scripts; `tests/test_star_uvt_report_artifacts.py` passed
with 4 tests; `logit_handoff_reduce_report.py` wrote `/tmp` artifacts from the
real saved matrix and reported `pass=true`; `gate4_quality_bracket_report.py`
wrote `/tmp` artifacts from the real saved quality/speed inputs; and
`targetgrid_analytic_vjp_trainer_report.py` regenerated its real report
artifacts and reported `pass=true`. A targeted scan found no remaining local
JSON loader definitions or direct parent-mkdir/write-text artifact writes in
those three report scripts.

Additional train-tree routing pass: `src/train/build_clip_dataset.py` now uses
`train_artifacts.write_json(...)` for per-clip `summary.json` and `dataset.json`,
and `train_artifacts.write_jsonl(...)` for full/split manifest JSONL files. The
dataset builder still owns video probing, clip sampling, frame extraction, and
the manifest schema; only serialized artifact writes moved to the shared helper.

Validation passed: py_compile passed for `build_clip_dataset.py`,
`train_artifacts.py`, and the three routed STAR reports; the focused
artifact/report-helper pytest slice passed with 9 tests; a temp-runtime call to
`build_clip_dataset.write_manifest(...)` wrote `manifest.jsonl`, `train_manifest.jsonl`,
and `dataset.json`; and a targeted scan found no remaining `json.*` calls or
local `write_jsonl(...)` definition in `build_clip_dataset.py`.

Additional train/export routing pass: `export_dynaworld_browser_bundle.py` now
uses `write_json(...)` for `manifest.json`; `run_dust3r_video.py` uses it for
`per_frame_cameras.json` and `summary.json`; `train_dynamic_powerfoam_metal.py`
uses it for camera-teacher init metrics, per-frame eval metrics, and the final
dynamic-geometry summary; and `train_powerfoam_metal.py` uses it for eval color
calibration metrics and `best_metrics.json`. Tensor binary writes, NumPy/NPZ
outputs, checkpoints, PNGs, and MP4s stayed local because they have different
serialization or atomicity contracts.

Validation passed: py_compile passed for the export/DUSt3R/PowerFoam files and
`train_artifacts.py`; `tests/test_train_artifacts.py` passed with 5 tests; the
PowerFoam-focused project gate
`tests/test_powerfoam_direct.py tests/test_multicam_video_data.py` passed with
25 passed and 30 skipped; a direct temp `write_json(...)` call confirmed list
payloads preserve unsorted key order when requested; and a targeted scan found
no remaining direct JSON file writes in the routed export, dataset, DUSt3R, and
PowerFoam metric-artifact sites.

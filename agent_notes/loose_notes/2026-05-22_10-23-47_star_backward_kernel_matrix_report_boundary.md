# STAR Backward Kernel Matrix Report Boundary

## Goal

Continue the trainer/benchmark modularization pass by removing another local
report-artifact and subprocess-launch copy from the STAR UVT benchmarking lane.

## Change

- Routed `research_experiments/star_uvt_backward_kernel_matrix.py` through
  `research_experiments/star_uvt_feature_tubes/report_artifacts.py`.
- The matrix script now reuses:
  - dual-mode package/direct import bootstrap,
  - Dynaworld root resolution,
  - optional JSON object loading,
  - report JSON writing,
  - report CSV writing,
  - report text writing,
  - logged subprocess execution with timeout, stdout/stderr capture, and
    elapsed-time status.
- Kept STAR v0/PRT case construction, row summarization, and markdown row
  content local because those are benchmark-specific.

## Validation

```bash
rtk .venv/bin/python -m py_compile \
  research_experiments/star_uvt_backward_kernel_matrix.py \
  research_experiments/star_uvt_feature_tubes/report_artifacts.py
```

Passed.

```bash
rtk .venv/bin/python research_experiments/star_uvt_backward_kernel_matrix.py \
  --dry-run --no-include-v0 --no-include-prt \
  --out-dir /tmp/star_uvt_backward_kernel_matrix_shared_artifacts_smoke
```

Passed. It wrote `/tmp/star_uvt_backward_kernel_matrix_shared_artifacts_smoke/manifest.json`
through the shared report JSON helper with `case_count=0`.

```bash
rtk .venv/bin/python - <<'PY'
import research_experiments.star_uvt_backward_kernel_matrix as matrix
print(matrix.ROOT)
PY
```

Passed, confirming package import resolves the shared report helper and Dynaworld
root.

## Handoff

This does not rerun the GPU/Metal kernel matrix. It is a plumbing cleanup so
future benchmark reruns use the same artifact and subprocess contracts as the
newer STAR feature-tube reports.

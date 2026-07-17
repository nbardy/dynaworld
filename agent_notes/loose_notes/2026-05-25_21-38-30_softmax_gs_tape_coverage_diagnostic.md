# 2026-05-25 21:38:30 Softmax-GS tape coverage diagnostic

We added a focused bounded-tape coverage diagnostic because the 64px/4f/512
Softmax-GS repeat improved train loss but lost heldout PSNR versus no-op. The
question was whether K=16 was still dropping too much Softmax-GS mass after
training.

New files:

- `research_experiments/softmax_gs/diagnose_tape_coverage.py`
- `tests/test_softmax_gs_tape_coverage_diagnostic.py`
- `outputs/benchmarks/2026-05-25_softmax_gs_tape_coverage_64_4f_512_k16/summary.json`
- `outputs/benchmarks/2026-05-25_softmax_gs_tape_coverage_64_4f_512_k16/summary.md`

Validation:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_softmax_gs_tape_coverage_diagnostic.py \
  tests/test_softmax_gs_reference.py -q
# 12 passed

.venv/bin/python -m py_compile research_experiments/softmax_gs/diagnose_tape_coverage.py
```

Diagnostic command:

```bash
PYTHONPATH=src/train PYTHONUNBUFFERED=1 GSP_TAPE_CAP=16 .venv/bin/python \
  research_experiments/softmax_gs/diagnose_tape_coverage.py \
  src/train_configs/local_mac_multicam_softmax_gs_enabled_tapescalar_k16_rgb_pyramid_64_4f_512splats_20step.jsonc \
  --train-steps 20 \
  --k-values 1,2,4,8,16 \
  --views train0,train1,heldout0 \
  --output-dir outputs/benchmarks/2026-05-25_softmax_gs_tape_coverage_64_4f_512_k16
```

Main numbers:

```text
K=16 residual/alpha mean/p99:
    train0 camera_0001 0.000652 / 0.008290
    train1 camera_0015 0.000879 / 0.009899
    heldout camera_0040 0.001930 / 0.012332

K=8 residual/alpha mean/p99:
    train0 camera_0001 0.006965 / 0.054060
    train1 camera_0015 0.010092 / 0.057736
    heldout camera_0040 0.040167 / 0.112505
```

Interpretation:

K=16 is not an obviously lossy bounded tape on the 512-splat row. The heldout
PSNR miss is unlikely to be explained only by omitted tail mass. K=8 remains
too lossy, especially on heldout. This strengthens the current decision: keep
Softmax-GS as an opt-in dynamic-GS renderer probe, but do not port it to STAR
UVT or WorldFoam without a repeated heldout PSNR win.

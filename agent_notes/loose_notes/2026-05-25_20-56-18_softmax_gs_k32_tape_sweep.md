# Softmax-GS K32 Tape Sweep Follow-Up

Date:
    2026-05-25 20:56:18 Asia/Ho_Chi_Minh

Context:
    After the selected scalar tape VJP landed, the K=8 50-step row was too
    lossy and K=16 recovered the earlier seeded source-view bracket. We needed
    to know whether increasing the runtime tape cap to 32 improves the tiny
    diagnostic before choosing a setting for the next matched quality row.

New config:
    `src/train_configs/local_mac_softmax_gs_enabled_tapescalar_k32_diagnostic_seed17_64_4f_128splats_50step.jsonc`

Command:

```bash
PYTHONPATH=src/train WANDB_MODE=offline GSP_TAPE_CAP=32 .venv/bin/python \
  src/train/train_video_token_implicit_dynamic.py \
  src/train_configs/local_mac_softmax_gs_enabled_tapescalar_k32_diagnostic_seed17_64_4f_128splats_50step.jsonc
```

Result:
    initial `0.4338`
    final `0.1588`
    tqdm mean `3.63it/s`
    offline run `wandb/offline-run-20260525_205435-wy8r4v9l`

Comparison:
    - no-op seeded 50-step: final `0.1467`
    - enabled recompute seeded 50-step: final `0.1512`
    - selected scalar tape K=8: final `0.2026`
    - selected scalar tape K=16: final `0.1472`
    - selected scalar tape K=32: final `0.1588`

Interpretation:
    K=32 trains and is not catastrophic, but it does not improve the tiny
    endpoint. For this row K=16 is the current selected-tape setting to carry
    into the matched dynamic-GS quality diagnostic. K=8 remains a negative
    residual/approximation result.

Decision:
    Do not spend STAR/WorldFoam work from this. The next Softmax-GS step should
    use K=16 for a matched dynamic-GS quality row, optionally logging
    residual/tape coverage alongside the train.

# Trainer Interface Cleanup 2026-04-20

The dynamic trainers had drifted in the same places: loose sequence dictionaries,
tuple-shaped gaussian outputs, copied render dispatch, copied W&B video logging,
and duplicated implicit-camera SE(3) heads.

The cleanup deliberately avoided a shared `BaseTrainer`. The three training paths
still have different enough loops that a base class would hide useful detail. The
shared layer is now the boundary vocabulary:

- `runtime_types.py` defines `SequenceData`, `ClipBatch`, `GaussianFrame`,
  `GaussianSequence`, `CameraState`, and step-loss payloads.
- `sequence_data.py` owns shared video/frame/camera loading, timestamp
  normalization, and contiguous clip sampling.
- `rendering.py` owns dense/tiled renderer dispatch from a `GaussianFrame`.
- `train_logging.py` owns W&B preview/video payload construction.
- `gs_models/implicit_camera.py` owns zero-init camera heads and SE(3) camera
  composition shared by the image-token and video-token implicit-camera models.

The trainer scripts now consume those helpers but remain separate readable
experiments:

- `dynamicTokenGS.py`: known/prebaked camera from DUSt3R JSON.
- `train_camera_implicit_dynamic.py`: image encoder with implicit camera.
- `train_video_token_implicit_dynamic.py`: video encoder with implicit camera.

Verification:

- `uv run python -m py_compile` over shared modules, trainer scripts, and updated
  model files passed.
- Import smoke under `PYTHONPATH=src/train` passed.
- Shared camera/video loaders loaded two-frame test sequences.
- Image-token and video-token implicit camera model forward smokes passed.
- One-step offline W&B trainer smokes passed for known-camera, image-implicit,
  and video-token implicit paths.

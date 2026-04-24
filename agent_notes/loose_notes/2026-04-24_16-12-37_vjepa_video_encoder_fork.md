# V-JEPA video encoder fork

## Context

User asked to fork the camera-implicit architecture so the video encoder can be
V-JEPA 2.1. The existing active path is the video-token implicit-camera trainer
and model in:

- `src/train/gs_models/dynamic_video_token_gs_implicit_camera.py`
- `src/train/train_video_token_implicit_dynamic.py`

The prior model already had a clean local `VideoEncoder` boundary behind
`self.video_encoder`, so this pass kept the decoder/camera heads unchanged and
made only the encoder backend swappable.

## What changed

- Added `video_encoder_backend` with supported values:
  - `local`
  - `vjepa_hf`
  - `vjepa_torchhub`
- Added `HuggingFaceVJEPAVideoEncoder`, using `AutoVideoProcessor` +
  `AutoModel` and reading encoder tokens through `get_vision_features()` when
  available, otherwise `outputs.last_hidden_state`.
- Added `TorchHubVJEPAVideoEncoder`, using
  `torch.hub.load("facebookresearch/vjepa2", model_id)` so the V-JEPA 2.1
  hub entrypoints such as `vjepa2_1_vit_large_384` can be used.
- Both V-JEPA wrappers freeze the backbone by default and learn only the
  projection from V-JEPA hidden size into the local `model_dim`.
- Added checked-in configs:
  - `src/train_configs/local_mac_overfit_video_token_implicit_camera_vjepa2_hf_vitl_256.jsonc`
  - `src/train_configs/local_mac_overfit_video_token_implicit_camera_vjepa2_1_torchhub_vitl_384.jsonc`

## Testing

- `uv run python -m py_compile src/train/gs_models/dynamic_video_token_gs_implicit_camera.py src/train/train_video_token_implicit_dynamic.py`
- Config normalization smoke for the existing local config and both new V-JEPA
  configs.
- Existing local encoder factory smoke: built the old config and confirmed the
  backend is still `VideoEncoder`.
- Tiny random forward smoke with `local_mac_overfit_video_token_smoke.jsonc`.
- Fake-loader smokes for both V-JEPA wrappers to avoid downloading real
  checkpoints:
  - HF wrapper produced projected tokens shaped `(1, 6, 128)`.
  - torchhub wrapper produced projected tokens shaped `(1, 8, 7)` in the direct
    wrapper test.
- `uv run ruff check --select F ...` passed.
- Full `ruff check` was not clean because unrelated/pre-existing style findings
  in `src/train/train_video_token_implicit_dynamic.py` are present while another
  manifest/eval-sequence branch is dirty.

## Caveats

- I did not download real HF or torchhub V-JEPA weights and did not run a full
  training job. The wrappers were smoke-tested with fake loaders only.
- The worktree had parallel dirty work appear during this pass, including docs,
  build-clip-dataset changes, trainer manifest support, new local-30 configs,
  `.gitignore`, key learnings, and the `third_party/fast-mac-gsplat` submodule.
  This V-JEPA pass only intentionally owns the V-JEPA encoder/model/config
  changes plus this note.

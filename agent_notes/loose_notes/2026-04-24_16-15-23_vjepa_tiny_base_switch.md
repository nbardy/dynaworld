# V-JEPA tiny base switch

## Context

Follow-up correction: ViT-L is still about 0.3B parameters, so it is not the
right default for a first local experiment. User asked to use the tiniest model.

## Change

- Switched the V-JEPA 2.1 torchhub config from `vjepa2_1_vit_large_384` to
  `vjepa2_1_vit_base_384`.
- Renamed the new config surface to:
  `src/train_configs/local_mac_overfit_video_token_implicit_camera_vjepa2_1_torchhub_vitb_384.jsonc`
- Dropped the HF ViT-L config from the checked-in experiment surface so the
  easy path points at the tiny 2.1 model rather than a 0.3B ViT-L checkpoint.
- Set `vjepa_feature_dim` to `768`, matching ViT-B width.
- Updated the torchhub wrapper fallback so setting only
  `video_encoder_backend = "vjepa_torchhub"` resolves to the tiny 2.1
  `vjepa2_1_vit_base_384` model instead of inheriting the HF ViT-L id.

## Testing

- `uv run python -m py_compile src/train/gs_models/dynamic_video_token_gs_implicit_camera.py src/train/train_video_token_implicit_dynamic.py`
- Config normalization smoke confirmed:
  - `video_encoder_backend = "vjepa_torchhub"`
  - `vjepa_model_id = "vjepa2_1_vit_base_384"`
  - `vjepa_feature_dim = 768`

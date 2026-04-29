# Scene-Distinct 256px V-JEPA Baseline Setup

The 20 train / 10 test scene-distinct YouTube dataset is the default local
training smoke set. The existing checked-in trainer config
`local_mac_tiny_30_video_token_smoke.jsonc` already points at the 64px manifest,
but the V-JEPA comparison configs were still single-video overfit configs and
one of them rendered at 128px despite using a 256px V-JEPA checkpoint.

This pass adds a matched 256px clip materialization of the same 30 source-video
split:

```text
data/youtube_scene_distinct/clip_sets/youtube_scene_distinct_30_256_4fps_16f/
```

The new dataset config/script reuse the existing
`data/youtube_scene_distinct/candidates`, raw MP4s, and extracted high-motion
segments. The 256 script defaults to `build-clips` so it can rebuild PNG clips
without re-searching or redownloading YouTube data.

New training configs:

- `src/train_configs/local_mac_scene_distinct_30_local_encoder_256_fast_mac_2048splats.jsonc`
- `src/train_configs/local_mac_scene_distinct_30_vjepa2_vitl_fpc16_256_frozen_256_fast_mac_2048splats.jsonc`

Both use the 30-clip manifest, 16 frames, 256px model/render size, 2048 splats,
fast-mac renderer, 100 steps, microbatch temporal reconstruction, and two test
sequences for validation logging. The only intended encoder difference is local
video encoder versus frozen HF V-JEPA 2 ViT-L fpc16/256 SSV2 features.

Command path:

```bash
./src/dataset_scripts/youtube_scene_distinct_30_256_seed.sh build-clips
./src/train_scripts/train_local_mac_30_clip_vjepa2_256_baseline.sh [local|vjepa|both]
```

Important caveat: the current local raw source sections were originally
downloaded by the 64px pass at max height 360. The new 256 materialization is
still real 256px training input, but it is re-encoded from those local source
sections unless `--overwrite-raw` is used with the 256 config to refresh source
MP4s at the config's 720px cap.

Validation from this pass:

- Built `youtube_scene_distinct_30_256_4fps_16f`: 30 clips, 20 train, 10 test,
  480 PNG frames, 41 MB.
- Loaded train/test sample tensors at `(16, 3, 256, 256)`.
- Ran one local-encoder 256px fast-mac training step on MPS:
  `loss=0.541627`, `recon=0.541627`.
- Found the HF V-JEPA fpc16/256 model in the local Hugging Face cache and ran a
  no-render/no-backward MPS forward/decode smoke:
  `xyz=(16, 2048, 3)`, `cameras=16`, `backend=vjepa_hf`.

The HF load report showed unexpected classifier/pooler keys from
`facebook/vjepa2-vitl-fpc16-256-ssv2`. That is expected for the current use
because `AutoModel` loads the backbone feature model from a classification
checkpoint and ignores the task head.

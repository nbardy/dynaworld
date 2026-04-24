# V-JEPA 2.1 commit handoff

This wraps the V-JEPA encoder fork for the video-token implicit-camera path.

Final intended commit scope:

- `src/train/gs_models/dynamic_video_token_gs_implicit_camera.py`
- `src/train/train_video_token_implicit_dynamic.py`
- `src/train_configs/local_mac_overfit_video_token_implicit_camera_vjepa2_1_torchhub_vitb_384.jsonc`
- V-JEPA loose notes from this session

Important dirty-tree caveat: the same model/trainer files currently also have
other in-flight work around manifest/eval sequences and cached/precomputed video
features. Those are not part of the V-JEPA commit and should remain unstaged if
they were not intentionally merged.

The smoke that matters for this branch passed with real pretrained weights:

```text
TorchHubVJEPAVideoEncoder vjepa2_1_vit_base_384
xyz (4, 8192, 3) cameras 4 rot_delta (4, 3)
```

The upstream raw TorchHub `pretrained=True` path tried to download from
`http://localhost:8300`, so the wrapper now initializes the hub architecture
without weights and loads the public Meta encoder checkpoint explicitly.

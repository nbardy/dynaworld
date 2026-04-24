# V-JEPA 2.1 pretrained smoke

We checked the new tiny V-JEPA 2.1 TorchHub path against real model code and
weights, not only config parsing.

The first direct TorchHub `pretrained=True` probe failed because the current
`facebookresearch/vjepa2` hub file has its test base URL set to
`http://localhost:8300`. The loader tried to fetch:

```text
http://localhost:8300/vjepa2_1_vitb_dist_vitG_384.pt
```

That means raw upstream TorchHub cannot be trusted for released weights unless
the caller overrides the checkpoint path or the upstream hub file is fixed.

Patched our wrapper to instantiate known V-JEPA 2.1 hub models with
`pretrained=False`, then load encoder weights from Meta's public checkpoint URL
into the encoder only. For the tiny model this uses:

```text
https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt
```

Validation after the patch:

```text
uv run python -m py_compile src/train/gs_models/dynamic_video_token_gs_implicit_camera.py src/train/train_video_token_implicit_dynamic.py
uv run ruff check --select F src/train/gs_models/dynamic_video_token_gs_implicit_camera.py src/train/train_video_token_implicit_dynamic.py
git diff --check
```

All passed.

The real pretrained smoke downloaded the 1.55 GB public ViT-B checkpoint, loaded
the encoder weights, and completed a full implicit-camera model forward on a
random 4-frame clip:

```text
TorchHubVJEPAVideoEncoder vjepa2_1_vit_base_384
xyz (4, 8192, 3) cameras 4 rot_delta (4, 3)
```

This verifies the V-JEPA 2.1 encoder, projection, query decoder, Gaussian heads,
and camera head wiring. It is not a training benchmark or renderer smoke.

# Oracle Relative-Camera Swap Baseline

## Context

Goal: start moving from external-rig multicam training toward a V-JEPA baseline
that can learn relative camera SE(3):

```text
video A -> world tokens W_A
video B -> world tokens W_B / reference features F_B
F_A + F_B -> Delta_A_to_B
render(W_A, Delta_A_to_B) -> GT video B
```

The first implementation step is intentionally not a learned relpose head. It
is an oracle source-relative camera baseline that uses calibrated camera deltas
to prove the render/loss contract.

## Implemented

Added source-relative camera utilities in `src/train/multicam_video_data.py`:

- `video_path_for_camera(...)`
- `MulticamVideoBundle.train_sequences`
- `MulticamVideoBundle.heldout_sequences`
- `source_relative_cameras_from_K_w2c(...)`

The relative camera convention is:

```text
stored w2c_view_anchor = inverse(c2w_view) @ c2w_anchor
target_in_source_w2c = w2c_target_anchor @ inverse(w2c_source_anchor)
target_in_source_c2w = inverse(target_in_source_w2c)
```

So the world is not moved. `W_source` remains source-anchored; only the query
camera changes.

Added opt-in trainer mode in
`src/train/train_multicam_precomputed_feature_implicit_dynamic.py`:

```jsonc
"train": {
  "camera_swap_mode": "oracle_relative",
  "camera_swap_pairs_per_step": 0,
  "camera_swap_include_self": true,
  "camera_swap_include_cross": true
}
```

In oracle-relative mode:

- the trainer loads all train-camera sequences, not only the condition camera
- V-JEPA caches are baked/loaded per train camera
- each sampled `CameraSwapPair` decodes only the source camera video
- target views use calibrated source-relative query cameras
- all self/cross pairs are supported by `camera_swap_sampling.py`
- validation renders train + heldout targets from source view 0 using the same
  source-relative camera path

Added config:

- `src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_static_dynamic_96_32_precomputed_vjepa2_1_vitb_384_128_16f_8192splats_oracle_relative_camera.jsonc`

Added focused math test:

- `tests/test_source_relative_cameras.py`

## Verification

Compile:

```bash
PYTHONPATH=src/train uv run python -m py_compile \
  src/train/multicam_video_data.py \
  src/train/train_multicam_precomputed_feature_implicit_dynamic.py \
  src/train/camera_swap_sampling.py
```

Focused assertion runner, because this `uv run python` environment currently
does not include `pytest`:

```bash
PYTHONPATH=src/train uv run python - <<'PY'
import torch
from camera_swap_sampling import build_train_camera_swap_pairs, camera_swap_pair_counts
from multicam_video_data import source_relative_cameras_from_K_w2c

pairs = build_train_camera_swap_pairs(2, train_camera_names=['cam_a', 'cam_b'])
assert camera_swap_pair_counts(pairs) == {'total': 4, 'self': 2, 'train_cross': 2, 'heldout': 0}

def translate(x, y, z):
    transform = torch.eye(4)
    transform[:3, 3] = torch.tensor([x, y, z])
    return transform

source_c2w = translate(1.0, 0.0, 0.0)
target_c2w = translate(1.0, 2.0, 3.0)
K = torch.eye(3)
cameras = source_relative_cameras_from_K_w2c(
    source_w2c=torch.linalg.inv(source_c2w).unsqueeze(0),
    target_K=K,
    target_w2c=torch.linalg.inv(target_c2w).unsqueeze(0),
    frame_indices=torch.tensor([0]),
)
assert torch.allclose(cameras[0].camera_to_world, torch.linalg.inv(source_c2w) @ target_c2w)
print('camera swap/source-relative assertions ok')
PY
```

1-step offline smoke:

```bash
cp src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_static_dynamic_96_32_precomputed_vjepa2_1_vitb_384_128_16f_8192splats_oracle_relative_camera.jsonc /tmp/oracle_relative_camera_smoke.jsonc
perl -0pi -e 's/"steps": 250/"steps": 1/; s/"log_every": 10/"log_every": 1/; s/"image_log_every": 50/"image_log_every": 1/; s/"video_log_every": 250/"video_log_every": 1/' /tmp/oracle_relative_camera_smoke.jsonc
PYTHONPATH=src/train WANDB_MODE=offline uv run python \
  src/train/train_multicam_precomputed_feature_implicit_dynamic.py \
  /tmp/oracle_relative_camera_smoke.jsonc
```

Smoke result:

- passed
- offline run: `wandb/offline-run-20260503_012110-zr7qkgro`
- loaded 2 train sequences
- cache hit for both `camera_0001` and `camera_0015`
- logged `CameraSwap/OracleRelativeMode = 1`
- rendered train and heldout validation media through the oracle-relative path

Final smoke eval after one step:

| View | PSNR | SSIM |
|---|---:|---:|
| TrainView0 | 7.1193 | 0.0899 |
| TrainView1 | 8.7221 | 0.1311 |
| Heldout camera_0040 | 7.7211 | 0.0831 |

These numbers are only a path smoke, not a quality baseline.

## Remaining Work

This does not yet implement:

- `relpose(F_A, F_B) -> residual SE(3)`
- pose supervision against calibrated `Delta_GT`
- cycle loss `Delta_A_to_B @ Delta_B_to_A ~= I`
- target/reference feature leakage guards for the learned head
- a real train/eval run that can be added to `BASELINES.md`

Next concrete task: add a tiny cross-attention relpose module that takes
projected V-JEPA features from source and target views and predicts a bounded
SE(3) residual around the oracle calibrated delta.

## 2026-05-03 Follow-up Checks

Additional checks after tightening the dirty implementation:

- `py_compile` passed for `src/train/multicam_video_data.py`,
  `src/train/train_multicam_precomputed_feature_implicit_dynamic.py`, and
  `src/train/camera_swap_sampling.py`.
- `pytest` is not installed in the active uv/venv environment, so the focused
  source-relative camera tests were run by directly importing and calling the
  test functions from `tests/test_source_relative_cameras.py`.
- Tiny no-V-JEPA oracle-relative smoke passed with 2 frames, 32px, 8 tokens,
  8 gaussians/token. It loaded 2 train sequences and rendered all 4 self/cross
  train pairs.
- Tiny V-JEPA oracle-relative smoke passed with 2 frames, 32px, 8 tokens,
  8 gaussians/token, and 2 sampled camera-swap pairs. It baked small feature
  cache entries for the two train views and exercised the precomputed-feature
  pairwise source path.
- Tiny heldout validation payload passed through the oracle-relative path and
  emitted heldout metrics for `W_camera_0001 + Delta_camera_0001_to_camera_0040`.

## Learned Residual Relpose Head

Follow-up implementation added:

- `src/train/relative_pose.py`
- `train.camera_swap_mode = "learned_residual"`

The relpose head is intentionally tiny:

```text
source_memory = project(F_source)
target_memory = project(F_target)
queries      = learned RELPOSE token(s)
queries xattn [source_role + source_memory, target_role + target_memory]
MLP(query0)  -> bounded rotation/translation residual
```

The output is initialized to exact zero residual, so the learned-residual path
starts as the oracle calibrated-delta path:

```text
Delta_pred = Delta_GT_source_to_target @ exp_se3(residual)
```

Guardrail preserved: target features are used only by the relpose head. The
world decode still receives the source view only. Heldout validation does not
consume heldout RGB features; it uses calibrated/query heldout deltas.

Verification:

- `py_compile` passed for `src/train/relative_pose.py`,
  `src/train/train_multicam_precomputed_feature_implicit_dynamic.py`, and the
  focused relpose tests.
- Direct test runner passed for `tests/test_relative_pose.py` and
  `tests/test_source_relative_cameras.py`.
- Tiny learned-residual V-JEPA smoke passed with 2 frames, 32px, 8 tokens,
  8 gaussians/token, `video_feature_token_stride=9`, and two sampled pairs
  (`1` self + `1` cross). It cache-hit the two train-camera V-JEPA payloads,
  constructed `RelativePoseCrossAttentionHead`, and produced zero initial
  relpose identity loss as expected.

Still not done:

- no real 250/1000-step learned-residual V-JEPA run yet
- no camera-token leakage probe yet
- no `BASELINES.md` row yet

## 20-Step Learned-Residual Run

Command used the checked learned-residual config with a temporary 20-step patch:

```bash
PYTHONPATH=src/train WANDB_MODE=offline uv run python \
  src/train/train_multicam_precomputed_feature_implicit_dynamic.py \
  /tmp/learned_residual_relpose_20step_2026-05-03.json
```

Patch details:

- `train.steps = 20`
- `train.camera_swap_pairs_per_step = 2`
- `train.camera_swap_self_pair_probability = 0.5`
- media/log cadence set to 20/5 for a quick offline check

Run:

- `wandb/offline-run-20260503_014141-tjg8bbou`
- full `128px`, `16f`, `8192` splats
- compact V-JEPA conditioning: cached `(1, 4608, 768)` fp16 -> stride-9
  projected `(1, 512, 64)` bf16
- 2 train sequences loaded: `camera_0001`, `camera_0015`
- heldout query: calibrated `Delta_camera_0001_to_camera_0040`

Step-0 eval:

| View | PSNR | SSIM |
|---|---:|---:|
| TrainView0 | 4.7844 | 0.0815 |
| TrainView1 | 4.7667 | 0.0887 |
| Heldout camera_0040 | 4.0802 | 0.0816 |

Step-20 eval:

| View | PSNR | SSIM |
|---|---:|---:|
| TrainView0 | 12.1041 | 0.1111 |
| TrainView1 | 11.9266 | 0.1184 |
| Heldout camera_0040 | 11.3707 | 0.1027 |

Relpose terms were live and nonzero by the end:

- `BankRate/relpose_identity_loss ~= 3e-05`
- `BankRate/relpose_cycle_loss ~= 3e-05`

This is still a smoke/probe, not a `BASELINES.md` row. It proves the learned
residual head, self/cross camera-swap training, cycle loss, and heldout
source-relative eval path all run together on the intended V-JEPA config shape.

## 250-Step Learned-Residual Baseline Run

Command:

```bash
PYTHONPATH=src/train WANDB_MODE=offline uv run python \
  src/train/train_multicam_precomputed_feature_implicit_dynamic.py \
  src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_vjepa_learned_residual_relpose_small_bf16_128_16f_8192splats.jsonc
```

Run:

- `wandb/offline-run-20260503_014531-woco72am`
- `128px`, `16f`, `8192` splats
- 2 train sequences loaded: `camera_0001`, `camera_0015`
- compact V-JEPA conditioning: stride-9 projected `(1, 512, 64)` bf16
- self/cross sampler: 2 sampled pairs per step with 0.5 self-pair probability
- heldout eval query: calibrated `Delta_camera_0001_to_camera_0040`

Step-0 eval:

| View | PSNR | SSIM |
|---|---:|---:|
| TrainView0 | 5.0189 | 0.0805 |
| TrainView1 | 4.8836 | 0.0880 |
| Heldout camera_0040 | 4.4559 | 0.0819 |

Step-250 eval:

| View | PSNR | SSIM |
|---|---:|---:|
| TrainView0 | 14.9334 | 0.1966 |
| TrainView1 | 15.5946 | 0.2731 |
| Heldout camera_0040 | 14.0453 | 0.1714 |

Run summary relpose terms:

- `BankRate/relpose_identity_loss = 0.00020`
- `BankRate/relpose_identity_loss_weighted = 0.00020`
- `BankRate/relpose_cycle_loss = 0.00024`
- `BankRate/relpose_cycle_loss_weighted = 0.00002`

This row is now recorded in `BASELINES.md`. It is the best measured heldout
PSNR on the DeepView 3-cam train2/test1 split so far, but it is not the end of
the story: the path still needs leakage probes, seed checks, and a longer run.
The important contract held through the run: target V-JEPA features can affect
only the relative-pose residual head, not `W_source`, and heldout eval does not
consume heldout RGB features.

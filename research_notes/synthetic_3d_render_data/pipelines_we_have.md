# Pipelines We Already Have

State as of 2026-04-25, sourced from `~/git/nova/data/`.

Read this before proposing to build a new synthetic-data pipeline. There are
two-and-a-half functional pipelines already; the right move is usually to
reuse v2 rather than start fresh.

## v1 — Blender + MoCap (over-engineered, blocked)

**Location**: `~/git/nova/data/v1/`

**What it is**: 7 Python modules (~100K total) — `pipeline.py`, `renderer.py`,
`retarget.py`, `camera_rig.py`, `importer.py`, `metadata.py`, `constants.py`.
Heavy class hierarchy: `BulletTimePipeline`, `SceneImporter`,
`AnimationRetargeter`, `CameraRig`, `BulletTimeRenderer`, `MetadataCollector`.

**What works**: Infrastructure loads, modules import, configs validate. Metal
backend confirmed. MVP demonstrated on a small skateboard test scene.

**What's broken / stalled**:
- CMU MoCap source URLs return 404. The whole retargeting branch can't be
  exercised without working BVH input.
- Asset directories are mostly empty (`mocap/cmu_134/`,
  `characters/{mixamo,custom}/`, `environments/`).
- Output dir is empty — real renders never landed here.
- Code style: heavy `try/except → log → return False` ladders, multiple silent
  fallbacks. Violates Nicholas's "one clean path / no silent fallbacks" rules.

**Verdict**: deprecate. The retargeting module is the only piece worth
preserving if a working BVH source ever appears (Mixamo, AMASS, BEDLAM).

## v2 — Blender, pre-made scenes (the working one)

**Location**: `~/git/nova/data/v2/bullet-time/`

**What it is**: ~8K Python total — `pipeline.py` (6.6K), `camera_rig.py`
(1.8K), `create_test_scene.py`. Single-file orchestrator that opens any
`.blend`, finds an animation target, builds an orbital camera rig, renders
H.264 MP4s direct from Cycles.

**Configs** (self-documenting with `_description`, `_use_case`,
`_estimated_time`, `_timeout_risk`):
- `ultra_fast.json` — 6s preview
- `five_angles.json` — 17s, five-angle bullet-time
- `fast_test.json` — 3min low quality
- `hq_single_mac.json` — 5–15min HQ single cam
- `dev_mac.json` — 15min Mac dev
- `prod_cuda.json` — 2hr+ NVIDIA production

**Proven outputs** in `~/renders/`:
- `bike_bullet_time/`, `bike_five_angles/`, `bullet_time_test_fast/`,
  `bullet_time_ultra_fast/`, `bullet_time_v2_test/`, `bullet_time_videos/`,
  `cloth_test/`, `hq_bike_final/`, `hq_bike_test/`, `mocap_test/`,
  `skateboard_realistic/`, `skateboard_videos/`, `test_video/`.
- Best evidence of real work: `hq_bike_final/bullet_time_cam_00.mp4` —
  1920×1080 / 256 samples / 35,652s render time on Mac M4.

**Populated assets**:
- `greasepencil-bike.blend` (13 MB) — fully working scene
- `cloth_inner_springs.blend` (65 MB) — physics test scene
- `spring.zip` (43 KB) — Blender Foundation Spring, **not yet extracted**

**Empty stub directories** (waiting for downloads):
- `assets/agent327/`, `assets/bbb/`, `assets/cosmos/`, `assets/sintel/`,
  `assets/spring/`

**Known issues**:
- `find_target_object()` has a 4-way silent-fallback chain (armature →
  animated mesh → any mesh → create empty at origin). Should be a typed
  `TargetSpec` per Nicholas's style rules. Easy fix.

**Verdict**: production-ready. Default for any new bullet-time scene work.
Extending it = drop a new `.blend` into `assets/`, run with an existing
config.

## v3 — Unity (stub, not real)

**Location**: `~/git/nova/data/v3/`

**What it is**: A `BulletTimeBatch.cs` (10K) Unity Editor script with menu
hooks for `RenderClip`, `RenderProduction`, `RenderFastTest`. Plus a Python
orchestrator that **generates synthetic videos using FFmpeg**, not Unity.

**The lie embedded in the output dir**: the 13 `.mp4` files in
`output/videos/` look like they came from Unity. They didn't — they're
FFmpeg-rendered cubes with config text overlays. No 3D rendering ever
happened.

**What's missing**:
- Unity itself is not installed (no Unity Hub, no Unity binary).
  CLI install failed in a prior session.
- `unity_project/`, `sample_projects/`, `assets/` are all empty.
- `BulletTimeBatch.cs` references `UnityEditor.Recorder.*` but has never
  been compiled or executed.

**To make real**: ~half-day setup minimum.
1. Install Unity Hub manually from unity.com (CLI unreliable).
2. Install Unity 2022 LTS via Hub (~5–7 GB).
3. Clone Boss Room (8 GB, lowest-risk URP project) into `sample_projects/`.
   Skip Megacity Metro — DOTS package versions break frequently.
4. `git lfs pull`.
5. Tag a hero object as `BulletTarget`.
6. Open the project, drop in `BulletTimeBatch.cs`, run "Render Fast Test (2s)".

**Disk reality**: Unity Hub + Editor + Boss Room + cache ≈ 20 GB. Currently
~48 GB free of 460 GB (89% full); clear ~50 GB headroom before starting.

**Verdict**: paused. Only revive if MetaHumans / city-scale crowds /
real-time iteration becomes a hard requirement. The Blender path already
covers what we need.

## Practical reuse for DynaWorld

If DynaWorld wants synthetic clips for a probe / pretrain / camera-leakage
test:

- **Don't build a new pipeline**. Use v2.
- **Add a synthetic-mode config** to v2 that exports per-frame camera
  intrinsics + extrinsics in a format DynaWorld's `CameraSpec` adapter can
  consume. The v2 metadata collector already writes JSON with camera params.
- **Render multiple camera paths over the same scene/time** — v2's
  `BulletTimeRenderer.render_cameras()` does this already; multiple cams
  rendering the same animation frames.
- **Pipe the outputs into DynaWorld's existing intake**:
  `multicam_val_v1_seed.sh` is the seed point; a synthetic source could be
  a new entry alongside AIST / DeepView / Neural3D / ViVo.

The minimum delta to make v2 useful for DynaWorld:
1. Fix the `find_target_object` silent fallback (typed `TargetSpec`).
2. Emit camera metadata in DynaWorld's `CameraSpec` schema (see
   `data/CAMERA_CONTRACT.md`).
3. Add a config preset that renders the same hero frame range from N cameras
   with synchronized timestamps.

That's a half-day of work, not a new project.

# PowerFoam Raytrace Topology And Start Diagnostics

Date: 2026-05-06

Scope: continue the PowerFoam Metal completion lane after the CUDA smoke and
ALIKED probes. The active blocker is still paper-scale real-scene acceptance,
not local forward/backward/4K trainability.

## What Changed

- Extended
  `research_experiments/dynamic_foam/verify_powerfoam_raytrace_real_view_alpha.py`
  with a diagnostic-only `first_sphere_hit` start mode.
- Added
  `research_experiments/dynamic_foam/diagnose_powerfoam_topology_edges.py`.
  It compares the fast sphere-overlap `cech_aabb` graph against the optional
  SciPy regular-triangulation teacher on a frozen checkpoint.

## Commands

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/dynamic_foam/diagnose_powerfoam_topology_edges.py \
  research_experiments/dynamic_foam/verify_powerfoam_raytrace_real_view_alpha.py
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with scipy python \
  research_experiments/dynamic_foam/verify_powerfoam_raytrace_real_view_alpha.py \
  src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_regular_128_16f_1024cells_40step_noaux.jsonc \
  --output outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_regular_128_16f_1024cells_40step_noaux/raytrace_start_modes_with_first_sphere_diagnostics.json \
  --views camera_0021 camera_0013 \
  --frames 0 4 \
  --sample-size 9 \
  --adjacency-modes cech_aabb regular_triangulation all_pairs \
  --start-modes origin default_per_ray near_plane first_sphere_hit
```

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with scipy python \
  research_experiments/dynamic_foam/diagnose_powerfoam_topology_edges.py \
  src/train_configs/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_regular_128_16f_1024cells_40step_noaux.jsonc
```

## Results

Start-mode diagnostic artifact:

```text
outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_regular_128_16f_1024cells_40step_noaux/raytrace_start_modes_with_first_sphere_diagnostics.json
```

`first_sphere_hit` improves Cech/AABB sampled alpha but does not solve it:

- `camera_0021`: stream alpha mean `0.8927`; near-plane/default Cech alpha
  `0.5223`; first-sphere-hit Cech alpha `0.6326`; first-hit mean alpha error
  `0.2641`.
- `camera_0013`: stream alpha mean `0.8992`; near-plane/default Cech alpha
  `0.4897`; first-sphere-hit Cech alpha `0.6615`; first-hit mean alpha error
  `0.3101`.

Topology diagnostic artifact:

```text
outputs/powerfoam_metal/local_mac_powerfoam_metal_multicam_deepview_8cam_holdout1_pycolmap_known_pose_frames0_4_8_12_1024px_true_multiframe_opencv_fisheye_appearanceonly_wandboffline_init_raytrace_regular_128_16f_1024cells_40step_noaux/topology_edge_diagnostics.json
```

Across frames `0/4/8/12`, each frame has:

- `4234` Cech/AABB undirected edges
- `7207` regular-triangulation undirected edges
- `3389` shared edges
- `3818` regular edges missing from Cech
- regular-edge coverage by Cech: `0.470237`

All missing regular edges are non-overlapping under the current sphere-overlap
test (`non_overlapping_fraction=1.0`, median overlap margin `-0.0259`).

## Interpretation

Cech/AABB can still be the fastest selected synthetic 4K path, but it is not a
conservative regular-triangulation ray-walk graph on this real checkpoint.
Streaming can scan all cells and clip independently; a ray-walk graph cannot
cross regular power faces that are absent from the Cech graph. The next topology
work should treat regular triangulation as a teacher for real-scene traversal
or build a fast hybrid graph, rather than only changing start cells.

# Point-Cloud Report Artifact Boundary

## Context

Continuation of the trainer-landscape modularization goal. The current cleanup
rule is still to share repeated artifact/report primitives only where the file
contract is common, while leaving domain artifacts and execution inputs local.

This slice focused on Dynamic Foam point-cloud summary JSONs.

## Change

- `build_pycolmap_known_pose_point_cloud.py` now writes its adjacent
  `output.with_suffix(".json")` summary through
  `report_artifacts.write_report_json(...)`.
- `merge_ascii_ply_point_clouds.py` now writes its merged PLY summary JSON
  through the same helper.
- `build_multiview_feature_triangulation_point_cloud.py` now routes its
  no-valid-points diagnostic summary through the same helper, matching the
  success-path summary write already routed there.
- The PLY writers remain local. They have a different serialization contract
  from report-shaped JSON.
- `build_pycolmap_known_pose_point_cloud.py` now imports optional `pycolmap`
  lazily at execution time, so `--help` works in lightweight local
  environments without pycolmap installed.

## Validation

```bash
rtk .venv/bin/python -m py_compile \
  research_experiments/dynamic_foam/build_pycolmap_known_pose_point_cloud.py \
  research_experiments/dynamic_foam/merge_ascii_ply_point_clouds.py \
  research_experiments/dynamic_foam/build_multiview_feature_triangulation_point_cloud.py \
  research_experiments/dynamic_foam/report_artifacts.py
```

Passed.

```bash
rtk env PYTHONPATH=src/train:research_experiments/dynamic_foam \
  .venv/bin/python research_experiments/dynamic_foam/build_pycolmap_known_pose_point_cloud.py --help
```

Passed. This had previously failed before argparse because `pycolmap` was
imported at module import time.

```bash
rtk .venv/bin/python research_experiments/dynamic_foam/merge_ascii_ply_point_clouds.py --help
rtk env PYTHONPATH=src/train:research_experiments/dynamic_foam \
  .venv/bin/python research_experiments/dynamic_foam/build_multiview_feature_triangulation_point_cloud.py --help
```

Both passed.

```bash
rtk uv run --with pytest python -m pytest tests/test_dynamic_foam_report_artifacts.py -q
```

Passed: 5 tests.

`rg` found no remaining local adjacent summary-write pattern in the three
touched point-cloud scripts.

## Handoff

This is artifact-boundary cleanup only. It does not change point-cloud
generation, COLMAP/pycolmap behavior, PLY contents, or triangulation math.

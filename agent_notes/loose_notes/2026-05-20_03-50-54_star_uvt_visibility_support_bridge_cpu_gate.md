# STAR UVT Visibility Support Bridge CPU Gate

Date: 2026-05-20 03:50 +07

## Why This Gate Exists

The active STAR UVT fast-shader goal had one important gap after the sparse-F1
alpha work: every alpha/grid/opacity follow-up was still a same-support
objective. Those losses can make already-hit pixels more opaque or change their
colors, but they do not create useful gradients for target pixels that the
current projected tubes do not touch.

The next aligned missing implementation detail is a support-changing visibility
bridge. This pass added a CPU-first prototype before touching the trainer or
Metal.

## Added

- `research_experiments/star_uvt_feature_tubes/visibility_support_bridge_prototype.py`
- `tests/test_star_uvt_visibility_support_bridge.py`
- benchmark outputs:
  - `outputs/benchmarks/2026-05-20_star_uvt_visibility_support_bridge_cpu_proxy.json`
  - `outputs/benchmarks/2026-05-20_star_uvt_visibility_support_bridge_cpu_proxy.md`

The prototype builds a tiny projected-UVT miss scene. Target foreground pixels
start with zero dense alpha hits. It compares:

1. dense same-support alpha optimization through the existing dense renderer;
2. a soft target-pixel to projected-tube coverage proxy that directly sends
   gradients to tube centers and velocities.

## Result

The gate passes.

| path | target alpha mean | target alpha >0.10 | note |
| --- | ---: | ---: | --- |
| initial | `0.0000` | `0.0000` | deliberate miss scene |
| same-support alpha | `0.0000` | `0.0000` | loss decreases by shrinking/removing wrong support, but target stays uncovered |
| support proxy | `0.0920` | `0.3235` | center/velocity gradients move projected support toward target |

The support proxy lowers proxy loss `45.109 -> 0.296`, lowers dense alpha loss
`0.028296 -> 0.021939`, and runs at `1.18ms` per CPU step in this toy gate.

## Interpretation

This is real progress on the missing mechanism, but it is not a visual-quality
claim. It proves that the next STAR UVT bridge can create geometry gradients
from target pixels that currently have no alpha support.

The next implementation step is a first-class trainer gate that adds this kind
of proxy beside the selected cached-V-JEPA/visual probe route, then reruns the
single-video dense media gate. Only if that clears the current `5.6-6.0` dense
RGB failure band should the 300-video scale lane run.

## Validation

```bash
PYTHONPATH=src/train:. rtk .venv/bin/python -m py_compile \
  research_experiments/star_uvt_feature_tubes/visibility_support_bridge_prototype.py \
  tests/test_star_uvt_visibility_support_bridge.py
```

```bash
PYTHONPATH=src/train:. rtk uv run --with pytest python -m pytest \
  tests/test_star_uvt_visibility_support_bridge.py -q
# 1 passed
```

```bash
PYTHONPATH=src/train:. rtk .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/visibility_support_bridge_prototype.py \
  --device cpu \
  --steps 80 \
  --out-json outputs/benchmarks/2026-05-20_star_uvt_visibility_support_bridge_cpu_proxy.json \
  --out-md outputs/benchmarks/2026-05-20_star_uvt_visibility_support_bridge_cpu_proxy.md
# pass true
```

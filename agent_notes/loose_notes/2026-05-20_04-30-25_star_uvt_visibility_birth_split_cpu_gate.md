# STAR UVT Visibility Birth/Split CPU Gate

Date: 2026-05-20

## Goal Context

The active STAR UVT feature-shader goal remains broader than this gate. The
immediate blocker after the opacity/precision support proxy was that the proxy
closed gradient plumbing but did not move dense support and cost too much
(`693.7ms` proxy work per step). The next documented candidate was support
density or support birth/split.

## What Changed

Added a CPU-first mechanism gate:

`research_experiments/star_uvt_feature_tubes/visibility_support_birth_split_prototype.py`

The gate reuses a fixed tube budget. It starts from the same zero-hit target
fixture family as the previous visibility bridge, then reallocates a subset of
dead/miss tubes onto the target mask by fitting a simple screen-space motion
line and seeding support offsets around it. It then runs ordinary dense-alpha
refinement to check whether background leakage can be reduced without losing
target coverage.

Focused test added in:

`tests/test_star_uvt_visibility_support_bridge.py`

The test checks that support birth/split keeps the same tube count and
increases target alpha support from the fixed-budget model.

## Gate Command

```bash
PYTHONPATH=src/train:. rtk .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/visibility_support_birth_split_prototype.py \
  --device cpu \
  --out-json outputs/benchmarks/2026-05-20_star_uvt_visibility_birth_split_cpu_gate.json \
  --out-md outputs/benchmarks/2026-05-20_star_uvt_visibility_birth_split_cpu_gate.md
```

## Result

Artifacts:

- `outputs/benchmarks/2026-05-20_star_uvt_visibility_birth_split_cpu_gate.md`
- `outputs/benchmarks/2026-05-20_star_uvt_visibility_birth_split_cpu_gate.json`

The gate passed.

| Path | Target alpha mean | Target alpha >0.10 | Background alpha mean | Train loss |
| --- | ---: | ---: | ---: | ---: |
| initial miss | `0.0000` | `0.0000` | `0.0207` | n/a |
| same-support alpha | `0.0000` | `0.0000` | `0.0003` | `0.0342 -> 0.0217` |
| center support proxy | `0.1492` | `0.5784` | `0.0267` | `44.6627 -> -0.0459` proxy |
| birth/split initial | `0.8784` | `1.0000` | `0.0479` | n/a |
| birth/split refined | `0.7553` | `1.0000` | `0.0072` | `0.0233 -> 0.0033` |

Birth/split used a fixed budget of `16` tubes and reallocated `8`. The fitted
target motion was center-at-`t=0` `[20.4778, 18.7611]` with velocity
`[0.6000, -0.3000]`.

## Read

This is the first positive mechanism gate after the failed support proxies.
The center/support proxy family tries to pull existing tubes across support and
either partially succeeds in CPU or fails to move dense support in the trainer.
Birth/split changes the support set directly, then the existing alpha gradient
can reduce background leakage.

This does not prove a first-class trainer or Metal shader yet. It does justify
the next implementation step: add a trainer-side opt-in that reallocates dead or
low-contribution tubes from target uncovered pixels, then run the same dense
diagnostic against a real sparse-1500 checkpoint.

## Validation

- `py_compile` passed for the new script and focused test.
- `pytest tests/test_star_uvt_visibility_support_bridge.py -q` passed:
  `2 passed`.
- The CPU birth/split gate exited successfully and wrote markdown/JSON.

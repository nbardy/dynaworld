# Softmax-GS Ingest And Integration Handoff

## Context

User asked to download arXiv `2604.27437`, sort it into research notes, convert
it to Markdown, and think through integration into dynamic GS and STAR UVT.

## What Changed

Created a new general Gaussian-splat paper bucket:

```text
research_notes/gaussian_splatting_papers/
```

Artifacts:

```text
research_notes/gaussian_splatting_papers/pdfs/2604_27437_softmax_gs.pdf
research_notes/gaussian_splatting_papers/text/2604_27437_softmax_gs.txt
research_notes/gaussian_splatting_papers/sources/2604_27437_softmax_gs/
research_notes/gaussian_splatting_papers/2604_27437_softmax_gs_converted.md
research_notes/gaussian_splatting_papers/2604_27437_softmax_gs_dynaworld_integration.md
research_notes/gaussian_splatting_papers/paper_index.md
research_notes/gaussian_splatting_papers/README.md
```

Also linked the bucket from `research_notes/README.md`.

## Paper Model

Softmax-GS adds per-Gaussian:

- `alpha`: generalized exponential boundary sharpness;
- `beta`: softmax competition strength;
- `gamma`: depth-separation decay of competition.

It changes per-ray compositing by competing overlapping splats, then correcting
two-way absorbance so order is approximately invariant and final transmittance
matches original product transmittance. The backward pass needs a K-limited
forward tape; paper uses `K=128` in real-scene experiments.

## DynaWorld Read

Dynamic GS:
    Worth a bounded renderer experiment. It might improve sparse dynamic splat
    parameter efficiency and reduce popping under camera deltas.

STAR UVT:
    Potentially useful as a post-support compositing/visibility law, but not
    the current support/coverage bridge. STAR's known blocker is insufficient
    target alpha coverage, and Softmax-GS mostly redistributes existing
    overlapping support.

Implementation gotcha:
    Current `src/train/renderers/fast_mac.py::project_for_fast_mac(...)` sends
    artificial rank depths into the Metal renderer. Softmax-GS `gamma` needs
    real projected depth or pixel-affine depth, so a naive shader fork would
    have a fake depth-separation signal.

## Suggested Next Gate

1. CPU/Torch Softmax-GS tiny reference for dynamic splats:
   order-invariance and transmittance tests.
2. One fast-mac variant behind config after reference parity.
3. Matched dynamic-gsplat smoke at fixed primitive count.
4. STAR CPU reference only if dynamic smoke is positive or if support coverage
   improves enough that overlap/composition becomes the visible blocker.

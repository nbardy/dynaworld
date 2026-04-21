# Potential Directions Index

## What Changed

Added `research_notes/potential_directions_index.md` as a routing layer for
Dyna World research ideas. The goal is to keep the top-level idea map separate
from dense paper notes, architecture proposals, and raw chat transcripts.

The index categorizes:

- fast rendering and training throughput,
- TokenGS-style splat tokens,
- single-step video diffusion to splats,
- pixel-space SDS from rendered splats,
- video diffusion distillation,
- rolling/diffusion forcing,
- ChopGrad and memory-safe pixel losses,
- parallel global vs causal autoregressive decode,
- geometry priors and camera grounding.

## Why This Shape

The user asked for a document that tracks potential directions such as fast
rendering, video diffusion distillation, rolling diffusion forcing, and the
rendered-pixel SDS idea. I made it a routing document rather than another dense
research note so future sessions can quickly choose a path and jump to deeper
documentation.

The SDS section captures the specific proposal:

1. TokenGS predicts clean 3D/4D Gaussian tokens.
2. The tokens render to 2D.
3. Noise is applied in image/video or VAE-latent space.
4. A frozen video diffusion teacher estimates the score/noise.
5. SDS/VSD gradients flow back through the differentiable renderer into the
   tokens and autoregressive predictor.

This preserves clean 3D state and avoids inventing a brittle diffusion noise
process over raw Gaussian parameters.

## Context Checked

- Read `AGENTS.md`; the local instructions point to parent `RTK.md`.
- Read parent `RTK.md`, which is a compatibility shim and points to parent
  architecture/core-goal docs.
- Checked parent `CORE_GOAL.md`, `ARCHITECTURE.md`,
  `docs/ARCHITECTURE_SUMMARY.md`, and the only reminder file. The ChopGrad
  reminder is due 2026-04-23, not today.
- Read the relevant Dyna World research notes and headings to anchor links:
  `KEY_ARCHITECTURE_DECISIONS.md`, `gemini_thread_3.md`,
  `SESSION_Q_AND_A_SYNTHESIS.md`, `single_step/STUDY_NOTES.md`,
  `video_diffusion_loss/STUDY_NOTES.md`, and the diffusion forcing synthesis.

## Key Learnings Update

No `agent_notes/key_learnings.md` update. This was documentation organization,
not a surprising new lesson from an experiment or failure.

# Lyra 2.0 — What's New vs Lyra 1

Reference links:
- Paper: https://arxiv.org/abs/2604.13036
- Repo: https://github.com/nv-tlabs/lyra
- Project page: https://research.nvidia.com/labs/sil/projects/lyra2/

## TL;DR

Lyra 2.0 is **not** just a weights refresh. It's a full pipeline redesign:
new **inference loop** (spatial memory routing) + new **training regime**
(self-augmented histories). The models are new because the training objective
changed, not just because they scaled up.

Lyra 1 was "feed-forward distill video diffusion → 3DGS". Lyra 2 is a
**memory-augmented continuous generation loop** built to hold up under long
camera journeys through large explorable scenes.

## Problems Lyra 2 fixes

When scaling Lyra 1 to big explorable environments, standard video diffusion
backbones break in two specific ways:

1. **Spatial forgetting** — camera turns around, the old region has slid out
   of the temporal context window, so the model hallucinates a new,
   inconsistent version of a place it already "saw".
2. **Temporal drifting** — small autoregressive errors accumulate frame by
   frame; geometry and appearance slowly warp until the scene collapses.

## Key change 1: Inference — Spatial Memory Routing

Inference is no longer `text/image → video → decode to 3D` in one pass.
Instead, the pipeline maintains an active 3D memory:

- **Persistent 3D geometry**: as exploration proceeds, per-frame 3D geometry
  is retained as a structural record of what has already been generated.
- **Information routing / view warping**: when the camera moves to a new
  target viewpoint, the pipeline queries the spatial memory for historical
  frames relevant to that new view.
- **Dense 3D correspondences**: past frames are tied to the new target
  viewpoint via dense 3D correspondences, giving the generator a geometric
  scaffold. The diffusion prior is then used **only for appearance
  synthesis**, not for re-inventing geometry. Revisited regions look the
  way they did when the camera left them.

Net effect: geometry is anchored to memory, not re-hallucinated each step.

## Key change 2: Training — Self-Augmented Histories

The new weights exist because the training objective changed:

- Classical autoregressive video training uses clean ground-truth history.
  That hides drift — the model never sees its own mistakes.
- Lyra 2 deliberately feeds the model its own degraded / error-prone
  outputs during training ("self-augmented histories").
- The model is then supervised to **correct** that drift and realign
  geometry on the fly, instead of passing errors forward.

So the new models are specifically trained to be drift-correcting, not just
bigger / cleaner versions of Lyra 1's distilled generator.

## Why this matters for us (dyna_world_v1)

Relevant to our proposed architectures and video-diffusion-loss threads:

- If we rely on a pure feed-forward video→3DGS distillation path, we will
  hit the same spatial-forgetting and drift walls Lyra 2 was built to fix.
- The "structural scaffold from memory, appearance from prior" split is a
  useful separation-of-concerns pattern for any dynamic-world generator:
  keep geometry deterministic/retrieved, keep the generative prior narrow.
- Training with degraded self-outputs (schedule sampling / self-augmented
  history) is cheap conceptually and worth considering for any
  autoregressive component we build.

## Open questions to chase in-repo

- Exact memory representation: are the retained per-frame "3D records"
  stored as splats, point clouds, depth+pose, or a learned latent volume?
- How is correspondence established at query time — geometric reprojection,
  learned matcher, or both?
- What's the compute/memory cost of the spatial memory as trajectories get
  long? Any eviction / compression strategy?
- How much of the drift-correction gain comes from self-augmented training
  alone vs the memory routing at inference?

Answering these from the paper + repo before deciding how much of this
architecture to borrow for dyna_world_v1.

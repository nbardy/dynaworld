# Expanded README progress

Updated the top-level `README.md` progress section after the user supplied a
more complete project status list.

What changed:

- Split baselines from renderer progress.
- Added completed renderer work: fast differentiable Gaussian rasterizer on Mac
  local GPU, differentiable trainer integration, and debug metrics.
- Added open viewer work for an HTML/WGPU token-format viewer.
- Added pretraining setup tasks for single-camera video data, multi-camera
  data, and scene-cut handling.
- Expanded model architecture tasks with the same-video chunk-mixing idea for
  forcing camera/time separation.
- Added a video diffusion bootstrap section covering score distillation,
  single-step zero-SNR features, and windowing/memory questions.
- Expanded future directions with learning-efficiency comparison against
  standard video diffusion.

No `agent_notes/key_learnings.md` update was made because this records project
status and hypotheses rather than a surprising technical lesson learned through
failure.

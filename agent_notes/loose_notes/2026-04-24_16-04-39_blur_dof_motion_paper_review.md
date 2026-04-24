# Blur / DoF / Motion Paper Review Session

Date: 2026-04-24

Trigger:

User asked whether DynaWorld can account for f-stop, focal length, camera blur,
motion blur, and focal / out-of-focus blur by adding a renderer pass and
inferring camera motion / velocity / focal state from camera tokens. User linked
GeMS, DOF-GS, the DOF-GS paper, and the DOF-GS implementation, and asked for 10
papers plus citation-tree review.

Actions:

- Read local guidance: new mechanisms should route through existing
  `framing_3` / `training_contract_v1` thinking rather than adding a new
  framing.
- Reviewed primary/project sources for Deblur-NeRF, NeRFocus, DoF-NeRF, PDRF,
  DP-NeRF, BAD-NeRF, ExBluRF, Deblurring 3D Gaussian Splatting, BAGS,
  BAD-Gaussians, DeblurGS, DOF-GS, DoF-Gaussian, BARD-GS, and GeMS.
- Queried Semantic Scholar for a partial citation snapshot. The API rate
  limited partway through, so the final research note records the tree as a
  topical dependency graph rather than an exhaustive citation-count graph.
- Added curated note:
  `research_notes/blur_dof_motion_paper_review.md`.
- Indexed the note in `research_notes/README.md` and
  `research_notes/potential_directions_index.md`.
- Added one dense key learning: focal length alone does not determine defocus;
  the useful learned quantity is effective CoC strength plus focus distance.

Current model:

Blur should be renderer/capture state, not baked scene state. Motion blur is
integration over exposure time. Defocus blur is integration over aperture. A
sharp exported world plus inferred lens/shutter state should explain blurry
observations and still render clean held-out sharp/novel queries.

Open implementation follow-up:

- Add an exact sampled renderer path first: sub-shutter camera/time samples and
  finite-aperture samples.
- Add `inv_focus_depth`, `log_Q_eff`, exposure duration, and SE(3) exposure
  trajectory to the camera-token/capture-state contract.
- Add CoC map and screen-velocity diagnostics before training on real blur.
- Only add covariance/kernel shortcuts after synthetic tests prove they match
  the exact sampled renderer.

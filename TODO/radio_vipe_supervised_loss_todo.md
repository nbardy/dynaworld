# Teacher-Supervision Loss TODO

Source:

- Project page: https://be2rlab.github.io/radio_vipe/
- Code: https://github.com/be2rlab/RADIO-ViPE
- Paper: https://arxiv.org/abs/2604.26067
- Score Jacobian Chaining: https://arxiv.org/abs/2212.00774
- Diffusing Differentiable Representations: https://papers.nips.cc/paper_files/paper/2024/hash/4ea6932c9845d6b2cfb89c72b41df3c5-Abstract-Conference.html

## Why This Is Relevant

RADIO-ViPE is not a direct replacement for our trainer. The useful idea is an
offline teacher / auxiliary-supervision lane: take a monocular RGB clip, run a
calibration-free semantic SLAM stack, and use its pose, intrinsics, depth,
feature, and stability estimates as supervised losses for `video => world
tokens`.

The project page claims raw monocular RGB input with no prior intrinsics, depth
sensors, or pose initialization. Its pipeline bootstraps intrinsics with
GeoCalib, extracts dense RADSeg/RADIO-style embeddings, estimates metric depth,
and jointly refines poses, disparities, and intrinsics through dense bundle
adjustment with adaptive robust kernels. That is exactly the kind of imperfect
teacher signal we can use without making it the source of truth.

## Proposed Loss Surfaces

Add an optional `teacher_supervision` path that can read per-frame SLAM teacher
artifacts:

- camera intrinsics and camera-to-world / world-to-camera pose
- relative pose between source and target frames or cameras
- depth or inverse-depth maps, with valid masks
- dense feature maps from RADIO/RADSeg/SigLIP space, possibly PCA-projected
- temporal-stability / dynamic-object weights for robust loss masking

Candidate losses:

- `L_pose_teacher`: SE(3) geodesic/translation loss from our predicted camera
  or relative-pose head to teacher pose.
- `L_intrinsics_teacher`: focal/FOV/principal-point loss when the trainer is
  learning intrinsics.
- `L_depth_teacher`: rendered expected depth or median-depth from splats
  matched to teacher inverse depth, with scale alignment for monocular clips.
- `L_feature_teacher`: render splat features and match teacher feature maps
  with cosine or normalized L2 loss.
- `L_stability_weighted_photo`: weight photometric and feature losses by the
  teacher's temporal-stability/dynamic masks so static regions anchor geometry
  while movable regions use robust kernels.
- `L_cross_view_feature`: when multiple camera views exist, render features
  under source and heldout cameras and require consistency in teacher feature
  space, not only RGB.

## Video Diffusion Teacher Loss

We also want a frozen video-diffusion teacher as an output-side loss. This is
different from using V-JEPA or diffusion activations as input features. The
teacher scores rendered clips and gives a gradient that flows through:

```text
world tokens -> splats -> renderer -> rendered video -> frozen teacher loss
```

There are two useful families:

1. **Deterministic teacher-feature loss, no noise path.** Run the frozen video
   model on rendered video and, where available, matched GT/reference video.
   Compare intermediate features, latents, attention summaries, or predicted
   clean-video embeddings. This is a normal supervised/perceptual loss through
   a frozen teacher. If memory is high, compute only a vector-Jacobian product
   or Jacobian-vector product for the teacher feature map instead of retaining
   the whole teacher graph.
2. **Score/Jacobian pullback, rigorous diffusion path.** Treat the diffusion
   model's score as a vector field in image/video/latent space, then pull that
   score back through the differentiable renderer:

   ```text
   grad_tokens ~= d(rendered_video) / d(tokens)^T * teacher_score(rendered_video)
   ```

   This is the Score Jacobian Chaining / DiffRep-style direction. Prefer this
   framing over naive SDS when the goal is a principled teacher loss. It may
   still use noisy diffusion timesteps internally, but the DynaWorld state stays
   clean: we never noise splat parameters or world tokens directly.

Candidate video-diffusion losses:

- `L_video_teacher_feature`: clean-render teacher features match clean-reference
  teacher features.
- `L_video_teacher_latent`: rendered clip encoded through the frozen teacher
  VAE/encoder matches reference latents.
- `L_video_teacher_score_pullback`: score/Jacobian pullback on source or novel
  camera renders, especially when no novel-view GT exists.
- `L_video_teacher_temporal`: teacher feature consistency over temporally
  dilated frame samples, aligned with `train.frame_sampling`.
- `L_video_teacher_multiview`: render multiple cameras and aggregate teacher
  scores so the optimization cannot satisfy the teacher from only one view.

## First Ablation

Do this as a controlled auxiliary loss, not a new default:

1. Export teacher artifacts for one small clip/sample into a DynaWorld-native
   cache format: `cameras`, `depth`, `features`, `valid_mask`,
   `stability_mask`.
2. Add config-only gates under `losses.teacher_supervision` with all weights
   defaulting to `0.0`.
3. Wire `L_pose_teacher` first for the current multicam relative-pose trainer.
4. Add `L_feature_teacher` second, using the existing feature-splatting path
   and a small projection dimension before the loss.
5. Add a deterministic video-teacher feature loss before a score/Jacobian
   pullback loss; this makes the plumbing debuggable without diffusion sampling
   noise.
6. Select by heldout-camera metrics and validation media, not source-view PSNR.

## Guardrails

- Do not replace photometric RGB/video loss. Treat this as an auxiliary teacher
  that stabilizes geometry, pose, and semantic features.
- Do not let teacher artifacts leak heldout-camera GT into a source-only
  pretraining claim. Separate single-camera teacher supervision from multicam
  heldout evaluation.
- RADIO-ViPE/RADIO features are not necessarily the same target as V-JEPA
  features. If both are present, name the feature spaces explicitly in cache
  metadata.
- Video diffusion teacher losses must not become another source-view shortcut.
  Apply them on heldout/perturbed-camera renders as early as possible.
- If we use score/Jacobian pullback, name whether it is SJC, DiffRep, SDS, or
  VSD. These are not interchangeable implementation details.
- For the "no noise" variant, prefer deterministic frozen-teacher feature or
  latent losses first. Only add score-noise schedules when the clean teacher
  loss has a working regression harness.
- Teacher depth/pose will be wrong sometimes. Robust masks and a loss-weight
  sweep are part of the experiment, not polish.
- Keep the selector as heldout-camera quality. A teacher can improve source
  reconstruction while hurting novel-view geometry; that is a failed ablation.

## Concrete Code Tasks

- [ ] Define `TeacherSupervisionBatch` / cache schema under `src/train/`.
- [ ] Add a small importer for RADIO-ViPE outputs once its artifact format is
      inspected.
- [ ] Add loss config normalization with zero defaults.
- [ ] Add pose-teacher loss to `train_multicam_relative_pose_implicit_dynamic.py`.
- [ ] Add feature-teacher loss through the F32 feature-splatting path.
- [ ] Add deterministic video-diffusion teacher feature loss on rendered clips.
- [ ] Add a score/Jacobian-pullback prototype, with the implementation labeled
      as SJC/DiffRep/SDS/VSD rather than generically "diffusion loss."
- [ ] Add validation metrics: teacher-pose error, teacher-depth error, and
      heldout-camera PSNR/SSIM/L1 deltas, plus teacher-feature deltas.
- [ ] Run a small ablation on the current good DeepView train
      `camera_0006,camera_0014` heldout `camera_0005` config.

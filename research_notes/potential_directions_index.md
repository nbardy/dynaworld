# Potential Directions Index

This is a routing document for Dyna World research directions. Keep dense
arguments, paper notes, and experiment designs in separate files; keep this file
as the high-level map that says what each direction is for, when to pursue it,
and where the deeper documentation lives.

Status labels:

- **Now:** directly supports the current MVP or immediate training velocity.
- **Probe:** worth a small falsification experiment before committing.
- **Background:** important context, but not the next implementation path.
- **Speculative:** long-horizon architecture work; preserve the idea without
  letting it derail the near-term loop.

## Current Research North Star

The near-term product path is a base `video => world token` model. World tokens
are scene-state tokens that decode to splats and stay consistent across novel
camera angles. High-quality dynamic splats, fast experimentation, and
diffusion-assisted refinement all serve that first proof.

The follow-up path is world-token prediction. Once the tokens are stable enough
to be a real scene state, train AR or diffusion models over those tokens for
video continuation, image=>video, and text=>video. The parent project goal keeps
FasterGS as the stable baseline and treats diffusion as a render/refinement
lever: [CORE_GOAL.md](../../CORE_GOAL.md).

## Direction Map

| Direction | Short description | Status | Dense documentation |
|---|---|---:|---|
| Fast rendering and training throughput | Make rasterization, viewport sizing, and training iteration speed stop being the bottleneck. | Now | [key_learnings.md](../agent_notes/key_learnings.md), [trainer cleanup note](TRAINER_INTERFACE_CLEANUP_2026_04_20.md), [rendering session journal](../agent_notes/loose_notes/2026-04-20_13-10-06_jsonc_config_and_viewport_rendering.md) |
| Video-to-world-token base model | Train `video => world tokens` where the tokens decode to splats that remain coherent under source and novel cameras. | Now | [training contract](training_contract_v1.md), [world/splat token philosophy](meta_philosophy/world_splat_tokens_vs_observed_modality_tokens.md), [framing 3](framing_the_problem/framing_3.md) |
| TokenGS-style splat tokens | Decouple Gaussian count and placement from pixels; predict global 3D/4D tokens through a decoder. | Probe | [gemini_thread_3.md](gemini_thread_3.md), [thread_1.txt](thread_1.txt), [KEY_ARCHITECTURE_DECISIONS.md](KEY_ARCHITECTURE_DECISIONS.md) |
| Single-step video diffusion -> splats | Use a frozen/frozen-mostly video diffusion model as a one-step geometry feature extractor, then train splat tokens through render loss. | Probe | [single_step/STUDY_NOTES.md](single_step/STUDY_NOTES.md), [gemini_thread_3.md](gemini_thread_3.md), [SESSION_Q_AND_A_SYNTHESIS.md](SESSION_Q_AND_A_SYNTHESIS.md) |
| Pixel-space SDS from rendered splats | Keep 3D tokens clean, render them to 2D, noise the rendered pixels/latents, and use a frozen video diffusion teacher for score gradients. | Probe | [KEY_ARCHITECTURE_DECISIONS.md](KEY_ARCHITECTURE_DECISIONS.md), [gemini_thread_3.md](gemini_thread_3.md), [diffusion forcing synthesis](chats/diffusion_splat_forcing_apr_20th_gemini/result.md) |
| Diffusion as loss vs diffusion as conditioning | Name the two distinct roles a pretrained video diffusion model plays — input-side feature conditioning vs. output-side score supervision — and design around both, especially for novel-view supervision without multi-view data. | Probe | [meta_philosophy/architecture_design_north_star.md](meta_philosophy/architecture_design_north_star.md), [meta_philosophy/our_problem_core_requirements_and_goals_and_current_philosophy_and_insight.md](meta_philosophy/our_problem_core_requirements_and_goals_and_current_philosophy_and_insight.md) |
| World-token AR/diffusion generation | Predict future or conditional world tokens for video continuation, image=>video, and text=>video after the base tokens pass novel-camera checks. | Speculative | [diffusion forcing synthesis](chats/diffusion_splat_forcing_apr_20th_gemini/result.md), [path B causal notes](proposed_architectures/path_b_causal_autoregressive/01_CAUSAL_STATE_UPDATE.md), [ChopGrad path B](proposed_architectures/path_b_causal_autoregressive/02_CHOPGRAD_TRAINING.md) |
| Physical lens and shutter blur | Model f-stop / focal length / finite-aperture DoF, camera motion blur, and dynamic-object blur as capture-state in the renderer rather than baked scene content. | Probe | [blur_dof_motion_paper_review.md](blur_dof_motion_paper_review.md), [paper corpus](blur_dof_motion_papers/paper_index.md) |
| Video diffusion distillation | Compress high-quality or bidirectional video diffusion teachers into faster students, usually via DMD-style objectives. | Background | [diffusion forcing synthesis](chats/diffusion_splat_forcing_apr_20th_gemini/result.md), [gemini_thread_3.md](gemini_thread_3.md) |
| Rolling / diffusion forcing | Use AR diffusion, rolling-window denoising, attention sinks, and self-forcing to support long or streaming worlds. | Speculative | [diffusion forcing synthesis](chats/diffusion_splat_forcing_apr_20th_gemini/result.md), [path B causal notes](proposed_architectures/path_b_causal_autoregressive/01_CAUSAL_STATE_UPDATE.md), [ChopGrad path B](proposed_architectures/path_b_causal_autoregressive/02_CHOPGRAD_TRAINING.md) |
| ChopGrad and memory-safe pixel losses | Truncate temporal gradients when a recurrent/cached decoder or backbone makes pixel losses scale with sequence length. | Background | [video_diffusion_loss/STUDY_NOTES.md](video_diffusion_loss/STUDY_NOTES.md), [SESSION_Q_AND_A_SYNTHESIS.md](SESSION_Q_AND_A_SYNTHESIS.md), [v1 beta ChopGrad](proposed_architectures/v1_beta/04_CHOPGRAD_INTEGRATION.md) |
| Parallel global vs causal AR decode | Decide whether to emit a whole time-conditioned splat set at once or update a fixed state autoregressively. | Probe | [path A parallel decode](proposed_architectures/path_a_parallel_global/01_PARALLEL_SPLAT_DECODE.md), [path A pros/cons](proposed_architectures/path_a_parallel_global/02_PROS_AND_CONS.md), [path B causal state](proposed_architectures/path_b_causal_autoregressive/01_CAUSAL_STATE_UPDATE.md) |
| Geometry priors and camera grounding | Use depth, VGGT-style features, camera trajectories, or monocular priors to stabilize scale and view consistency. | Background | [thread_2.txt](thread_2.txt), [LYRA_2_NOTES.md](LYRA_2_NOTES.md), [diffusion forcing synthesis](chats/diffusion_splat_forcing_apr_20th_gemini/result.md) |

## Direction Details

### Fast Rendering And Training Throughput

Why it matters: every research path needs cheap render-loss iteration. The
current dense PyTorch renderer is usable for smoke tests, but it is not the
long-term answer for large Gaussian counts or high render sizes.

Known lessons:

- Lowering token count changes model capacity; lowering render/loss viewport is
  the safer smoke-test knob.
- Naive Python tiled rendering was slower than dense PyTorch on the Mac test
  case. Tile-based rendering only pays off with cheap culling/sorting and a
  fused-enough implementation.
- Rendering must use camera extrinsics. Otherwise frame-conditioned Gaussians
  can cheat instead of learning dynamic novel-view geometry.

Promising sub-directions:

- Benchmark or integrate a fused FasterGS-compatible rasterizer path.
- Keep renderer resolution as a first-class viewport knob, with intrinsics
  scaled at the render boundary.
- Add culling/sorting only where profiling shows dense all-Gaussians/all-pixels
  math is the bottleneck.

### Video-To-World-Token Base Model

This is the current first goal. The model consumes observed video and emits
world tokens. Those tokens are decoded to splats, rendered through real camera
paths, and trained against video. The token set is not allowed to be a hidden
view cache: it only counts if it stays coherent when rendered from held-out or
perturbed cameras.

Minimum contract:

- source-camera reconstruction should keep working as the basic adapter test;
- held-out-frame reconstruction should force preimage information into the
  tokens instead of memorizing only encoded frames;
- novel-camera renders should be evaluated for floaters, holes, scale drift,
  and camera-token leakage;
- the exported token format should name camera, time, static scene state, and
  dynamic scene state boundaries clearly enough for a later generator to
  predict them.

Stage 2 depends on this. AR or diffusion over world tokens is only meaningful if
the tokens already behave like scene state. Otherwise the generator is just
learning a video model with extra baggage.

### TokenGS-Style Splat Tokens

TokenGS is not diffusion. It is a feed-forward encoder-decoder where learnable
3DGS tokens cross-attend to image features and directly predict Gaussian
parameters. The core value for Dyna World is decoupling primitive count and
placement from pixels or camera rays.

Use this when the goal is an explicit, editable, renderable 3D/4D asset. Keep it
separate from video diffusion teacher mechanics: diffusion may provide features,
scores, or distillation targets, but the asset lives in Gaussian token space.

Open questions:

- How many tokens are needed for static scenes, short dynamic clips, and
  longer 4D rollouts?
- Which token split works best: static/dynamic masks, polynomial coefficients,
  temporal centers/windows, or AR state deltas?
- Can token tuning adapt a scene cheaply enough to be useful in the product loop?

### Single-Step Video Diffusion To Splats

This path treats a frozen or LoRA-adapted video diffusion model as a one-step
geometry feature extractor. The strongest version is the IT-DiT idea from
`gemini_thread_3.md`:

1. Lock the video diffusion model to a single high-noise / zero-terminal-SNR
   pass.
2. Extract hidden DiT/U-Net activations rather than decoding pixels through the
   video VAE.
3. Let implicit splat tokens cross-attend to those activations.
4. Predict explicit 4D Gaussian parameters.
5. Render with the differentiable splat rasterizer and train on videometric loss.

This is a reconstruction/extraction path, not a from-scratch generator. It
should be tested with a small static probe before dynamic or open-ended claims.

### Pixel-Space SDS From Rendered Splats

This keeps end-to-end training without explicit 3D ground truth, but moves the
diffusion noise into 2D rendered image/video space instead of corrupting 3D
Gaussian parameters directly. In the current plan it is either novel-camera
pressure for the base world-token model or a later generation prior, not a
replacement for the source-camera render contract.

Proposed loop:

1. A TokenGS-style model autoregressively predicts the next clean sequence of
   3D/4D Gaussian tokens.
2. These clean tokens are rendered into a 2D frame or short clip with the
   differentiable Gaussian renderer.
3. Noise is added to the rendered 2D image, video, or VAE latent, depending on
   what the frozen teacher video diffusion model expects.
4. The frozen video diffusion model predicts the noise/score for that noised
   render.
5. SDS/VSD converts the teacher score into a gradient on the rendered pixels or
   latents.
6. That gradient backpropagates through the differentiable renderer into the
   3DGS tokens and the autoregressive token predictor.

Why this is attractive:

- It avoids defining a fragile noise process over mixed 3D Gaussian parameters
  like mean, covariance, opacity, color, and token identity.
- It lets the 3D state remain clean and renderable at every step.
- It can use large frozen 2D/video diffusion priors without requiring paired
  3D ground truth.

Where it likely fits:

- Text/image/video-prior-guided generation when we do not have explicit 3D GT.
- Regularizing long rollouts so rendered clips stay on the video manifold.
- A teacher signal for AR TokenGS when photometric GT is missing or sparse.

Main risks:

- SDS can optimize for 2D plausibility while hiding bad 3D behind the current
  camera path. Multi-view sampling and camera perturbations are mandatory.
- Score gradients may fight exact photometric reconstruction if used on clips
  with known GT frames. It should be weighted as a prior, not blindly treated as
  ground truth.
- Video diffusion teachers may reward temporal texture coherence more than
  physical 3D consistency unless camera/view conditioning is explicit.

### Diffusion As Loss Vs Diffusion As Conditioning

A pretrained video diffusion model can play two distinct roles in this pipeline.
They mostly compose, but the design tradeoffs are different and worth naming.

**Diffusion as conditioning.** Frozen diffusion features (activations,
intermediate latents) feed the splat decoder as input. The diffusion model
shapes what the decoder sees. Examples: IT-DiT single-step activation
extraction, implicit query tokens cross-attending to DiT hidden states. The
model carries prior knowledge into the forward pass.

**Diffusion as loss.** Frozen diffusion scores rendered outputs. Score
distillation (SDS / VSD / DiffRep-style Jacobian pullback) backprops gradients
through the renderer into the splats or tokens. Examples: pixel-space SDS,
noised-latent SDS, rigorous Jacobian pullback. The model evaluates predictions
and provides off-manifold supervision.

Why the distinction matters:

- **Off-manifold supervision.** L1, MSE, LPIPS, and SSIM are averaging losses
  that measure distance to a reference. They cannot push the output onto a
  plausible data manifold; they only pull it toward a specific target.
  Diffusion-score loss is different: it tells the model "this sample is
  unlikely under the data distribution," which catches failure modes
  photometric loss cannot see.
- **Novel-view supervision without multi-view data.** Photometric loss has no
  signal for novel-view renders because no GT frame exists. A diffusion-score
  term can evaluate the novel-view render directly. This makes
  diffusion-as-loss a direct lever on failure mode F2 (cheating splats), and
  an objective-family answer to the novel-view problem that requires no
  architectural factorization.
- **Likely pretraining acceleration.** Averaging losses converge to blurry
  local optima on hard multi-modal data. A diffusion-score term that rewards
  on-manifold outputs may speed early pretraining by forcing the model to
  commit to plausible predictions rather than hedging to the mean.
- **Post-training sweet spot.** Diffusion-as-loss probably shines hardest in
  post-training, once the base model has geometric competence. The score
  refines novel-view plausibility and dynamic realism without the adversarial
  pathologies of a GAN post-training stage.

Design knobs:

- Weight schedule between photometric (precise reconstruction on training
  cameras) and diffusion loss (on-manifold + novel-view plausibility).
  Photometric dominates where GT exists; diffusion takes over where it does
  not.
- Camera perturbation strategy: when and how to render at non-training
  cameras for the diffusion-loss term (every step, scheduled, curriculum).
- Teacher choice: video diffusion for dynamics, image diffusion for per-frame
  plausibility, both.
- Where to apply the score: tokens (via decoder), pixels (via renderer), or
  latents (renderer + VAE encoder).

Risks:

- Score gradients can fight photometric reconstruction; weight as a prior,
  not as GT replacement.
- 2D teachers can reward texture coherence over 3D geometry. Multi-view
  sampling and camera perturbations remain load-bearing.
- SDS has known mode-collapse behavior; VSD or DiffRep-style formulations may
  be necessary in practice.

### Physical Lens And Shutter Blur

This direction treats real camera artifacts as capture-state in the renderer:
finite aperture creates depth-of-field / defocus blur, while shutter integration
creates camera and object motion blur. The target is a sharp exported splat
world plus inferred lens/shutter state, not splats that permanently bake blur.

The key renderer quantities are:

- projection intrinsics (`fx/fy/cx/cy`) and crop/resize state;
- focus distance, preferably represented as inverse focus depth;
- effective CoC strength `Q = F * D = F^2 / f_number`, not raw focal length
  alone;
- exposure duration, shutter curve, and SE(3) camera trajectory during exposure;
- dynamic splat/object trajectories during exposure.

Use [blur_dof_motion_paper_review.md](blur_dof_motion_paper_review.md) before
implementing this, then use the [paper corpus](blur_dof_motion_papers/paper_index.md)
and [extraction notes](blur_dof_motion_papers/extraction_notes.md) for local
PDFs/text, equation anchors, and follow-up formula/dataset tables. The practical
plan is exact sub-shutter/aperture sampling first, CoC/screen-velocity
diagnostics second, and covariance/kernel shortcuts only after synthetic tests
prove they match the sampled renderer.

### Video Diffusion Distillation

Distillation is about making a capable video generator cheap enough for
interactive or training-loop use. The current forcing literature points toward
DMD-style objectives, but the details matter:

- Self-Forcing reduces train/test mismatch by training on model-generated
  history.
- Causal Forcing warns that bidirectional teachers can provide invalid
  trajectory initialization for causal students.
- Reward Forcing uses motion-aware reward weighting to fight static or sluggish
  generated videos.

For Dyna World, this should come after a useful representation path exists. It
is a speed/compression strategy, not the first proof that video priors can
produce stable splat geometry.

### Rolling / Diffusion Forcing

Rolling Forcing and related AR diffusion methods are the long-horizon path:
generate or refine a stream by denoising a moving temporal window while keeping
some form of persistent memory.

**Important framing: novel views are a train/inference distribution gap.** The
forcing family is usually pitched as closing the train/inference gap for AR
rollout (the model sees its own predictions at inference; train on its own
predictions, not GT history). The same logic extends to cameras: at inference
the model is asked for cameras the training distribution never rendered. If
the rollout includes camera perturbations or chunk-swapped-camera predictions,
forcing-family training is not only about temporal drift — it is about
supervising the novel-view distribution itself.

Potential Dyna World translation:

- Clean static tokens are the stable world anchor.
- Dynamic tokens get per-token noise schedules.
- A rolling temporal window jointly denoises several future dynamic-token states.
- Attention sinks or EMA sinks preserve global style, scale, and origin without
  unbounded context growth.
- Rendered frames are emitted as the trailing edge of the window becomes final.

This is the right vocabulary for continuous worlds, but it is still speculative
until the smaller TokenGS and single-step extraction probes work.

### World-Token AR/Diffusion Generation

This is stage 2. Once `video => world tokens` works, train a model that predicts
world tokens themselves:

- video continuation: condition on observed world tokens and predict future
  world tokens;
- image=>video: initialize or condition the first world-token state from an
  image, then roll forward;
- text=>video: condition token generation on text while keeping the renderer as
  the final contract;
- hybrid refinement: use diffusion to denoise token trajectories while AR keeps
  streaming state.

This should not be confused with using a second-stage generator to repair novel
views. Novel-camera consistency is a base-token requirement. Generation is for
unobserved time, new prompts, and conditional creation after that requirement is
met.

### ChopGrad And Memory-Safe Pixel Losses

ChopGrad matters when the path from latent/token predictions to pixels is
temporally recurrent, especially through causal video VAE decoders or cached
video backbones. It is less central when rendering is just evaluating
time-conditioned splat coefficients independently at each frame.

Use it if:

- pixel loss backpropagates through a causally cached VAE/video decoder,
- an AR splat state update requires backpropagation through many prior states,
- or a video backbone is unfrozen across a long temporal graph.

Do not force it into simple global-polynomial rendering unless profiling shows a
real recurrent memory path.

### Geometry Priors And Camera Grounding

Geometry priors are stabilizers, not replacements for render supervision.
Candidates include monocular depth, VGGT-style representation alignment, learned
camera trajectories, and ViPE/FasterGS-style initialization.

Use these to answer:

- What sets metric or relative scale?
- How does the model avoid plausible 2D video with impossible 3D?
- Can the token predictor remain robust when camera poses are noisy or implicit?

## Suggested Next Probes

1. **World-token contract smoke:** define the minimal exported token schema and
   source/novel-camera eval that decides whether a representation counts as
   world tokens.
2. **Renderer throughput probe:** benchmark the current dense path against any
   available fused/FasterGS-compatible path at fixed token counts, render sizes,
   and camera counts.
3. **Static IT-DiT probe:** freeze a small video/image diffusion backbone, add
   TokenGS-style queries, and test whether one-step activations beat a
   from-scratch TokenGS baseline on a tiny static multiview set.
4. **2D SDS toy:** optimize a small clean splat token set from rendered SDS
   gradients with multiple camera perturbations; verify that geometry improves,
   not just the teacher-view render.
5. **World-token generation paper pass:** compress the AR/diffusion forcing
   synthesis into a smaller implementation note, but keep it behind the base
   token consistency proof.

## Maintenance Rules

- Add new ideas here only as short route entries. Create a dense note when the
  idea needs derivations, experiment design, or paper-level detail.
- If an experiment changes a status label, update the table and point to the
  loose note or dense research note that justifies the change.
- Do not duplicate long arguments from the linked docs. This file should stay
  skimmable enough to choose the next research thread in a few minutes.

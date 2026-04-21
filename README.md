# dynaworld

**Hollywood's Open Source World Model.**

Dynamic video => compressed splats.

DynaWorld is less about making new worlds and more about modality shifts.
It lets you go from `Video <=> splats` both directions, so you can use splats
when they're best and video when its best.

The first use case is camera change. You shot the footage. Now you want a
different angle. DynaWorld turns that video into a compact dynamic splat scene,
lets you move the camera inside it, then renders video again.

The second use case is special effects. Traditional physics pipelines or
algorithmic special effects can run on the splat representations. As powerful
as video diffusion models can be for special effects, sometimes mathematical
representations are better when we want the control of traditional generative
algorithms.

This is made to be used in conjunction with video diffusion for upscaling,
editing, and final-pixel cleanup. They are complementary, not opposites.
Video diffusion is the generative portion. DynaWorld is focused on the
exploratory portion.

You can shoot a clip, or generate one with video diffusion. Then DynaWorld
creates an exploratory version of that clip.

## Video `<=>` Video

A world model needs to be `Video <=> Video` so it can train self structured on
video, unsupervised or self supervised.

`Video <=> Video` is the only training data for world models that scales.
Everything else requires expensive labeling, and labels don't scale.

The training contract is simple:

1. encode video
2. decode splats
3. render video
4. compare to the video you started from

Splats sit in the middle as the compact intermediate. No fake 3D labels. No
synthetic ground truth. Render splats and compare directly against
ground-truth video.

The useful signal is in the preimage.

## Cheap Adapters

Modalities don't require pretraining. The implicit latents in video models are
the key. Decoding them to splats is cheap adapter training.

A lightweight splat head on a frozen video backbone. Not a new foundation
model.

## Generative Capacity

There is not a generative phase planned for generating the initial world. Leave
that phase to world models that generate the starting world.

DynaWorld still needs generative capacity. Novel camera angles mean the model
has to know whats back there, even if the input has not seen it. Looking behind
an object is hallucinating some content. Extending the background is
hallucinating some content. But it is not fully generating in time.

For now, keep the model focused. It only generates novel angles.

The stronger training task is probably to force the model to render images it
didn't encode. Encode part of a clip, decode splats as a function of `t`, then
train on the GT of images it didn't encode. That forces data that is not in the
encode path to come from the world model, the 3D inductive bias in the splat
renderer, and the time-conditioned decoder.

## Core Beliefs

See `research_notes/README.md` for the long-form rationale.

1. World models are video models. A strong video backbone already carries geometry, motion, and lighting structure.
2. DynaWorld is less about making new worlds and more about modality shifts.
3. `Video <=> Video` is the training contract. It is the only training data for world models that scales. Splats sit in the middle.
4. The useful signal is in the preimage.
5. Static and dynamic are the same problem.
6. Foundations are sacred. Modalities don't require pretraining. A lightweight splat head on a frozen video backbone is cheap adapter training.
7. Supervision should stay in pixel space. Render splats and compare to video.
8. Memory should be spent on dynamic scene state, not luxury parameters.

## Phases

**Phase I - `Video <=> splats`.** Dynamic reconstruction and fast camera
editing. Where we are today.

**Phase II - Interaction.** Actions inside the world model. Agents and physics
handles that let you control the dynamic scene, not just re-view it.

No planned Phase III for text-to-world generation. Generate new dynamic worlds
somewhere else, then use DynaWorld to make them exploratory.

## Progress

Current work is focused on small single-video overfit runs. The goal is to keep
training loops working, fast, and convergent before moving to larger data.
Completed items here have been tested on small single-video overfit runs unless
noted otherwise.
Longer-form research notes live under `research_notes/`.

### Baselines

- [x] Top-level video to splat baseline, reproducing TokenGS as the reference baseline.
- [x] Implicit camera baseline, extending the TokenGS baseline.

### Renderer

- [x] Fast differentiable Gaussian rasterizer on Mac local GPU for local experimentation.
- [x] Differentiable renderer integration set up for trainer loops.
- [x] Debug metrics added to trainer loops.
- [ ] Attach to a video diffusion model for video diffusion features.
- [ ] Run single-step Marigold-style features for maximum information extraction.

### Viewer

- [ ] Set up an HTML viewer that can open the token format, load the MLPs, bake splats, sort them, and render with WGPU.

### Pretraining Setup

- [ ] Collect diverse single-camera video datasets for the first pretraining pass.
- [ ] Collect multi-camera data for novel-view-synthesis finetuning.
- [ ] Decide whether scene cuts should be marked and split during preprocessing.

### Model Architecture

- [ ] Sort out how to handle time.
- [ ] Support longer videos and sliding-window training.
- [ ] Better support novel camera angles when training mostly from the input camera angle.
- [ ] Separate camera and video representations well enough that camera changes do not collapse into video-embed leakage.
- [ ] Test the direct path: pretrain on single-camera source video, then finetune on paired camera data so the model can encode one view, swap the camera token, and decode another view.
- [ ] Find pretraining pressure that encourages camera-token swapping behavior before paired-camera finetuning.
- [ ] Test same-video chunk mixing: encode two chunks from the same video, combine the first chunk's video tokens with the second chunk's camera token, and train against the second chunk's ground truth so shared video tokens learn to render under a different camera path and time.
- [ ] Turn each clip into "multi view" with a crop and perspective warp. Classic videography trick where you take high resolution footage, crop a corner, and rescale so the perspective looks like it is in the center of each frame. Feels valuable, but downsides: it is still from the same angle, and too much perspective warp is a bit cheap and does not exactly align to GT camera data.
- [ ] Try the crop variant without perspective warp: shift the rays so the crop is defined as a camera extrinsic. More honest, but then it is sort of learning crop shift only, and might not generalize as well to non-crop shifts where the shift is not the center of the camera.
- [ ] Try doing both crop variants plus chunk mixing together in pretrain and see if that is enough to get a good prior.
- [ ] Worry about the wrong task forcing the camera implicitly into image tokens if we try to hide camera position too much in non-principled ways.
- [ ] Try a BERT-like random masking dropout scheme in pretrain as an alternative to only chopping the video in half. Might be more robust, but worry that it will force the camera data to hide itself in the image tokens.
- [ ] Stop and reflect on the bigger architecture / objective question: AR vs diffusion, rolling vs forcing diffusion, and the rolling window stuff. We need to more natively extend to partial / long context and rolling context. Maybe even AR on tokens per frame. End up robust to noise at inference time by training on noisier data. Get away from the single encoder => decode paradigm toward something more elegant.

### Novel View Post-Training

- [ ] Second stage post-training: render novel passes and train a GAN on them, so we do a GAN for novel and non-novel views. It has to learn to make them both look the same.
- [ ] Maybe some sort of reward style training here as well. See Chopgrad ([arxiv 2603.17812](https://arxiv.org/abs/2603.17812)) which recently did really high quality correction.
- [ ] The plan is a bit of both: (1) give the model some prior off-camera capability in pretrain, (2) refine that in post training.

### Video Diffusion Bootstrap

- [ ] Evaluate whether score distillation is useful by noising rendered output images or using a differentiable diffusion technique to push gradients back into the renderer.
- [ ] Test direct video features from a single-step zero-SNR schedule.
- [ ] Work out how video diffusion features interact with windowing and memory limits.

### Future Directions

#### Auto-Research

- [ ] Set up auto-research swarm configs.
- [ ] Keep local Mac shader support fast enough that contributors can run this locally without cloud GPU cost.
- [ ] Set up "World Model at Home".
- [ ] Document how to contribute auto-research so users can run local experiments and contribute findings back.
- [ ] Investigate async training across users' home GPUs.

#### Foundation Model Training

- [ ] Benchmark whether splat decoding is a useful inductive bias for video generation itself versus standard video diffusion.
- [ ] Compare learning efficiency from scratch for splat-decoding video models against standard video diffusion baselines.

## Setup

```bash
uv sync
git submodule update --init --recursive
```

## Camera Prebake

From the repo root:

```bash
./src/train_scripts/get_camera.sh
```

Default inputs and outputs live under `test_data/`.

## Train

Single-image baseline:

```bash
uv run python src/train/tokenGS.py src/train_configs/local_mac_overfit_single_image.jsonc
```

Current recommended local dynamic run:

```bash
./src/train_scripts/train_full_dynamic_with_camera_prebake_all_frames.sh
```

This defaults to the 128px/4fps known-camera fast-mac-gsplat v5 baseline:

```bash
src/train_configs/local_mac_overfit_prebaked_camera_128_4fps_wide_depth_fast_mac_8192splats.jsonc
```

It uses 128 learned tokens, 64 Gaussians per token, native batched fast-mac v5
rendering, low-opacity randomized scale init, Adam, cosine LR decay, and
gradient clipping. That is the current "make it work decently first" baseline,
not the final quality target.

Taichi remains available for comparison:

```bash
./src/train_scripts/train_full_dynamic_with_camera_prebake_all_frames.sh \
  src/train_configs/local_mac_overfit_prebaked_camera_128_4fps_wide_depth_taichi_8192splats.jsonc
```

Legacy 32px dense comparison:

```bash
./src/train_scripts/train_full_dynamic_with_camera_prebake_all_frames.sh \
  src/train_configs/local_mac_overfit_prebaked_camera.jsonc
```

Smaller 64px/4fps comparison:

```bash
./src/train_scripts/train_full_dynamic_with_camera_prebake_all_frames.sh \
  src/train_configs/local_mac_overfit_prebaked_camera_64_4fps.jsonc
```

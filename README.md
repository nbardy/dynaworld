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

## Status

Local Mac overfit baselines today. Real dataset and scaling next. The
longer-form research notes live under `research_notes/`.

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

Dynamic training with DUSt3R camera prebake:

```bash
./src/train_scripts/train_full_dynamic_with_camera_prebake_all_frames.sh
```

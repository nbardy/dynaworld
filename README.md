# dynaworld

Dynaworld is an Open Source World Model.

There are many ideas of what world models are and many different uses for them. Common examples are "text to static world" systems such as World Labs, or "actions inside a world" systems such as Genie.

World models also take a stance on what comes into the model and what comes out of it: text or image in, text or image out. In our case, the contract is simple:

`Video in -> splats out`

This is uniquely suited to novel view synthesis on existing videos. The goal is to let videographers edit their existing footage in real time with fast local splat rendering.

There is another type of world model that is still largely ignored: dynamic video to splats.

How can we ingest a dynamic video and output a scene of Gaussian splats that lets us explore and move around a dynamic video from novel camera angles? We want to change the camera inside a dynamic world, not just a static world. And we want to condition on real-world video, not generate novel worlds from scratch.

This lets us jump into our favorite existing worlds through videos of what is actually happening there.

Phase I of Dynaworld focuses on dynamic `video in -> splats out`. Phase II extends that foundation toward actions inside the world model, but the first job is to make dynamic novel-view video editing work.

## Core Beliefs

- World models are video models. A strong video backbone already carries geometry, motion, and lighting structure.
- The contract we care about is `video in -> splats out`, not text-conditioned world generation.
- Foundations are sacred. We want lightweight splat heads on top of frozen or mostly frozen video models, not a new foundation model from scratch.
- The useful signal is in the preimage. A single forward pass through a video model can expose enough spatiotemporal structure to decode dynamic splats.
- Supervision should stay in pixel space. We render splats and compare directly against ground-truth video instead of relying on blurry latent-space losses.
- Memory should be spent on dynamic scene state, not luxury parameters. Compact splat parameterizations matter if we want long videos to fit and train.

The longer-form research notes and prompt scaffolding live under `research_notes/`.

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

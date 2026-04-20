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

## Setup

```bash
uv sync
git submodule update --init --recursive
```

## Camera Prebake

From the repo root:

```bash
./get_camera.sh
```

Default inputs and outputs live under `test_data/`.

## Train

Single-image baseline:

```bash
uv run python train_scripts/tokenGS.py
```

Dynamic training with DUSt3R camera prebake:

```bash
./train_full_dynamic_with_camera_prebake_all_frames.sh
```

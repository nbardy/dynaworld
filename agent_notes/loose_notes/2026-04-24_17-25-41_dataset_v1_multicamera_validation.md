# DATASET_V1 Multi-Camera Validation

The user asked to capture the dataset direction after the YouTube curated-span
follow-up and pointed to AIST Dance DB as a better validation source.

Wrote `DATASET_V1.md` to separate two roles:

- scene-distinct YouTube clips stay as a tiny local Mac smoke/generalization
  dataset;
- synchronized multi-camera datasets become the real validation track because
  they provide same-time held-out camera ground truth.

Key contract:

```text
original timeframe + source camera(s) -> held-out novel camera, same timeframe
```

Then compute SSIM/PSNR/LPIPS and frame-distribution FID against the actual
held-out frames.

AIST Dance DB is a strong first validation candidate because its official docs
describe fixed cameras `c01-c09`, moving camera `c10`, and filenames that encode
the camera and sequence factors. The first pass should use fixed cameras only
and hold out cameras inside validation sequences, while still splitting train
and validation by sequence identity to avoid motion/lighting/dancer leakage.

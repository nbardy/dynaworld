# Taichi Mac Docs Wrapup

We tightened the Taichi fork documentation so future agents do not have to infer
the winning path from benchmark chatter.

Taichi submodule:

- `third_party/taichi-splatting/README.md` now starts with the install/import/run
  path for the fastest Mac configuration:
  `kernel_variant="metal_reference"`, `sort_backend="auto"`,
  `backward_variant="pixel_reference"`, `metal_compatible=True`.
- The README records the fastest final variants we measured:
  `metal_reference` + auto sort was the best working Mac path, while
  `bucket_taichi`, `taichi_field`, and `ordered_taichi` were correctness or
  reference paths but not speed wins.
- `third_party/taichi-splatting/NOTES.md` now carries the future-agent notes:
  what worked, what failed, accuracy status, synthetic timing snapshots, key
  pressure findings, and the next likely architecture.

Important distinction:

- The Taichi fork is a real submodule at `third_party/taichi-splatting` and
  points at `nbardy/taichi-gsplat-differentiable-render-mac`.
- The raw fast Mac renderer work remains separate from this Taichi fork and was
  intentionally not touched in this wrapup.

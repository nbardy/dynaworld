# Browser splat coverage and validation cadence

- The first 1.5-second validation cadence made charts much denser, but a live run
  fell from roughly 878 to 697 completed steps/s while validation snapshots were
  frequent. Full validation now runs every 2,048 completed steps. This keeps chart
  samples evenly spaced in optimizer-step space and prevents faster training from
  increasing the validation duty cycle.
- The multicamera 3D renderer intentionally clears to black and shows only model
  output. Compositing each camera's temporal-mean target behind it would make the
  preview look complete but would leak heldout pixels and misrepresent quality.
- The fixed `0.55 * geometryScale` initialization made all points share one world
  radius even though the exported SfM cloud has strongly nonuniform spacing. Splat
  radii now use 0.8 times nearest-neighbor distance, clamped to the optimizer's
  existing world-scale range. This follows the usual 3DGS scale-init principle and
  gives sparse cloud regions broader initial support without increasing primitive
  count or training work.
- Capacity remains a real limitation: the current shared-memory backward has a
  hard 768-splat tape. Raising the UI slider without changing that kernel would be
  incorrect; a tiled/chunked compositor and backward is still required for useful
  multi-thousand-splat capacity.
- Live profiling separated optimizer and preview cost: the adaptive-radius run
  measured about 379 steps/s while rasterizing three Retina-width panels at 20
  FPS and about 812 steps/s with preview disabled. The internal three-panel
  framebuffer is now capped at 960 pixels wide (320 per panel) while CSS keeps
  the same layout. That is still substantially above the 96x72 training data and
  avoids spending most GPU time supersampling a diagnostic preview.
- The backward now retains the shared-memory alpha tape through 768 splats and
  uses a storage-backed tape above that portable limit. The canonical browser
  bundle was re-exported with 1,536 distinct train-visible SfM seeds and the UI
  exposes 96-1,536. Live checks measured about 793 steps/s at 768 and 374 at
  1,536; 1,536 was denser but slightly worse at matched wall time while all
  footprints remain isotropic, so 768 stays the default.

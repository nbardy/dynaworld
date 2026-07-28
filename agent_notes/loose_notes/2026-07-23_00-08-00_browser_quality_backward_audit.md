# Browser quality and backward audit

- User reported that the now-real 17-camera trainer moved splats but looked much
  worse than the earlier browser demo. Live state at 34k steps was finite and
  plateaued around train/heldout PSNR `13.6/12.9`, so this was not a null-loss
  recurrence.
- Visual inspection found the render billboard used a `6r` quad with a
  `local=3q` Gaussian, giving an effective sigma of `2r`; training used sigma
  `r`. The render quad is now `3r`.
- Backward used the mixed temporal gate where the derivative requires only the
  dynamic Gaussian core, and omitted the path from camera-space depth through
  projected radius. Both derivatives were corrected.
- SfM seeds were composited in farthest-point-selection order. Initialization
  now sorts the selected points far-to-near for the anchor camera. Other camera
  views still need conditional depth order.
- Clean probes at similar 8-9k steps moved from about `13.1/12.8 dB` to
  `13.3/13.2 dB` train/heldout; heldout SSIM reached `0.394`.
- The legacy image-space shader remains in `trainerWebGpu.js`. It also bound ten
  storage buffers without requesting the limit, so it now negotiates limits and
  fails fast on WGSL/pipeline errors. `benchmarkLegacy2d.html` makes it an
  executable baseline. Its 64-step source-view run started at PSNR/SSIM
  `33.58/0.9956` and ended at `33.47/0.9955`; the high visual quality is mostly
  the source camera's temporal-mean target used as a background shortcut.
- The main honest quality ceiling is now representation/support: 768 isotropic
  splats over black, no per-view depth sorting, no view-dependent color, and no
  densification/pruning. The next useful implementation is a tiled active-list
  backward that permits thousands of splats and conditional depth order, not a
  larger LR.

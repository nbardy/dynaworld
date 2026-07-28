# Browser worker real-training fix

- Live QA exposed that the apparent 750-880 steps/s path initially had zero
  sampled loss and unchanged validation. The GPU commands were no-ops: the
  multicamera gradient kernel has nine storage buffers, while the requested
  WebGPU device retained the default limit of eight.
- The trainer now requests `maxStorageBuffersPerShaderStage: 9` after checking
  adapter support, and fails fast on WGSL compilation or pipeline validation
  errors. This turned the previously silent invalid pipeline into real training.
- The worker now bounds submitted work to 32 steps. Training remains in its own
  worker and never awaits metrics or validation, but an unbounded GPU queue can
  no longer starve Pause, sampled-loss copies, or parameter snapshots.
- Sample telemetry reads the update shader's 16-byte aggregate stats record
  rather than mapping the full per-sample loss buffer. Validation computes
  parameter delta in its own worker.
- Final Apple browser QA: 887 completed steps/s, 60 FPS UI, sample loss 0.01382,
  train PSNR 13.2 dB, heldout PSNR 13.0 dB, train SSIM 0.425, heldout SSIM
  0.371, and mean absolute parameter delta 7.22e-2. Pause responded promptly,
  the loss log toggle changed state, and no browser warnings/errors were seen.

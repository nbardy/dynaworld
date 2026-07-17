# Gate4 WorldFoam Reflection Stop

We stopped implementation after the Gate4 affine moving-camera WorldFoam lane reached a useful but still scoped result.

What is real:

- The mixed affine slab tape is effectively flat across frame count on the 2/4/8/16 synthetic moving-camera harness: train mixed tape storage is 1.121612 MB -> 1.112348 MB while explicit per-frame ray storage is 0.098304 MB -> 0.786432 MB.
- The latest normal `direct_atomic_grad_only` repeat20 train/eval artifact verifies cleanly with median timing required:
  - artifact: `research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_gradonly_repeat20_render32_site12_2_4_8_16.json`
  - verifier: `research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_gradonly_repeat20_render32_site12_2_4_8_16_verifier.json`
  - total mean scale 2f->16f: 1.156x
  - total median scale 2f->16f: 1.142x
  - backward mean scale 2f->16f: 1.087x
  - backward median scale 2f->16f: 0.967x
  - train PSNR rises slightly from 13.845 to 13.998; heldout PSNR rises from 14.288 to 14.592.
- The owner-update VJP mode now has an autograd train/eval path and a stricter aux-loss proof that alpha/depth output adjoints are nonzero:
  - artifact: `research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_ownerupdate_rgba_depth_aux_render32_site12_2_4_8_16.json`
  - verifier: `research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_ownerupdate_rgba_depth_aux_render32_site12_2_4_8_16_verifier.json`
  - total mean scale 2f->16f: 1.126x
  - backward mean scale 2f->16f: 0.949x
  - alpha output adjoint abs sum is about 0.02 for every frame count; depth output adjoint abs sum is about 1.6e-4.
- The focused Gate4 unit/verifier suite passes: 28 tests OK.

What is not proven:

- This is not "WorldFoam proper" and not a full dynamic Gaussian trainer. The verifier scope explicitly stays frozen-geometry, site-RGBA, synthetic moving-camera train/eval.
- The speed is sublinear in the harness, but absolute MPS timings are still tens of milliseconds per step, not STAR-UVT-clean. The result is algorithmically encouraging but not yet competitive on wall-clock.
- The owner-update aux path proves gradients can flow through alpha/depth VJP seeds, but quality is lower than the normal RGB-only direct atomic mode on this tiny harness.
- This does not yet prove real-scene 4K scaling, CUDA parity, large splat count behavior, or paper-level quality.

Interpretation:

The STAR idea did port partially: frame count can be moved out of the dominant storage term by compiling a mixed affine slab tape instead of carrying per-frame rays. That is the key win. The weak point is that WorldFoam still has heavier per-candidate/replay/control overhead than STAR UVT, so the asymptotic story is better than the current wall-clock story.

Good next step after this stop:

Before adding more math, profile the absolute-time gap. The right question is no longer "can the mixed tape be sublinear?" but "why is the compiled mixed tape still 60-130 ms/step at render32/site12 when STAR UVT is much cleaner?" Focus on kernel launch count, replay loop structure, CPU/MPS sync points, and whether owner/candidate work can be fused further.

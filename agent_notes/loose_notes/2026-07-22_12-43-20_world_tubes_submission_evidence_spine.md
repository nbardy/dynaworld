# World Tubes submission evidence spine

## Goal

Collapse the many Gauged UVT, STAR UVT, WorldFoam, and dynamic-3DGS offshoots
into one falsifiable paper: **World Tubes in Gauged Camera Space**, with
projective STAR UVT as the implementation, per-frame STAR replay as the causal
scaling baseline, WorldFoam as the retained-depth challenger, and dynamic 3DGS
as the conventional baseline.

## Work completed in this chunk

- Added evidence schema v1 across all three lanes: PSNR/SSIM/L1/LPIPS,
  serialized storage, parameter and optimizer bytes, sampled MPS/CUDA memory,
  synchronized phase timing, target/rasterized cost, and representation
  diagnostics.
- Added the fail-closed matrix runner and generated JSON/CSV/Markdown/LaTeX/SVG
  artifacts from a live three-lane smoke.
- Completed and verified the exact `F=4..128` replay-versus-compiled scaling
  row. A first cap-128 attempt correctly failed overflow; cap 256 passed.
- Generated the synthetic theorem table from certified fixture reports and the
  new scaling artifact. Narrowed the manuscript to bounded chart segments
  instead of claiming an unimplemented full `360/720` chart transition.
- Converted the 848-line Markdown paper into a standalone LaTeX manuscript.
- Added controlled D-NeRF download/inspection plumbing and started acquiring
  two additional Neural3D scenes.
- Removed the unrelated V-JEPA git checkout from the paper dependency lock;
  LPIPS is now reproducibly locked.
- Committed the STAR comparison changes inside the nested STAR repository as
  `eee22e4`.

## Live full-row evidence

The progressive 512-wide Coffee Martini seed-17 comparison completed 600
steps for World Tubes and dynamic 3DGS. Heldout results were poor but valid:

- World Tubes: PSNR `5.8945`, SSIM `0.03360`, LPIPS `0.98461`, train wall
  `124.58s`, peak MPS driver memory `3.114GB`.
- Dynamic 3DGS: PSNR `4.9110`, SSIM `0.28266`, LPIPS `0.90229`, train wall
  `142.58s`, peak MPS driver memory `20.557GB`.

Do not tune these away or promote them prematurely. They are one seed of the
frozen protocol and need the WorldFoam lane, repeat seeds, controls, and breadth
before interpretation. The first orchestration attempt was interrupted only
after its completed comparison because W&B tried to upload a 33MB dirty-tree
diff. The comparison artifact was preserved; orchestration resumed with
`--reuse-existing`, and future paper runs disable diff/code upload while
recording exact clean commit hashes.

The first WorldFoam attempt exposed a second orchestration cost: the inherited
`steps/6` image cadence performs a full 300-frame train/heldout evaluation and
video encode at every checkpoint. The first step-100 artifact pass took about
eight minutes, making the seven-row matrix dominated by redundant media. That
attempt was preserved as `worldfoam_interrupted_redundant_eval_cadence` and
stopped. Paper cadence is now initial plus final only (`image_log_every =
video_log_every = steps`), retaining the clean-init versus trained selection
boundary while removing five duplicate full-video passes. W&B ids are now
deterministic/resumable so an orchestration retry updates rather than silently
forking provenance.

## Newly exposed correctness boundary

The unified dataset contract was generic, but WorldFoam initialization still
came from a Coffee Martini point cloud. The runner now makes initialization
explicit (`base_config`, `video`, or scene-specific PLY), records it in the run
summary, and fails if a requested point cloud is absent. Breadth rows may not
silently reuse Coffee geometry.

## Stop rule

No browser, V-JEPA, feature-token, Softmax, new-gauge, direct-serial, native
WorldFoam, native-resolution, or external-SOTA branch is allowed to displace
the frozen matrix, breadth adapter, manuscript, and reproducibility package.

## Breadth preparation follow-up

- Added two camera-split protocols chosen from the actual LLFF optical-axis
  angles, not numeric camera adjacency: `cam13/cam18 -> cam00` has
  `34.42°` train separation and balanced `17.28°/17.23°` heldout distances;
  `cam02/cam07 -> cam12` has `42.66°` and `21.19°/21.53°`. Together with the
  existing `cam04/cam09 -> cam06` split, these cover three balanced triplets.
- Extended the existing train-only feature-triangulation builder with a
  `--paper-protocol` dataset override, so each new split can produce its own
  clean WorldFoam initializer without duplicating the PowerFoam config.
- The first `cook_spinach` download stream failed near 1.1GB. The shared HTTP
  helper previously restarted partial files; it now resumes verified `206`
  ranges, rejects invalid ranges, and restarts safely only when a server ignores
  the Range header. Focused tests cover both behaviors, and the live retry
  continued growing the existing `.part` file.

## Gauged compiler boundary and final-eval repair

- Clarified the non-negotiable method boundary in the manuscript and
  `CODE_ORGANIZATION.md`: STAR UVT is the spacetime-Gaussian/Metal lineage;
  gauged camera space is the compiler semantics layered on it. The latter owns
  camera-ray pullback, depth-fiber marginalization with retained conditional
  depth, projective gauge domains, visibility-event stratification, and local
  fallback. The saved stress fixture's raw crossing has `0.186742` error while
  the two-stratum repair has zero error. Cleanup must not flatten those into a
  plain STAR alias.
- The first full WorldFoam 600-step preflight reached step 600 but disappeared
  during eager final evaluation before writing `paper_protocol_summary.json`.
  The old path concatenated both full 300-frame float32 train and heldout
  renders, targets, alphas, metric temporaries, LPIPS inputs, local videos, and
  W&B videos. It was bounded on MPS per render call but unbounded on CPU after
  concatenation.
- Paper evaluation now accumulates full-frame L1/MSE/PSNR/SSIM and final
  heldout LPIPS online in bounded batches. Only 32 uniformly sampled frames are
  retained for media; metrics still cover every frame. A
  `checkpoint_pre_final_eval.pt` is atomically written before the expensive
  final pass.
- A fresh two-step three-lane MPS smoke completed through the new path at
  `/tmp/world_tubes_stream_eval_smoke_v2`, including full evidence, pre-eval
  and final checkpoints, final train/heldout videos, the WorldFoam protocol
  summary, and the unified run summary. The streamed metric accumulator matches
  the original full-clip implementation to `1e-6` in its focused regression.

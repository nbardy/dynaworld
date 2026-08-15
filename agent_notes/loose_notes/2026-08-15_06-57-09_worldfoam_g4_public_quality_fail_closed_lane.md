# WorldFoam G4 public-quality fail-closed lane contract

## Outcome

G4 is now specified as a real ablation rather than a generic test or a relabelled
PowerFoam row. The checked-in contract is exactly:

- three calibrated Neural3D scenes: Coffee Martini, Cook Spinach, and Cut
  Roasted Beef;
- seeds `17/29/43`;
- four paired routes: compiled `worldfoam_native4d`, same-representation
  framewise WorldFoam replay, World Tubes, and dynamic 3DGS;
- fixed `384x512`, 300 frames, 300 optimizer steps, four sampled frames per
  step, 1,024 primitives/sites, and `235,929,600` target pixels;
- 36 fresh-process measured rows total.

The two additional fixed-512 scene protocols make the public scene breadth
pixel/schedule matched without pretending that native4d already supports the
progressive stage-transition transaction.

## Implemented

- `src/train_configs/paper_protocols/worldfoam_native4d_g4_public_quality_v1.jsonc`
  freezes routes, scenes, seeds, budgets, host guard, and numerical acceptance.
- `research_experiments/world_foam_lane2/run_worldfoam_public_quality_ablation.py`
  builds the full 36-command plan, aborts before the first baseline if any
  production route is unavailable, runs one fresh process at a time when
  explicitly authorized, and can collect only a complete row set.
- `research_experiments/world_foam_lane2/verify_worldfoam_public_quality_ablation.py`
  independently reopens and hash-checks every raw receipt, protocol, dataset
  manifest, final checkpoint, full-temporal heldout video, and W&B run file.
  It recomputes paired acceptance and rejects smoke, proxy, simulated,
  fake-native, source-only, procedural-target, training-view-only, dirty-source,
  non-final-checkpoint, and representation-mismatched evidence.
- `generate_worldfoam_public_quality_assets.py` deterministically emits the
  verifier-derived `g4_public_quality_table.tex`, scene/seed PSNR SVG, CSV, and
  JSON summary, and independently rebuilds them byte-for-byte. It refuses to
  run until the complete G4 artifact is accepted.
- The strict Paper-B ICLR verifier now invokes this independent G4 verifier and
  requires exactly 36 rows; a ledger label plus `public_quality_evidence=true`
  is no longer enough.
- Focused verification: eight G4 contract/asset tests pass, including adversarial
  hash rebinding after proxy/fake-native/representation mutation. Two strict
  Paper-B fixture gates also pass with an independently injected 36-row G4
  verifier.

## Exact production blockers

The dry plan correctly reports `runtime_ready=false` and launches no work:

1. `worldfoam_target_dataset_binding.py` accepts only `target_split=train`.
2. `neural3d_mapped_rgb8_adapter.py` emits only the train split; there is no
   identity-sealed heldout-camera target/evaluator bridge.
3. `worldfoam_training_memory_ablation_adapter.py` uses
   `_ProceduralDirectSelectedPixelTargetSource` and explicitly emits
   `public_quality_evidence=false`.
4. The production full-geometry adapter is synthetic-only even apart from its
   procedural target source.
5. There is no `train_worldfoam_native4d_public_quality_row.py` production
   worker or runtime-verified capability receipt.
6. No public-native4D runtime capability receipt exists.
7. The current fused-slab binary is older than its native sources and must be
   rebuilt and schema-attested before the capability receipt can exist.
Separately, the existing unified `worldfoam` lane remains per-frame
`MetalPowerFoamVideo`; it is neither compiled native4d nor a valid
same-representation native4d replay control, so it cannot be substituted for
either missing WorldFoam route.

The runner checks all of these before invoking any subprocess, so it cannot
leave completed Gaussian baselines that look like a partial G4 experiment.

## Next implementation sequence

1. Extend the mapped target binding/cache with a distinct heldout split and
   bind it to the same calibrated camera generation and physical frame grid.
2. Replace the procedural target source at a new public adapter boundary; do
   not weaken the memory-ablation adapter's synthetic claim.
3. Add bounded forward prediction and full-temporal heldout PSNR/SSIM/LPIPS/L1
   evaluation from the final checkpoint.
4. Implement the two WorldFoam route modes against the identical retained-depth
   world: compiled shared adjoint and sequential framewise replay.
5. Rebuild and attest the real Metal extension, then emit the exact worker
   capability receipt only after both modes are runtime verified.
6. Run the 36 rows on a quiet, clean host. Collect and verify the artifact
   before allowing the Paper-B generator to replace the G4 placeholder.

No Metal, MPS, native rebuild, or public training was launched in this work.

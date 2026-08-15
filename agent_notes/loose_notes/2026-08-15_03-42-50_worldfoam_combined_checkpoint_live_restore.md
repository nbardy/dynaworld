# WorldFoam combined-checkpoint live restore

## Scope

Closed the fresh-process restart seam for the fixed-camera combined
material/geometry state. No MPS workload, native build, or publication ablation
was run.

## Implemented

- `restore_paper_kinetic_fixed_camera_combined_generation_from_payload(...)`
  validates the payload under the caller-owned combined SGD policy, rebuilds a
  fresh provider/world from the saved geometry, restores the material state,
  creates a fresh bounded artifact store, and cold-compiles the saved manifest.
- Restore fails unless dataset, target residency, camera grid, program factory,
  provider, world, site content, initializer, geometry generation, manifest,
  and cold-recompile seal reproduce the checkpoint exactly.
- The returned `PaperKineticFixedCameraRestoredReadyGeneration` contains no
  checkpoint tensors. Its tensor-free receipt binds source generations and
  accounts for source payload + parsed checkpoint + live state + bounded store.
- `claim_paper_kinetic_fixed_camera_restored_ready_generation_for_next_step(...)`
  requires an explicit zero-retained-byte caller attestation before producing
  the next full-geometry coordinator.
- Added a selected-track manifest constructor. It accepts canonical per-view
  pixel ids, merges only adjacent selected pixels, splits at the bounded request
  size, and caps explicit track ids and interval descriptors. It never expands
  an implicit full `H*W` view.

## Important identity result

The restored checkpoint generation is exact: provider, world, state, material,
manifest, and cold-recompile digests match the uninterrupted generation.
After resuming, fresh-process replay/authorization digests intentionally differ
because they bind fresh runtime identities. The next-step loss, all material and
geometry gradients, and all updated parameter tensors match exactly in the CPU
fake-native behavior test. Do not require post-restart capability identities to
equal the uninterrupted process.

## Compatibility metadata

The legacy checkpoint payload fields
`combined_checkpoint_restore_integrated=false` and
`production_trainer_integrated=false` remain required by schema compatibility.
They are not current capability discovery. The lazy combined update receipt now
reports the callable restore API as current source capability; ablation code
must not treat the legacy payload booleans as authoritative.

## Verification

- Combined checkpoint/state suite: `9 passed`.
- Broader adjacent CPU suite: `32 passed`, with one superseded expectation in
  `test_paper_kinetic_fixed_site_material_step.py`: the new direct-selected-
  pixel target path correctly reports zero full decoded-frame bytes while the
  old test still expects a positive value.
- Python compilation passed for the changed source/tests.


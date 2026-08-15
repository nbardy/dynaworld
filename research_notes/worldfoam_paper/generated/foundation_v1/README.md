# WorldFoam Paper-B foundation artifact bundle

This is a deterministic, fail-closed evidence bundle. It is not itself a submission package.

- Accepted inputs: m0_m5_segment_parity, m3_m5_partial_chord_fit, adaptive_m3_m5_basis_selection, constant_density_ordered_transfer, synthetic_visibility_g0_g3
- Rejected inputs: compiled_lie_frame_density
- Missing inputs: g4_public_quality, g6_native_memory
- Native memory fit: false
- Public quality evidence: false
- Adaptive M3/M5 CPU synthetic basis selection: accepted
- G0/G3 synthetic CPU visibility: accepted (S1-S8/C1-C7 only)
- Public/native visibility advantage: false
- Evidence ready for ICLR packaging: false

G4/G6 placeholders are emitted only for missing or rejected gates. Accepted gates are replaced by independently rebuilt numeric assets.

Gate ledger digest: `642a539968236a64d8e314cf82ca4ccd93a24e75f878b5b9e611737230322bec`

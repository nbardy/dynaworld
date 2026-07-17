# World Foam Boundary Delta Tape Probe

After the owner-delta probe showed a strong owner-edit signal but no exact
`length`/`mid` representation, we added a boundary-order delta probe:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/probe_segment_boundary_delta_tape.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_segment_boundary_delta_tape_probe_render32_2_4_8_16.json
```

Why this matters: if we store boundary ids in depth order, the existing Gate 4
rational boundary-depth coefficients can recover the segment cuts, so
`length`/`mid` do not have to be stored per frame. That makes boundary-order
deltas a closer exact-replay candidate than owner-only deltas.

Result:

- status: `informational`
- full boundary count scale 2f->16f: `8.0574x`
- boundary edit-op scale 2f->16f: `6.2125x`
- delta replacement boundary storage scale 2f->16f: `8.2260x`
- 16f replacement boundary-order storage: `0.3462x` full segment CSR
- 16f boundary edit-op stream estimate: `0.3389x` full segment CSR
- 16f boundary edit ops per transition: `12.65`
- 16f boundary rows are almost all unique: `0.9991x` samples per-track and
  `0.9983x` samples globally

Interpretation: boundary-order deltas are useful evidence that exact geometry
can be recovered from a smaller-than-full tape, but raw all-boundary order is
too noisy to be the final STAR-like representation. The likely next design is a
hybrid: use owner edit deltas for topology, use boundary-depth coefficients only
for the local edited intervals, and zero-prune the VJP rows like STAR UVT.

The canonical status summary now includes this probe and still verifies green
with `failures: []`.

# World Foam Delta Tape And STAR Port Map

We used two sidecar checks after the compact segment-tape shader:

1. STAR UVT structural audit.
2. A new Python-only World Foam delta-tape probe.

The STAR audit says the clean part of STAR is not just "fused shaders"; it is
that the primitive and tile row both span time. A tube is evaluated in UVT
space, binned into 3D UVT tile rows, and backward uses tile-pair/suffix/zero-
pruned gradient emission instead of per-frame primitive rows.

Port target for World Foam:

- use slab-level rows like `(track or track tile, time slab, topology/owner
  slot)`, not `(track, frame, segment)`
- keep rational boundary-depth coefficients from Gate 4 for per-frame numeric
  evaluation inside the row
- port STAR's suffix compositing idea for owner/site RGBA gradients
- zero-prune gradient rows with invalid ids when a slab/topology slot has no
  useful contribution

The new probe is:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/probe_segment_delta_tape.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_segment_delta_tape_probe_render32_2_4_8_16.json
```

Key result:

- status: `informational`
- frame scale 2f->16f: `8.0x`
- full segment scale: `8.0559x`
- full compact CSR storage scale: `8.0554x`
- coarse changed-row event scale: `13.5737x` (rejected)
- owner edit-op scale: `1.2969x` (promising)
- owner-delta storage scale: `7.5201x`
- 16f owner-delta storage: `0.3245x` of full compact CSR
- 16f edit-op stream owner-only estimate: `0.0948x` of full compact CSR
- 16f geometry-row delta estimate: `0.9292x` of full compact CSR

Interpretation: owner topology has a sublinear edit signal, but it is not yet
an exact renderer representation. Unchanged owner topology still has changing
segment `length`/`mid` values; those need a coefficient model or compact stream
before this can replace the current per-sample segment tape.

The canonical status summary now includes this probe and still verifies green:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_fused_slab_status_summary.py \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_status_summary_verifier.json
```

Verifier status: `ok`, `failures: []`.

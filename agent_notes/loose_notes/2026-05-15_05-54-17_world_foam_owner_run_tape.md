# World Foam Same-Owner Run Tape Probe

We tested a practical compression path that reuses the existing compact
segment-tape Metal kernels: merge adjacent segment-tape rows with the same
winning owner.

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/probe_segment_owner_run_tape.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --timing-iters 3 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_segment_owner_run_tape_probe_render32_2_4_8_16.json
```

Key 16f result:

- full segments: `1272086`
- owner-run segments: `129395`
- segment ratio: `0.1017`
- storage ratio: `0.1094`
- max forward RGB error vs full tape: `4.17e-7`
- max forward alpha error vs full tape: `5.36e-7`
- max forward depth error vs full tape at current density: `4.77e-7`
- RGB-only VJP relative error: `6.95e-6`
- full RGB-only VJP timing: `16.48 ms`
- owner-run RGB-only VJP timing: `1.51 ms`

Interpretation: this is the first result that looks like a practical speed
path rather than only a structural diagnostic. It is exact for RGB/alpha under
same-owner merging, and the RGB-only VJP matches while using the existing Metal
kernel input format.

Scope boundary: the compressed depth midpoint is a current-density effective
midpoint, and threshold truncation is current-density dependent. This makes the
probe a strong RGB-training candidate, not a final density-independent
geometry/depth-gradient tape. A full RGBA/depth-gradient replacement still needs
either a density-independent moment representation or a different depth contract.

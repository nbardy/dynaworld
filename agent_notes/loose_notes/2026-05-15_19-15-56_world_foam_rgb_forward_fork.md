# 2026-05-15 WorldFoam block-coeff RGB forward fork

I forked the f32 block-coeff WorldFoam replay path into an RGB-only forward
shader and wired it through the research harness as:

```text
endpoint-record-edit-block-coeff-rgb
```

The intent was to test whether STAR-UVT-style cleanup could remove avoidable
RGBA/depth work from WorldFoam while preserving the existing block-coefficient
math and direct RGB VJP path.

Implementation touched the `world_foam_lane2_fused_slab_v0` Metal extension:

- added `wf2_endpoint_record_edit_block_coeff_rgb_replay_tensor`
- exposed `endpoint_record_edit_block_coeff_rgb_replay`
- added a Python autograd wrapper that reuses the existing
  `endpoint_record_edit_block_coeff_vjp_direct_atomic_rgb_only` backward
- added the train/eval tape mode and compare-harness inclusion flag
- extended probe/unit coverage so RGB-only replay is checked against the f32
  block-coeff RGB output and endpoint-run RGB

Build and focused verification passed:

```bash
( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  && PYTHONDONTWRITEBYTECODE=1 uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/probe_endpoint_record_edit_replay.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s research_experiments/world_foam_lane2 \
  -p 'test_probe_endpoint_record_edit_replay.py' -q

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s research_experiments/world_foam_lane2 \
  -p 'test_*.py'

git diff --check
git -C third_party/fast-mac-gsplat diff --check
```

The raw Metal probe artifact is:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_replay_blockcoeff_rgbforward_probe_render16_16f.json
```

It now reports `status=ok` with `scaling_gate_applicable=false` for the
single-frame-count probe. Correctness is good:

- `metal_block_coeff_rgb_forward_matches=true`
- max RGB-only forward absolute error versus endpoint-run RGB:
  `5.960464477539062e-7`

But the speed result was negative on the rerun:

- endpoint forward: `7.64 ms`
- f32 block-coeff forward: `14.43 ms`
- RGB-only block-coeff forward: `30.51 ms`

The earlier one-step comparison had made RGB-only look promising, but the
warmed five-step train/eval and this rerun both say not to trust that first
read. The stable interpretation is that RGB-only replay is correct, but this
fork does not yet make WorldFoam competitive.

The warmed 16f/32f comparison artifact is:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_vs_record_edit_blockcoeff_rgbforward_repeat_loaded_warm2_steps5_render16_16_32.json
```

Key totals:

- endpoint-run: `4.42 ms -> 4.88 ms`
- f32 block-coeff: `5.92 ms -> 6.22 ms`
- RGB-only block-coeff: `28.26 ms -> 34.10 ms`

The f32 block-coeff path has the desired sublinear shape across repeated 16f to
32f synthetic scaling, but is still slower than endpoint-run at this tiny
render size. The RGB-only fork is both slower and unstable in the train loop.

Current answer to the theory/practice question:

- In theory, WorldFoam's edit/block-coeff representation can be sublinear in
  frame count because repeated owner/cut structure and per-track coefficients
  are reused instead of rasterizing independent frame records.
- In practice, the current WorldFoam Metal implementation has only partial
  sublinear evidence: f32 block-coeff scales better than frame count in the
  synthetic repeated-frame harness, but absolute wall time and backward/VJP
  scheduling are not STAR-UVT competitive.
- STAR UVT is cleaner because the hot loops are organized around a compact
  tile/time/pair contract with less replay-side owner-edit indirection. The
  WorldFoam fork still carries more metadata motion, cut replay, and separate
  VJP cost.

I also re-checked the local STAR UVT notes/artifacts before answering the
user-facing comparison. STAR UVT has real sparse/sublinear rasterizer evidence,
but the docs are careful about the same boundary: it is not the same as saying
the full trainer is solved. The fixed-tube STAR scale notes keep 7168 tubes and
16 frames while target pixels grow 64x; active tile-tube pairs stay near-flat
at `451838`, `516990`, `531638`, and `539762`. The backward scale notes keep
compact rows roughly flat at `207324`, `211374`, and `213642`, but explicitly
do not claim full STAR training is already faster than the strongest baseline.

So the compact answer is:

- STAR UVT: yes for the rasterizer/compact row count; full training still has
  caveats.
- WorldFoam: yes only as a representation and partial f32 block-coeff scaling
  shape; current end-to-end train timings are not practically competitive.

Next useful fork is probably not another RGB/half storage tweak. The evidence
points at the need to collapse WorldFoam's forward and RGB VJP around the same
compact block/tile work contract, or to port more of STAR UVT's tile-pair
execution shape directly instead of only porting coefficient math.

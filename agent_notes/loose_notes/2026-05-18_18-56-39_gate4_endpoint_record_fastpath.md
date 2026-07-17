# Gate4 Endpoint-Record Fast Path

## Context

We stopped after the owner-run reverse-tape keeper and the owner-update negative
to test the more STAR-like endpoint-record idea on the real moving first-person
WorldFoam setup. The existing endpoint-record shader path had looked excellent
through 8 frames, but the old host packer stalled on the 16-frame 64px/24-site
case because it still built endpoint rows through slow per-frame all-boundary
work.

## Changes

- Added Gate4 affine-tape to endpoint-record helpers in
  `research_experiments/world_foam_lane2/gate4_affine_slab_tape.py`:
  - `build_gate4_endpoint_run_sequences(...)`
  - `build_gate4_boundary_depth_coefficients(...)`
  - compact endpoint cut sentinels and `Gate4EndpointRunRecord`
- Added a focused unit test in
  `research_experiments/world_foam_lane2/test_gate4_moving_ray_slab_compiler.py`
  that compares Gate4 endpoint rows against the old slow owner-run sequence
  builder on a moving ray toy scene.
- Added `--endpoint-record-source gate4-affine` to
  `research_experiments/world_foam_lane2/train_eval_owner_run_tape.py`.
  For delta-replace coeff16 fused-MSE endpoint-record modes this source skips
  the full segment-tape baseline construction and feeds the packed
  framegroup16 shader from Gate4-built endpoint rows and affine depth
  coefficients.
- Replaced per-segment full `owner_at_4d` scans in the Gate4 endpoint builder
  with a boundary-pair local owner update: compute the first segment owner once,
  then update across each sorted bisector cut by comparing the current owner
  with the boundary's left/right sites.

## Commands

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/gate4_affine_slab_tape.py \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  research_experiments/world_foam_lane2/test_gate4_moving_ray_slab_compiler.py

PYTHONPATH=research_experiments/world_foam_lane2:src/train \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler -v

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --frame-counts 2,4,8,16 \
  --render-size 64 \
  --site-count 24 \
  --steps 20 \
  --warmup-steps 5 \
  --optimizer-mode manual-vjp \
  --tape-mode endpoint-record-delta-replace-coeff16-auto-framegroup16-fused-mse \
  --endpoint-record-source gate4-affine \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_pairupdate_repeat20_render64_site24_2_4_8_16.json \
  --partial-out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_pairupdate_repeat20_render64_site24_2_4_8_16.partial.json

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_framegroup16_timing_robust.py \
  research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_pairupdate_repeat20_render64_site24_2_4_8_16.json \
  --expected-frames 2,4,8,16 \
  --expected-tape-mode endpoint-record-delta-replace-coeff16-auto-framegroup16-fused-mse \
  --max-total-scale 2.0 \
  --max-backward-scale 2.0 \
  --max-storage-scale 1.10 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_pairupdate_repeat20_render64_site24_2_4_8_16.verify.json
```

## Results

The robust verifier passed:

- total median ms by frame: `2.871 / 3.424 / 2.303 / 2.985`
- backward median ms by frame: `2.485 / 2.990 / 1.990 / 2.665`
- total median scale `2f -> 16f`: `1.040x` for `8x` frames
- backward median scale `2f -> 16f`: `1.072x` for `8x` frames
- selected storage scale `2f -> 16f`: `1.040x`
- heldout PSNR: `14.170 / 13.993 / 14.221 / 14.232`

Mean step timings from the artifact:

- total mean ms: `3.247 / 3.498 / 2.406 / 3.039`
- backward mean ms: `2.861 / 3.115 / 2.091 / 2.712`

This is the first real 64px/24-site WorldFoam endpoint-record result that is
STAR-shaped in the warm training step through 16 real frames.

## Caveats

The full pipeline is not fixed yet. The fast path intentionally skipped
baseline full/owner-run segment tapes, so fields named `*_vs_full` in this
artifact are placeholders for this source and should not be cited as full-tape
compression proof. The useful compression number here is the selected storage
scale across frame count.

Setup is still bad and remains the next bottleneck:

- train endpoint sequence build seconds: `9.96 / 18.11 / 33.13 / 63.17`
- heldout endpoint sequence build seconds: `3.76 / 6.92 / 12.13 / 22.61`
- train candidate replay iterations: `2.68M / 5.32M / 10.63M / 21.26M`

So the Metal fused-MSE shader is now competitive and sublinear in warm-step
practice, but host-side endpoint-row materialization is still roughly linear
and too slow at 16f. The next fork should move endpoint-row construction out of
Python: either build owner-run endpoint rows directly in the Gate4 compiler, or
use a vectorized/C++/Metal prepass over the affine candidate rows.

## Interpretation

STAR UVT looked cleaner because its time-tube representation is already the
training-time compact representation. WorldFoam can reach the same warm-kernel
shape once it is represented as endpoint records, but it currently pays an
expensive Python compilation step to discover those records from moving
power-cell cuts. The math port is working; the remaining work is making the
record compiler as compact and native as the shader.

## Follow-up: vectorized depth plus owner-changing scan

After the first passing artifact, the setup path was still clearly the wrong
side of the result. I added two host compiler optimizations in
`gate4_affine_slab_tape.py`:

- vectorized affine boundary-depth evaluation across frames with NumPy
- record construction by scanning only owner-changing boundary cuts after the
  first owner lookup, instead of walking every candidate segment as a possible
  output row

The broader endpoint-record tests still pass, including a multitrack/time-slab
comparison against the slow owner-run sequence builder:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/test_gate4_moving_ray_slab_compiler.py \
  research_experiments/world_foam_lane2/gate4_affine_slab_tape.py

PYTHONPATH=research_experiments/world_foam_lane2:src/train \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler -v
```

Final repeat20 artifact and verifier:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_vecdepth_ownerscan_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_vecdepth_ownerscan_repeat20_render64_site24_2_4_8_16.verify.json
```

The robust verifier passed:

- total median ms by frame: `2.062 / 2.058 / 2.405 / 2.871`
- backward median ms by frame: `1.747 / 1.779 / 2.082 / 2.563`
- total median scale `2f -> 16f`: `1.392x` for `8x` frames
- backward median scale `2f -> 16f`: `1.467x` for `8x` frames
- selected storage scale `2f -> 16f`: `1.040x`
- heldout PSNR: `14.170 / 13.993 / 14.221 / 14.232`

Setup improved materially, but did not disappear:

- train endpoint sequence build seconds:
  `4.55 / 6.38 / 10.11 / 17.61`
- heldout endpoint sequence build seconds:
  `2.00 / 2.76 / 4.39 / 7.63`
- 16f setup improvement versus the first Gate4 endpoint-record artifact:
  train `63.17s -> 17.61s` (`3.6x` faster), heldout `22.61s -> 7.63s`
  (`3.0x` faster)

Reflection: this is a good stop point. The warm Metal endpoint-record path is
sublinear in practice and passes the scale verifier, but the setup/compiler
path is still doing too much per-frame host work. The right next move is not
another fused-MSE shader variant; it is a native/vectorized endpoint-row
compiler that removes Python from the owner/cut-row materialization loop.

## Follow-up: vectorized initial owners and batched frame sorts

I took one more host-compiler pass before moving on to native work:

- removed the last per-frame Python `owner_at_4d` call from
  `build_gate4_endpoint_run_sequences(...)`; each track now computes initial
  owners for its frames with a small vectorized site-distance batch
- for single-slab Gate4 tapes, sorted all candidate depths for a track's frames
  in one `np.argsort(..., axis=0)` call, then sliced each frame's valid prefix
- removed stale helper code that was only supporting the old Python owner path

The same focused compiler tests still pass:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/test_gate4_moving_ray_slab_compiler.py \
  research_experiments/world_foam_lane2/gate4_affine_slab_tape.py

PYTHONPATH=research_experiments/world_foam_lane2:src/train \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler -v
```

Final artifact and verifier from this pass:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_vecowner_batchsort_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_vecowner_batchsort_repeat20_render64_site24_2_4_8_16.verify.json
```

The robust verifier passed:

- total median ms by frame: `2.225 / 2.148 / 2.422 / 2.929`
- backward median ms by frame: `1.889 / 1.835 / 2.099 / 2.605`
- total median scale `2f -> 16f`: `1.316x`
- backward median scale `2f -> 16f`: `1.379x`
- selected storage scale `2f -> 16f`: `1.040x`
- heldout PSNR: `14.170 / 13.993 / 14.221 / 14.232`

Setup moved a little further:

- train endpoint sequence build seconds:
  `4.54 / 6.24 / 9.95 / 16.46`
- heldout endpoint sequence build seconds:
  `1.95 / 2.71 / 4.20 / 7.43`
- 16f setup improvement versus the first Gate4 endpoint-record artifact:
  train `63.17s -> 16.46s` (`3.8x` faster), heldout `22.61s -> 7.43s`
  (`3.0x` faster)

This is a small but real improvement, not a qualitative fix. The setup path is
still host-side Python over track/frame/cut rows. The next meaningful fork is a
native endpoint-row compiler or a row/owner-run kernel that consumes a compact
record representation directly.

## Follow-up: chunked batched candidate sorting

I tried pushing the single-slab Gate4 endpoint builder further by batching
candidate depth sort work across tracks instead of sorting one track at a time.
The implementation keeps the exact endpoint-record semantics:

- single-slab tapes now process tracks in fixed-size chunks
- each chunk pads row-local candidate coefficients, evaluates depth for all
  frames, and sorts along the candidate axis with one NumPy batch call
- initial owner selection is still batched for all nonempty track/frame rows in
  the chunk
- multi-slab tapes keep the previous exact fallback path

The exactness gate still passes:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/test_gate4_moving_ray_slab_compiler.py \
  research_experiments/world_foam_lane2/gate4_affine_slab_tape.py

PYTHONPATH=research_experiments/world_foam_lane2:src/train \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler -v
```

I tried chunk sizes `1024`, `256`, and `128`. The 128-track chunk is the current
default because its 16f spot check was best, but the full-matrix result is still
only marginal. Final full artifact and verifier:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_chunkbatch128_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_chunkbatch128_repeat20_render64_site24_2_4_8_16.verify.json
```

The robust verifier passed:

- total median ms by frame: `2.070 / 2.190 / 2.442 / 2.928`
- backward median ms by frame: `1.758 / 1.868 / 2.106 / 2.607`
- total median scale `2f -> 16f`: `1.414x`
- backward median scale `2f -> 16f`: `1.483x`
- selected storage scale `2f -> 16f`: `1.040x`
- heldout PSNR: `14.170 / 13.993 / 14.221 / 14.232`

Setup comparison versus the first Gate4 endpoint-record artifact:

- train endpoint sequence build seconds:
  `9.96 / 18.11 / 33.13 / 63.17` -> `4.46 / 6.25 / 9.63 / 16.49`
- heldout endpoint sequence build seconds:
  `3.76 / 6.92 / 12.13 / 22.61` -> `1.90 / 2.66 / 4.13 / 7.31`

Compared with the previous `vecowner_batchsort` full artifact, this is mixed:
`16f` train setup is effectively tied/slightly worse (`16.46s -> 16.49s`), but
heldout setup improves (`7.43s -> 7.31s`) and `8f` train setup improves
(`9.95s -> 9.63s`). Treat it as a marginal vectorization pass, not a solved
compiler. The native endpoint-row compiler remains the real next move.

## Follow-up: skip full per-sample validation in the benchmark fast path

The setup profile showed that the remaining bottleneck was no longer endpoint
row playback or row materialization, but the debug validation inside
`build_gate4_affine_slab_tape`: every sample re-ran `event_set_for_ray`, which
in turn recomputed every 4D boundary crossing. I split that into an explicit
`sample_validation` mode:

- `sample_validation="full"` remains the default and keeps
  `missing_sample_events` authoritative.
- `sample_validation="skip"` preserves the same compiled candidate rows and
  endpoint records, estimates replay iterations from row/slab/frame counts, and
  marks metadata as non-authoritative for missing-event validation.
- `train_eval_owner_run_tape.py --endpoint-record-source gate4-affine` now uses
  the skipped mode because the focused unit test covers exactness against the
  slow owner-run reference.

Focused exactness gate:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/test_gate4_moving_ray_slab_compiler.py \
  research_experiments/world_foam_lane2/gate4_affine_slab_tape.py \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py

PYTHONPATH=research_experiments/world_foam_lane2:src/train \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler -v
```

Result: `Ran 6 tests ... OK`. The new test builds both full-validation and
skipped-validation tapes, asserts identical CSR candidate rows/coefficients,
and asserts the endpoint records match before comparing against the slow
owner-run rows.

Full artifact and verifier:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_skipvalidate_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_skipvalidate_repeat20_render64_site24_2_4_8_16.verify.json
```

The robust verifier passed:

- total median ms by frame: `2.005 / 2.188 / 2.195 / 3.789`
- backward median ms by frame: `1.702 / 1.875 / 1.916 / 3.348`
- total median scale `2f -> 16f`: `1.890x`
- backward median scale `2f -> 16f`: `1.966x`
- selected storage scale `2f -> 16f`: `1.040x`
- heldout PSNR: `14.170 / 13.993 / 14.221 / 14.232`

Setup comparison versus the previous chunkbatch128 artifact:

- train endpoint sequence build seconds:
  `4.46 / 6.25 / 9.63 / 16.49` -> `3.18 / 3.89 / 4.61 / 6.59`
- heldout endpoint sequence build seconds:
  `1.90 / 2.66 / 4.13 / 7.31` -> `1.32 / 1.54 / 1.96 / 2.64`

This is the first setup pass that materially changes the 16f compiler cost:
train setup is `2.5x` faster than chunkbatch128 and `9.6x` faster than the
first Gate4 endpoint-record artifact (`63.17s -> 6.59s`). Heldout setup is
`2.8x` faster than chunkbatch128 and `8.6x` faster than the first artifact
(`22.61s -> 2.64s`).

Caveat: the warm kernel still passed the formal sublinear gate, but it is now
near the threshold (`1.89x` total median, `1.97x` backward median for `8x`
frames) and slower than the previous chunkbatch128 warm timings. Candidate
counts and replay iterations are unchanged, so this looks like MPS timing
variance or a run-order effect rather than a semantic change. Treat this as a
major setup win, not a new warm-runtime winner.

## Follow-up: vectorized track-boundary coefficients

The next profile after validation skipping showed the remaining setup split
between Gate4 candidate compilation, endpoint-record materialization, and a
separate all-track boundary coefficient build. I removed the repeated scalar
coefficient path by computing the `[track, boundary, 4]` affine depth
coefficient table once with NumPy matrix operations and reusing it for:

- slab event selection,
- slab-mid-depth candidate ordering,
- row-local candidate coefficient materialization, and
- the shader-facing all-boundary coefficient tensor.

The exactness gate still passes:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/test_gate4_moving_ray_slab_compiler.py \
  research_experiments/world_foam_lane2/gate4_affine_slab_tape.py \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py

PYTHONPATH=research_experiments/world_foam_lane2:src/train \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler -v
```

Result: `Ran 6 tests ... OK`.

Full artifact and verifier:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_veccoeff_skipvalidate_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_veccoeff_skipvalidate_repeat20_render64_site24_2_4_8_16.verify.json
```

The robust verifier passed:

- total median ms by frame: `2.133 / 2.151 / 2.354 / 3.019`
- backward median ms by frame: `1.825 / 1.837 / 2.041 / 2.674`
- total median scale `2f -> 16f`: `1.415x`
- backward median scale `2f -> 16f`: `1.465x`
- selected storage scale `2f -> 16f`: `1.040x`
- heldout PSNR: `14.170 / 13.993 / 14.221 / 14.232`

Setup comparison versus the validation-skipped artifact:

- train endpoint sequence build seconds:
  `3.18 / 3.89 / 4.61 / 6.59` -> `2.14 / 2.70 / 3.55 / 5.52`
- heldout endpoint sequence build seconds:
  `1.32 / 1.54 / 1.96 / 2.64` -> `0.84 / 1.03 / 1.57 / 2.10`
- train coefficient build seconds:
  about `0.90s` at small frame counts and `0.93s` at 16f -> `0.02-0.03s`

This is a real compiler/setup improvement and also restores the warm-runtime
gate margin relative to the validation-skipped run. It still is not native row
construction: endpoint records are still materialized as Python record lists
before packing. The current next fork is to emit the delta-replace arrays
directly from the Gate4 candidate pass, skipping the intermediate
`Gate4EndpointRunRecord` sequence objects and the unused edit tape for packed
delta modes.

## Follow-up: direct Gate4 delta-replace emission

I added a direct delta-replace tape builder for the Gate4 endpoint-record path.
It reuses the same candidate depth replay and batched first-owner selection, but
appends the base row and per-frame changed rows directly into the packed
delta-replace arrays. The train harness uses this direct builder for
`--endpoint-record-source gate4-affine` when the resolved mode is a delta mode,
so both `pack_endpoint_record_edit_s` and `pack_endpoint_record_delta_replace_s`
are now zero in that path.

The exactness test now compares direct Gate4 delta tensors against the old
`build_gate4_endpoint_run_sequences(...)` plus
`pack_endpoint_record_delta_replace_tape(...)` path. Focused gate:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/test_gate4_moving_ray_slab_compiler.py \
  research_experiments/world_foam_lane2/gate4_affine_slab_tape.py \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py

PYTHONPATH=research_experiments/world_foam_lane2:src/train \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler -v
```

Result: `Ran 6 tests ... OK`.

Full artifact and verifier:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_directdelta_veccoeff_skipvalidate_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_directdelta_veccoeff_skipvalidate_repeat20_render64_site24_2_4_8_16.verify.json
```

The robust verifier passed:

- total median ms by frame: `2.303 / 3.594 / 2.255 / 2.967`
- backward median ms by frame: `1.955 / 3.269 / 1.971 / 2.625`
- total median scale `2f -> 16f`: `1.288x`
- backward median scale `2f -> 16f`: `1.342x`
- selected storage scale `2f -> 16f`: `1.040x`
- heldout PSNR: `14.170 / 13.993 / 14.221 / 14.232`

Setup comparison versus the vectorized-coefficient sequence-packer artifact:

- train endpoint build/direct-delta seconds:
  `2.14 / 2.70 / 3.55 / 5.52` -> `2.09 / 2.65 / 3.56 / 5.17`
- heldout endpoint build/direct-delta seconds:
  `0.84 / 1.03 / 1.57 / 2.10` -> `0.85 / 1.07 / 1.49 / 2.03`
- train edit pack seconds:
  `0.073 / 0.145 / 0.235 / 0.311` -> all `0`
- train delta pack seconds:
  `0.020 / 0.036 / 0.064 / 0.106` -> all `0`

This is a modest setup win rather than a new math result. It removes the
intermediate Python sequence objects from the selected packed-delta path and
keeps the warm-kernel verifier comfortably sublinear. Remaining setup cost is
inside the direct row replay itself: cut-array construction, owner-run row
walking, and first-owner selection. A native row compiler is still the real
endpoint setup fix.

## Follow-up: preallocated cut arrays and topology-reuse negative

I tried one small allocation cleanup in `_cut_arrays_from_ordered_depth_ids`:
preallocate the `near + kept_depths + far` arrays instead of using
`np.concatenate`. The focused exactness gate still passed:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/test_gate4_moving_ray_slab_compiler.py \
  research_experiments/world_foam_lane2/gate4_affine_slab_tape.py \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py

PYTHONPATH=research_experiments/world_foam_lane2:src/train \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler -v
```

Result: `Ran 6 tests ... OK`.

Full artifact and verifier:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_prealloc_directdelta_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_prealloc_directdelta_repeat20_render64_site24_2_4_8_16.verify.json
```

The robust verifier passed:

- total median ms by frame: `2.315 / 2.579 / 2.366 / 2.952`
- backward median ms by frame: `1.982 / 2.270 / 2.048 / 2.624`
- total median scale `2f -> 16f`: `1.275x`
- backward median scale `2f -> 16f`: `1.324x`
- selected storage scale `2f -> 16f`: `1.040x`
- heldout PSNR: `14.170 / 13.993 / 14.221 / 14.232`

Setup comparison versus the direct-delta artifact:

- train endpoint direct-delta build seconds:
  `2.09 / 2.65 / 3.56 / 5.17` -> `2.14 / 2.65 / 3.42 / 5.45`
- heldout endpoint direct-delta build seconds:
  `0.85 / 1.07 / 1.49 / 2.03` -> `0.85 / 1.11 / 1.33 / 1.95`

Practical read: prealloc is safe and verified, but it is neutral rather than a
clear setup win. It mostly changes allocation shape; the real cost is still the
Python owner-run replay.

I also tried a conservative topology-reuse fork inside direct delta emission:
if adjacent frames had the same cut-id order, start segment, and first owner,
reuse the previous owner-run records when all segments were strictly nonempty.
Exactness passed, but the 16f timing spot was negative:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_toporeuse_directdelta_spot_render64_site24_16.json
```

- total median: `3.814 ms`
- backward median: `3.459 ms`
- train endpoint build: `5.40 s`
- heldout endpoint build: `2.21 s`
- heldout PSNR: `14.232`

That path was reverted. The lesson is that Python-side topology checks add
overhead without removing enough row replay on this moving-camera gate. The
next serious fork should move row construction native or change the tape format
so the Python loop disappears, not add more Python cache conditions.

## Follow-up: owner-membership table keeper

A profile of the current keeper still put the non-I/O setup cost in direct row
replay:

```text
build_gate4_endpoint_delta_replace_tape: 6.991s cum
_append_delta_track_rows: 4.143s cum
_owner_run_records_from_cut_arrays: 3.927s cum
np.flatnonzero: 1.535s cum across 967891 calls
_cut_arrays_from_ordered_depth_ids: 1.187s cum
build_gate4_affine_slab_tape: 3.784s cum
```

I added a small owner-membership table:
`boundary_other_by_owner[owner, boundary_id] -> other_owner_or_-1`. The hot
owner-run walk now resolves "does this boundary touch the current owner?" with
one table lookup and one `>= 0` check, instead of gathering left/right arrays
and doing two equality comparisons for every search.

Focused exactness gate:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/test_gate4_moving_ray_slab_compiler.py \
  research_experiments/world_foam_lane2/gate4_affine_slab_tape.py \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py

PYTHONPATH=research_experiments/world_foam_lane2:src/train \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler -v
```

Result: `Ran 6 tests ... OK`.

Full artifact and verifier:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_ownerother_directdelta_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_ownerother_directdelta_repeat20_render64_site24_2_4_8_16.verify.json
```

The robust verifier passed:

- total median ms by frame: `2.127 / 3.592 / 2.368 / 2.916`
- backward median ms by frame: `1.795 / 3.272 / 2.036 / 2.599`
- total median scale `2f -> 16f`: `1.371x`
- backward median scale `2f -> 16f`: `1.448x`
- selected storage scale `2f -> 16f`: `1.040x`
- heldout PSNR: `14.170 / 13.993 / 14.221 / 14.232`

Setup comparison versus the prealloc artifact:

- train endpoint direct-delta build seconds:
  `2.14 / 2.65 / 3.42 / 5.45` -> `2.09 / 2.53 / 3.23 / 4.85`
- heldout endpoint direct-delta build seconds:
  `0.85 / 1.11 / 1.33 / 1.95` -> `0.83 / 1.05 / 1.32 / 1.90`

Setup comparison versus the earlier direct-delta artifact:

- train 16f endpoint build: `5.17s -> 4.85s`
- heldout 16f endpoint build: `2.03s -> 1.90s`

This is the current keeper for the Python-side endpoint fast path. It is still
not the final STAR-clean construction path: the owner-run search is cheaper,
but row replay remains Python-side and still accounts for most non-I/O setup.

## Follow-up: native chunk row packer

The next fork moved the direct delta-replace row walk into the
`world_foam_lane2_fused_slab_v0` C++ extension for the single-time-slab Gate4
path. Python still builds sorted cut arrays and first-owner probe points, but
the per-track/per-frame owner-run replay and base/change delta packing now run
inside `gate4_delta_replace_from_cuts_cpu`.

Build and op-registration check:

```bash
( cd third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  && uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld \
    python setup.py build_ext --inplace )

PYTHONPATH=third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
import torch
import torch_world_foam_lane2_fused_slab
print(hasattr(torch.ops.world_foam_lane2_fused_slab_v0,
              "gate4_delta_replace_from_cuts_cpu"))
PY
```

Result: `True`.

Exactness gates:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/gate4_affine_slab_tape.py

PYTHONPATH=research_experiments/world_foam_lane2:src/train \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler -v

PYTHONPATH=third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:research_experiments/world_foam_lane2:src/train \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler -v
```

Both unittest passes ran `Ran 6 tests ... OK`; the second pass exercises the
native direct-delta path with the extension importable.

Full artifact and verifier:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativechunk_directdelta_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativechunk_directdelta_repeat20_render64_site24_2_4_8_16.verify.json
```

The robust verifier passed:

- total median ms by frame: `2.258 / 2.154 / 2.464 / 2.966`
- backward median ms by frame: `1.935 / 1.833 / 2.144 / 2.640`
- total median scale `2f -> 16f`: `1.314x`
- backward median scale `2f -> 16f`: `1.364x`
- selected storage scale `2f -> 16f`: `1.040x`
- heldout PSNR: `14.170 / 13.993 / 14.221 / 14.232`

Setup comparison versus the owner-membership keeper:

- train endpoint direct-delta build seconds:
  `2.09 / 2.53 / 3.23 / 4.85` -> `1.97 / 2.20 / 2.63 / 3.48`
- heldout endpoint direct-delta build seconds:
  `0.83 / 1.05 / 1.32 / 1.90` -> `0.83 / 0.89 / 1.06 / 1.53`

This is the current Gate4 endpoint-record keeper. It proves the warm
WorldFoam endpoint path is sublinear over real 2/4/8/16-frame loads, and it
shrinks setup enough that the remaining non-I/O profile has moved to cut-array
construction, coeff/ray materialization, and tensor/list merging around the
native op. It is still not as clean as STAR UVT because the representation is
assembled from per-frame cut rows before it becomes a compact endpoint tape.

I also tried keeping native chunk outputs as tensors and merging them once
instead of converting each chunk to Python lists. Exactness still passed in both
fallback and extension-visible unittest modes, but the 16f spot was negative:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativechunk_tensormerge_spot_render64_site24_16.json
```

- total median: `3.118 ms`
- backward median: `2.704 ms`
- train endpoint build: `4.01 s`
- heldout endpoint build: `2.12 s`
- heldout PSNR: `14.232`

That code path was reverted. The likely issue is that many small tensor slices
and cats are not cheaper than the existing list append pattern at this chunk
size. The next setup cut should move cut-row construction or first-owner
selection into the native chunk op, not just change the merge container.

I then tried the narrower "first-owner selection native" fork: add a C++ op
that computed the initial owner from `ray_coeff`, `frame_t`, and site arrays,
then called the existing native row packer. This removed the Python
`owner_points -> _owner_indices_for_points -> scatter initial_owners` path but
kept Python cut-row construction.

Artifacts:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativeowner_directdelta_spot_render64_site24_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativeowner_directdelta_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativeowner_directdelta_repeat20_render64_site24_2_4_8_16.verify.json
```

Correctness/build gates passed while the fork was present:

- rebuilt `world_foam_lane2_fused_slab_v0`
- op registration showed both old and native-owner ops present
- fallback unittest: `Ran 6 tests ... OK`
- extension-imported unittest: `Ran 6 tests ... OK`

The full sweep robust verifier was technically `status=ok`, but it was not a
keeper:

- total median ms: `2.117 / 2.433 / 8.227 / 3.185`
- backward median ms: `1.733 / 2.041 / 6.883 / 2.711`
- total median scale `2f -> 16f`: `1.504x`
- backward median scale `2f -> 16f`: `1.565x`
- 16f endpoint direct-delta build: train `4.22s`, heldout `1.61s`

Compared with the current native-chunk keeper, 16f train setup regressed
`3.48s -> 4.22s`, 16f warm median regressed `2.966ms -> 3.185ms`, and 8f had a
large timing outlier. I reverted the C++ op and Python dispatch, rebuilt the
extension, verified the losing op was absent, and reran both fallback and
extension-imported unittests (`Ran 6 tests ... OK`). The lesson is that native
first-owner selection alone is too small and adds enough crossing/loop overhead
to lose. The next real setup fork should move cut-row assembly itself native or
change the representation so those rows are never materialized in Python.

I tried that next cut as a native sorted-row packer:
`gate4_delta_replace_from_sorted_cpu` consumed the vectorized
`chunk_depths/chunk_ids/valid_counts` arrays directly, treated `near`/`far` as
implicit cuts, and emitted the packed delta arrays. This passed the small
exactness suite with the extension explicitly imported, but the 16f real gate
exposed two problems:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativesorted_directdelta_spot_render64_site24_16.json
```

- 16f setup improved: train `3.15s`, heldout `1.28s`
- warm timing regressed: total `6.179 ms`, backward `4.132 ms`
- endpoint segment counts changed versus the current keeper:
  train selected/full `222276` instead of `222501`

The setup signal is real, but the changed segment counts mean this fork is not
semantically exact on the real moving-camera gate. I reverted the sorted op and
dispatch, rebuilt the extension, verified `gate4_delta_replace_from_sorted_cpu`
was absent, and reran fallback plus extension-imported unittests (`Ran 6 tests
... OK`). A future native cut-row fork needs an exact sorted-vs-cut parity test
on a high-cap moving-camera fixture before timing it.

I then added that guard fixture before stopping: the new
`test_highcap_single_slab_sorted_rows_match_cut_array_delta_records` case builds
a deterministic 24-site / 16-frame moving-camera slab with rows over 200
candidates, compares the direct Gate4 delta tensors against the cut-array
record path, and asserts that a no-dedupe sorted-row reconstruction would
disagree on this fixture. Current gates:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/test_gate4_moving_ray_slab_compiler.py \
  research_experiments/world_foam_lane2/gate4_affine_slab_tape.py

PYTHONPATH=research_experiments/world_foam_lane2:src/train \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler -v

PYTHONPATH=third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:research_experiments/world_foam_lane2:src/train \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
import sys
import unittest
import torch_world_foam_lane2_fused_slab  # noqa: F401
suite = unittest.defaultTestLoader.loadTestsFromName(
    'research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler'
)
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
PY
```

Both unittest modes now run `Ran 7 tests ... OK`. The active extension surface
still has `torch.ops.world_foam_lane2_fused_slab_v0.gate4_delta_replace_from_cuts_cpu`
registered and does not have the reverted native-owner or native-sorted ops.

## Follow-up: exact native sorted-row op, not promoted

I tried a corrected native sorted-row op after adding the high-cap fixture. The
new C++ op, `gate4_delta_replace_from_sorted_cpu`, preserves the Python
cut-array semantics that the first sorted attempt missed:

- dedupe is adjacent-depth dedupe with `1e-6`, matching
  `_cut_arrays_from_ordered_depth_ids(...)`
- near/far sentinels are preserved before the first nonempty segment scan
- first-owner selection uses the same midpoint and power-distance argmin
- invalid rows stay empty through an explicit row-active mask

The op is built into `world_foam_lane2_fused_slab_v0`, but the promoted Python
path is still the native cut-array keeper. The sorted op is behind
`GATE4_ENABLE_EXPERIMENTAL_NATIVE_SORTED_DELTA = False` because timing did not
promote.

Correctness gates:

```bash
( cd third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  && uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld \
    python setup.py build_ext --inplace )

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/gate4_affine_slab_tape.py \
  research_experiments/world_foam_lane2/test_gate4_moving_ray_slab_compiler.py

PYTHONPATH=research_experiments/world_foam_lane2:src/train \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler -v

PYTHONPATH=third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:research_experiments/world_foam_lane2:src/train \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
import sys
import unittest
import torch_world_foam_lane2_fused_slab  # noqa: F401
suite = unittest.defaultTestLoader.loadTestsFromName(
    'research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler'
)
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
PY
```

Both unittest modes run `Ran 7 tests ... OK`; the extension-imported path now
temporarily enables the experimental sorted op inside the high-cap fixture and
compares it against the cut-array delta records.

Real 16f tensor equality probe:

```text
sorted build_endpoint_record_sequences_s: 2.739s
cut-array build_endpoint_record_sequences_s: 3.423s
all 10 delta tensors: exact equal
```

That proves the corrected sorted op fixes the old semantic bug on the real
moving-camera fixture. It also proves the warm regression below is not caused
by different tape sizes or record tensors.

Timing artifacts:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativesorted_exact_spot_render64_site24_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativesorted_exact_spot_rerun_render64_site24_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativechunk_currentbinary_forcedcuts_spot_render64_site24_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativechunk_defaultrestored_spot_render64_site24_16.json
```

Results:

- exact native sorted spot: setup train/heldout `2.84s/1.15s`, but warm median
  total/backward `8.248/7.847 ms`
- exact native sorted rerun: setup `2.98s/1.17s`, warm median
  total/backward `7.840/7.319 ms`
- current-binary forced cut-array path: setup `3.78s/1.61s`, warm median
  total/backward `3.840/3.545 ms`
- restored default cut-array path: setup `3.37s/1.48s`, warm median
  total/backward `2.990/2.621 ms`

Read: exact native sorted is a real setup win and now semantically correct, but
it is not promoted because the timed benchmark repeatedly regressed warm-step
measurement. The active default remains the native cut-array keeper; a future
attempt needs to isolate why an identical delta tensor produced after the
sorted native path perturbs the timed MPS step before using it in a scale run.

I added a same-process MPS isolation probe:

```text
research_experiments/world_foam_lane2/probe_gate4_sorted_delta_mps_timing.py
research_experiments/world_foam_lane2/results/2026-05-18_gate4_sorted_delta_sameprocess_mps_probe_16f.json
```

It prepares the cut-array and experimental sorted deltas in one process, checks
every shared `selected_device` tensor after moving to MPS, then times the
packed framegroup16 fused-MSE VJP as cut -> sorted -> cut. Result:

- all selected-device tensors were equal
- cut prepare: endpoint build `3.421s`, move-to-MPS `0.0756s`
- sorted prepare: endpoint build `2.661s`, move-to-MPS `0.2391s`
- cut-first median VJP: `7.913 ms`
- sorted-second median VJP: `7.545 ms`
- cut-third median VJP: `6.769 ms`

This says the warm slowdown is not a sorted-record semantic issue: when both
tapes are resident, the cut-array keeper is just as slow. I also tried a naive
`torch.mps.synchronize(); torch.mps.empty_cache()` before the measured train
loop:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativechunk_emptycache_spot_render64_site24_16.json
```

That was negative: the default cut-array path slowed to total/backward
`8.010/7.554 ms` median, so explicit cache draining is not a fix. I reverted
that code change. Current tests after the revert:

```text
py_compile: ok
fallback unittest: Ran 7 tests ... OK
extension-imported unittest: Ran 7 tests ... OK
```

Next useful fork: isolate MPS residency/allocation effects without holding both
tapes live. The semantic sorted op is fixed; the promotion blocker is now the
benchmark/device lifetime interaction.

I extended the same probe with clean-process single-tape modes plus lifetime
toggles. These are diagnostics, not promotion gates, because the standalone
VJP timing does not exactly reproduce the full train loop.

Additional artifacts:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_sorted_delta_cleanprocess_cut_probe_16f.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_sorted_delta_cleanprocess_cut_probe_orderfix_16f.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_sorted_delta_cleanprocess_sorted_probe_orderfix_16f.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_sorted_delta_cleanprocess_sorted_probe_gc_16f.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_sorted_delta_cleanprocess_sorted_probe_clone_16f.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_sorted_delta_cleanprocess_sorted_probe_sync_16f.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_sorted_delta_cleanprocess_sorted_probe_sync_warm20_16f.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_sorted_delta_cleanprocess_cut_probe_sync_warm20_16f.json
```

The first cut-only probe accidentally allocated target/site MPS tensors before
tape preparation and was slow (`7.554 ms` median VJP). After fixing the probe to
match the trainer lifetime order (prepare tape first, allocate target/site
second), cut-only dropped to `3.751 ms`. Sorted-only under the same order stayed
slow at `7.834 ms`. Python `gc.collect()` before device allocation (`7.050 ms`)
and cloning all selected-device tensors after transfer (`7.674 ms`) did not fix
it. A plain `torch.mps.synchronize()` before target/site allocation helped
sorted (`4.880 ms`; `4.524 ms` with warmup 20), but the matched cut-sync/warm20
row was also slow (`4.825 ms`), so sync is not a reliable promotion patch.

Read: the clean-process probe is useful for lifetime-order diagnosis, but the
full train/eval artifacts remain the authoritative promotion evidence. Do not
promote exact native sorted from this probe alone.

## Follow-up: explicit sorted CLI scale gate

I wired the corrected sorted op behind an explicit harness flag instead of the
module global:

```text
--experimental-native-sorted-delta
```

The flag propagates through `_prepare_owner_run_tapes(...)`, `_run_one(...)`,
and `run_train_eval(...)`, and the result JSON records
`experimental_native_sorted_delta` at the top level, row level, and Gate4
endpoint metadata. This makes sorted timing auditable without changing the
default native cut-array keeper.

Full scale run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --frame-counts 2,4,8,16 \
  --render-size 64 \
  --site-count 24 \
  --steps 20 \
  --warmup-steps 5 \
  --optimizer-mode manual-vjp \
  --tape-mode endpoint-record-delta-replace-coeff16-auto-framegroup16-fused-mse \
  --endpoint-record-source gate4-affine \
  --experimental-native-sorted-delta \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativesorted_cli_repeat20_render64_site24_2_4_8_16.json \
  --partial-out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativesorted_cli_repeat20_render64_site24_2_4_8_16.partial.json
```

Verifier:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_framegroup16_timing_robust.py \
  research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativesorted_cli_repeat20_render64_site24_2_4_8_16.json \
  --expected-frames 2,4,8,16 \
  --expected-tape-mode endpoint-record-delta-replace-coeff16-auto-framegroup16-fused-mse \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativesorted_cli_repeat20_render64_site24_2_4_8_16.verify.json
```

Result:

- run status: `ok`
- robust verifier: `status=failed`
- heldout PSNR unchanged: `14.170 / 13.993 / 14.221 / 14.232`
- selected storage scale: `1.040x`
- endpoint build train/heldout at 16f: `3.30s / 1.33s`
- total prepare train/heldout at 16f: `3.55s / 1.38s`
- total median ms: `3.507 / 4.371 / 4.826 / 7.447`
- backward median ms: `3.007 / 3.681 / 4.263 / 6.866`
- robust scale failures: total mean/median `2.044x/2.124x`, backward
  mean/median `2.143x/2.284x`

Read: the exact sorted op is useful evidence that the STAR-like math port can
match the cut-array record tensors and save some setup, but the full train/eval
gate says it is not competitive with the native cut-array keeper. Keep it
disabled and diagnostic-only until the warm MPS timing regression is explained.

## Follow-up: native cut-prep fork, not promoted

I tried a narrower variant than the full sorted final packer:

```text
gate4_cut_arrays_from_sorted_cpu
--experimental-native-cut-prep-delta
```

The C++ helper consumes the sorted chunk tensors and emits the exact
`cut_depths`, `cut_ids`, `cut_offsets`, `start_segments`, and `initial_owner`
arrays. Python then calls the already promoted
`gate4_delta_replace_from_cuts_cpu` final packer. The goal was to remove the
remaining Python cut-row assembly while preserving the keeper's final tensor
allocation path.

Correctness:

- extension rebuild succeeded
- fallback unittest: `Ran 7 tests ... OK`
- extension-imported unittest: `Ran 7 tests ... OK`
- high-cap fixture now checks both native cut-prep and native sorted against
  the packed cut-array delta records

16f spot:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativecutprep_spot_render64_site24_16.json
```

- 16f total/backward median: `4.588/3.936 ms`
- endpoint build train/heldout: `2.80s/1.15s`
- heldout PSNR: `14.232`

Full scale:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativecutprep_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativecutprep_repeat20_render64_site24_2_4_8_16.verify.json
```

- run status: `ok`
- robust verifier: `status=failed`
- total median ms: `2.463 / 2.579 / 5.248 / 5.438`
- backward median ms: `2.025 / 2.190 / 4.460 / 4.472`
- 16f endpoint build train/heldout: `3.00s/1.15s`
- selected storage scale: `1.040x`
- heldout PSNR unchanged: `14.170 / 13.993 / 14.221 / 14.232`
- robust failures: total median scale `2.208x`, backward mean/median
  `2.064x/2.208x`

Read: Python cut-row assembly is a real setup cost, and the cut-prep fork
removes part of it, but this is not enough to promote. Like the sorted final
packer, it creates a slower warm-step profile than the native cut-array keeper.
The next useful fork should either avoid materializing endpoint rows entirely
or change the device-side representation; another host-side split of sorted
rows into cuts is unlikely to win.

## Follow-up: existing device-side layout screen

Before writing another Metal kernel, I screened the existing endpoint-record
delta representations on the same 64px/24-site/16f Gate4 path. This was a
sequential same-process ranking pass, not a promotion gate, because the packed
control in this process was slower than the keeper artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_variantspot_packed_render64_site24_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_variantspot_i16x3_fg16_render64_site24_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_variantspot_i16x3_ownerreduce_render64_site24_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_variantspot_i16cols_fg16_render64_site24_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_variantspot_i16x4_fg16_render64_site24_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_variantspot_i16x3_fg64_render64_site24_16.json
```

Results:

- packed framegroup16: `4.097/3.558 ms` total/backward median
- i16x3 framegroup16: `4.513/4.102 ms`
- i16x3 owner-reduce framegroup16: `4.645/4.130 ms`
- i16cols framegroup16: `5.387/4.416 ms`
- i16x4 framegroup16: `8.409/6.838 ms`
- i16x3 framegroup64: `11.022/10.443 ms`
- all heldout PSNRs stayed at `14.232`

Read: the existing device-side layout family does not contain a hidden winner
for the current high-cap Gate4 path. Packed framegroup16 remains the right
device-side representation among the implemented variants. The next real
shader fork needs to avoid endpoint-row materialization or move to a genuinely
different device representation, not another i16 packing shape.

## Follow-up: minimal packed selected-device fork, not promoted

I added `--experimental-minimal-packed-delta-device` to test whether the warm
packed framegroup16 fused-MSE path was losing from unused MPS tensor residency.
In that mode the selected loss device keeps only the tensors the packed kernel
reads (`frame_t`, base/change offsets, packed records, chunk offsets, coeffs,
target/config). The full endpoint-record replay device is built lazily only
for final train/heldout PSNR rendering.

Correctness:

- py_compile passed for the touched train/eval harness and Gate4 test modules.
- fallback unittest: `Ran 7 tests ... OK`
- extension-imported unittest: `Ran 7 tests ... OK`
- 16f spot PSNR stayed unchanged at `14.232`.

Artifacts:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_minimalpacked_spot_render64_site24_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_default_after_minimalpacked_spot_render64_site24_16.json
```

Results:

- minimal selected-device 16f total/backward median: `8.641/8.003 ms`
- same-session default 16f total/backward median: `8.006/7.456 ms`
- minimal selected-device train move/endpoint-build: `0.267s/4.006s`
- same-session default train move/endpoint-build: `0.130s/4.506s`
- both heldout PSNRs: `14.232`

Read: this was a noisy slow MPS session, so it does not replace the clean
nativechunk keeper artifact. But against its same-session control, removing
unused selected-device tensors was not a win. The remaining path should change
the endpoint representation or shader contract; just shrinking tensor
residency around the same packed kernel is not enough.

## Follow-up: materialized i16x3 framegroup16 shader, scale-ok but not keeper

The fork already had a Metal kernel named
`endpoint_record_delta_replace_coeff16_i16x3_framegroup16_materialized_mse_vjp_direct_atomic_rgb_only`,
but the train/eval harness did not expose it. I added the explicit tape mode:

```text
endpoint-record-delta-replace-coeff16-i16x3-framegroup16-materialized-fused-mse
```

This shader materializes each chunk's selected endpoint rows into threadgroup
memory before the per-frame RGB/VJP loop. That changes warm-kernel work more
than the tensor-residency fork, so it deserved a full 2/4/8/16 run.

Artifacts:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_i16x3_materialized_spot_render64_site24_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_i16x3_materialized_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_i16x3_materialized_repeat20_render64_site24_2_4_8_16.verify.json
```

Results:

- robust verifier: `status=ok`
- total median ms: `4.991 / 5.132 / 9.428 / 8.388`
- backward median ms: `4.561 / 4.772 / 8.716 / 7.450`
- total/backward mean scale 2f -> 16f: `1.424x / 1.418x`
- total/backward median scale 2f -> 16f: `1.680x / 1.633x`
- selected storage scale: `1.054x`
- heldout PSNR: `14.170 / 13.993 / 14.221 / 14.232`

Read: this is a valid scale-positive shader diagnostic, but not a promotion.
The clean native cut-array direct-delta keeper is still much faster at 16f
(`2.966/2.640 ms` total/backward median) and scales better on mean
(`1.207x/1.241x`). Materializing row contents into threadgroup memory does not
beat pointer-select plus packed records for this high-cap Gate4 path.

## Follow-up: packed materialized framegroup16 shader, scale-ok but not keeper

I added a narrower materialized shader:

```text
endpoint-record-delta-replace-coeff16-packed-framegroup16-materialized-fused-mse
```

The hypothesis was that the old materialized i16x3 path was dominated by the
three-array i16 layout, not by row materialization itself. This fork keeps the
compact packed int32 record representation, snapshots the selected rows for a
16-frame chunk into one threadgroup int array, and then unpacks from
threadgroup memory in the per-frame RGB/VJP loop. I also fixed the existing
i16x3 materialized harness path to build 16-frame chunk offsets; the previous
2/4/8/16 gate only worked because all tested counts were at or below one
materialized chunk.

Artifacts:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_packed_materialized_spot_render64_site24_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_packed_materialized_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_packed_materialized_repeat20_render64_site24_2_4_8_16.verify.json
```

Checks:

- extension rebuild: ok
- op export: `has_op True`, wrapper export: `has_wrapper True`
- `test_gate4_moving_ray_slab_compiler`: 7 tests ok
- robust verifier: `status=ok`
- 32f repeated-fixture smoke: `status=ok`; this proves the 16-frame
  materialized chunk-offset path launches across more than one chunk, not a
  performance claim

Results:

- full ladder total median ms: `4.518 / 6.777 / 7.503 / 5.757`
- full ladder backward median ms: `4.130 / 5.919 / 7.074 / 5.209`
- scale: total mean/median `0.909x / 1.274x`, backward mean/median
  `1.049x / 1.261x`
- storage scale: `1.040x`
- heldout PSNR unchanged: `14.170 / 13.993 / 14.221 / 14.232`
- standalone 16f spot was cleaner than the full-ladder 16f row:
  `4.219/3.766 ms` total/backward

Read: packed materialization is a real improvement over i16x3 materialization
(`5.757/5.209 ms` vs `8.388/7.450 ms` at 16f in full ladders), but it still
does not beat the native cut-array direct-delta keeper (`2.966/2.640 ms` at
16f). The promotion answer is still no. The useful lesson is sharper:
threadgroup materialization itself costs too much on this path even when the
row format is compact. The next fork should not just repackage selected rows;
it needs to remove per-frame row replay work or isolate the MPS residency
interaction that made sorted/cut-prep slow despite correct tensors.

## Follow-up: high-site threadgroup grad reduction, rejected

I tried raising the framegroup site-gradient reduction cap because the current
64px/24-site gate misses the existing `<=16` reduction path and therefore uses
global grad atomics per segment. A direct raise to 32 sites compiled at the C++
level but failed when Metal created the pipeline for the packed framegroup16
kernel:

```text
Threadgroup memory size (34048) exceeds the maximum threadgroup memory allowed (32768)
```

I narrowed the cap to 24 sites, which launches on the exact 24-site gate, but
the 16f spot was much slower:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_packed_reduce24_spot_render64_site24_16.json
```

- reduce24 16f total/backward median: `8.134/7.710 ms`
- heldout PSNR unchanged: `14.232`

I reverted the cap to the keeper value `16`. A restored-default 16f spot also
ran after the revert:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_default_after_reduce24_restore_spot_render64_site24_16.json
```

That restored spot was quality-correct but noisy/slow (`4.618/4.076 ms`), so it
is not a new keeper artifact. The clean keeper remains the earlier robust
native cut-array ladder. Read: for this path, using more threadgroup memory to
avoid 24-site global grad atomics is worse than leaving the atomics alone.

## Follow-up: smallrun16 replay-cap specialization, rejected

I added a guarded packed framegroup16 fork that compiles the warm VJP kernel
with local replay arrays capped at 16 segments instead of the generic
`WF2_MAX_REALRAY_SEGMENTS=129`. This was motivated by the Gate4 64px/24-site
ladder reporting only 10-12 selected tape segments per sample.

Wired mode and op:

```text
endpoint-record-delta-replace-coeff16-packed-framegroup16-smallrun16-fused-mse
endpoint_record_delta_replace_coeff16_packed_framegroup16_smallrun16_mse_vjp_direct_atomic_rgb_only
```

Validation:

- extension rebuild ok
- wrapper export and torch op export both true
- `test_gate4_moving_ray_slab_compiler`: 7 tests ok
- full ladder verifier intentionally written and run; it failed the promotion
  thresholds

Artifacts:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_packed_smallrun16_spot_render64_site24_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_packed_smallrun16_repeat2_spot_render64_site24_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_packed_smallrun16_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_packed_smallrun16_repeat20_render64_site24_2_4_8_16.verify.json
```

The first 16f spot was bad (`9.198/8.225 ms` total/backward median). A repeat
was much cleaner (`4.270/3.818 ms`), so I ran the full ladder instead of
rejecting on one noisy launch. The full ladder failed robust promotion:

- verifier: `status=failed`
- total median ms: `4.474 / 6.558 / 4.832 / 13.474`
- backward median ms: `3.978 / 4.684 / 3.878 / 12.213`
- total mean/median scale: `3.010x / 3.012x`
- backward mean/median scale: `2.729x / 3.070x`
- storage scale: `1.040x`
- PSNR unchanged: `14.170 / 13.993 / 14.221 / 14.232`

Read: smaller thread-private replay arrays are not the missing win. The fork is
correct enough to keep as an explicit diagnostic mode, but it should not be
auto-selected or promoted. The robust full train/eval gate caught a regression
that a single warmed 16f spot would have over-interpreted.

## Follow-up: min-state recompute shader, rejected at spot screen

I added another packed framegroup16 shader fork that tries to reduce private
replay state without changing endpoint rows or frame chunking:

```text
endpoint-record-delta-replace-coeff16-packed-framegroup16-recompute-fused-mse
endpoint_record_delta_replace_coeff16_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only
```

The Metal kernel stores only `owners`, `lengths`, and `trans_before` for the
reverse pass. It no longer stores per-run `segment_trans`, `segment_alpha`,
`weights`, or `segment_rgb`; reverse reloads site RGB/density and recomputes
the transmittance terms.

Validation:

- extension rebuild ok
- wrapper export and torch op export both true
- `test_gate4_moving_ray_slab_compiler`: 7 tests ok

Artifacts:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_packed_recompute_spot_render64_site24_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_packed_recompute_repeat2_spot_render64_site24_16.json
```

Results:

- first 16f spot: `11.795/9.314 ms` total/backward median
- warmed repeat: `7.377/6.689 ms`
- heldout PSNR unchanged: `14.232`

Read: recomputing reverse terms costs more than it saves in private memory. It
is clearly slower than the clean native cut-array keeper (`2.966/2.640 ms` at
16f), so I did not spend a full ladder. Keep this as a diagnostic mode only.

## Follow-up: packed scalar VJP shader, scale-ok but not keeper

I added a scalar-launch fused-MSE VJP that keeps the packed int32 endpoint row
representation from the framegroup16 keeper:

```text
endpoint-record-delta-replace-coeff16-packed-fused-mse
endpoint_record_delta_replace_coeff16_packed_mse_vjp_direct_atomic_rgb_only
```

This isolates launch shape from endpoint-row storage: unlike the existing
unpacked scalar coeff16 mode, this shader unpacks packed int32 base/change
records in-kernel; unlike framegroup16, it does not build or pass
`track_chunk_change_offsets_i16`.

Validation:

- Python compile passed for the harness and wrapper exports
- extension rebuild ok
- wrapper export and torch op export both true
- `test_gate4_moving_ray_slab_compiler`: 7 tests ok
- full train/eval ladder verifier reports `status=ok`

Artifacts:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_packed_scalar_spot_render64_site24_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_packed_scalar_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_packed_scalar_repeat20_render64_site24_2_4_8_16.verify.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_default_after_packed_scalar_render64_site24_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_packed_scalar_repeat2_spot_render64_site24_16.json
```

Results:

- first isolated 16f spot: `2.679/2.291 ms` total/backward median
- full ladder total median ms: `3.097 / 6.107 / 2.216 / 3.746`
- full ladder backward median ms: `2.133 / 3.939 / 1.892 / 3.020`
- full ladder total mean/median scale: `1.363x / 1.210x`
- full ladder backward mean/median scale: `1.515x / 1.416x`
- storage scale: `1.040x`
- heldout PSNR unchanged: `14.170 / 13.993 / 14.221 / 14.232`
- same-window default 16f control: `4.273/3.790 ms`
- packed-scalar 16f repeat after that control: `4.753/3.736 ms`

Read: this is a useful diagnostic fork, but not a keeper. It passes the formal
sublinear ladder and can beat the current framegroup16 path in a noisy
same-window comparison, but it does not beat the clean native cut-array keeper
artifact and the 16f repeats vary too much. The scalar launch shape is not
automatically bad when paired with packed records, but it is not stable enough
to auto-select.

## Follow-up: CPU rebase of endpoint delta tensors, negative

I added a harness-level diagnostic flag:

```text
--experimental-cpu-rebase-delta
```

It clones every endpoint delta tensor into a fresh contiguous CPU tensor before
the MPS transfer. This tests whether native sorted/cut-prep regressions are
caused by CPU tensor provenance or aliasing rather than record values.

Validation:

- Python compile passed for `train_eval_owner_run_tape.py`
- full train/eval 16f screens ran successfully with unchanged PSNR

Artifacts:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativesorted_rebase_spot_render64_site24_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativecutprep_rebase_spot_render64_site24_16.json
```

Results:

- native sorted + CPU rebase setup improved, but 16f warm timing stayed slow:
  `6.605/6.138 ms` total/backward median
- native cut-prep + CPU rebase was also slow:
  `6.226/5.456 ms` total/backward median
- CPU rebase itself costs under `1 ms` in these spots, so the loss is not clone
  overhead; the warm MPS path remains bad

Read: fresh contiguous CPU clones are not enough to recover the native
sorted/cut-prep setup wins. The problem is likely downstream MPS allocation or
device-side residency/order, not simple CPU aliasing. Keep the flag as a
diagnostic only.

## Follow-up: packed kernel-order selected-device diagnostic

I added another harness-level diagnostic flag:

```text
--experimental-kernel-order-packed-delta-device
```

For the promoted packed framegroup16 fused-MSE mode, it keeps only warm-kernel
tensors on MPS and allocates them in the same order the kernel consumes buffers:
coefficients, frame times, base offsets, packed base records, change offsets,
packed change records, chunk offsets, and configs. It also uses a Python boolean
mode marker instead of allocating an unused MPS flag tensor.

Validation:

- Python compile passed for `train_eval_owner_run_tape.py`
- 16f train/eval screen completed successfully with unchanged PSNR

Artifacts:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_kernelorder_packed_spot_render64_site24_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_default_after_kernelorder_spot_render64_site24_16.json
```

Results:

- kernel-order packed selected-device 16f:
  `4.326/3.866 ms` total/backward median
- same-window default control 16f:
  `7.890/5.768 ms` total/backward median
- clean saved keeper remains much faster:
  `2.966/2.640 ms` total/backward median at 16f
- heldout PSNR unchanged: `14.232`

Read: allocation order and avoiding unused MPS training tensors can matter in a
slow session, but this still does not recover the clean keeper timing. Do not
promote it; keep it as evidence that the MPS residency issue is real but not
solved by local selected-device ordering alone.

## Follow-up: native C++ endpoint-record packer, diagnostic negative

I added a native CPU bit-packing op:

```text
world_foam_lane2_fused_slab_v0.pack_endpoint_records_i32_cpu
```

and exposed it through the train/eval harness flag:

```text
--experimental-native-pack-records
```

This keeps the promoted packed framegroup16 shader unchanged and swaps only the
owner/left/right -> packed int32 conversion before the MPS transfer. The intent
was to test whether Python/Torch vectorized packing or packed-tensor provenance
was contributing to the slow-session behavior.

Validation:

- extension rebuild passed
- Python compile passed for `train_eval_owner_run_tape.py` and the Gate4 test
- focused Gate4 unit passed: `8` tests, including the new native bit-layout and
  invalid-record guards
- full train/eval ladder ran with unchanged PSNR

Artifacts:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativepack_spot_render64_site24_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativepack_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativepack_repeat20_render64_site24_2_4_8_16.verify.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_default_after_nativepack_spot_render64_site24_16.json
```

Results:

- native-pack 16f spot: `4.200/3.721 ms` total/backward median
- native-pack full ladder medians:
  - total `3.653 / 2.387 / 3.791 / 6.806 ms`
  - backward `3.189 / 1.965 / 3.189 / 4.765 ms`
- native-pack full ladder scale is still sublinear:
  - total mean/median `1.958x / 1.863x`
  - backward mean/median `1.863x / 1.494x`
  - storage `1.040x`
- robust verifier failed only on a contaminated `4f` backward max/median outlier
  (`8.086 > 8.000`), but the artifact should still be treated as non-keeper
  because absolute medians are worse than the clean promoted artifact
- same-window default 16f control after native-pack was also slow:
  `6.143/5.249 ms` total/backward median, so this was a contaminated MPS
  session and not a clean A/B

Read: moving pack math into a separate C++ CPU op is not the missing STAR-like
cleanliness. It can be kept as an explicit diagnostic flag, but do not promote
it. If packing is revisited, fold packed-record emission into the native
cut-array row walk so it does not add a second pass and does not replace the
current clean packed framegroup16 keeper until a full train/eval verifier wins.

## Follow-up: native-emitted packed records and stricter promotion gate

I then folded packed-record emission into the native Gate4 cut-array row walk:

```text
world_foam_lane2_fused_slab_v0.gate4_delta_replace_packed_from_cuts_cpu
--experimental-native-emitted-pack-records
```

The focused Gate4 unit passed after catching one misplaced single-slab guard,
and full train/eval completed with unchanged PSNR. The result is a correctness
pass but a timing negative:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativeemitpack_spot_render64_site24_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativeemitpack_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativeemitpack_repeat20_render64_site24_2_4_8_16.verify.json
research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativeemitpack_repeat20_render64_site24_2_4_8_16.reference_verify.json
```

Raw robust verification passed because the 2f row was already slow:

- total medians: `7.348 / 4.272 / 4.311 / 4.837 ms`
- backward medians: `5.118 / 3.714 / 3.345 / 3.987 ms`
- total/backward median scale: `0.658x / 0.779x`
- storage scale: `1.040x`
- PSNR unchanged: `14.170 / 13.993 / 14.221 / 14.232`

That is not a real promotion result. I extended
`verify_framegroup16_timing_robust.py` with an optional
`--reference-artifact` gate and `1.20x` default median non-regression limits
for total/backward timing. The clean native cut-array keeper passes against its
own verifier, while native-emitted packing now fails against that keeper:

- 2f total/backward: `3.255x / 2.645x` reference
- 4f total/backward: `1.983x / 2.027x` reference
- 8f total/backward: `1.750x / 1.560x` reference
- 16f total/backward: `1.631x / 1.510x` reference

Read: the promotion gate now matches the lab judgment. Future WorldFoam shader
forks must show sublinear frame scaling and must not be materially slower than
the clean keeper. This closes the misleading "sublinear because the anchor row
was contaminated" hole.

## Follow-up: 16-thread launch and deferred heldout MPS residency, negatives

I tested the first small shader fork suggested by the code audit: for the
packed framegroup path, use a 16-thread launch when `frame_count <= 16` instead
of always dispatching 32 local lanes. This changed only the C++ launch width in
`world_foam_lane2_metal.mm`, then rebuilt the local variant and reran the
focused Gate4 unit. The unit passed, but the 16f timing screen was a hard
negative:

```text
research_experiments/world_foam_lane2/results/2026-05-19_gate4_endpoint_record_packed_launch16_spot_render64_site24_16.json
```

- 16f median total/backward: `32.245 / 24.774 ms`
- clean keeper 16f total/backward: `2.966 / 2.640 ms`
- PSNR unchanged

I reverted the launch change and rebuilt the extension; the Metal source diff
is empty again and the Gate4 unit passes after the restore. Read: Apple/MPS
really wants the 32-lane launch shape here, or at least the idle lanes are not
the bottleneck. Do not revisit direct16 without a deeper occupancy/register
profile.

I also added an opt-in harness flag:

```text
--defer-heldout-device
```

This keeps heldout MPS tape/targets out of the timed train loop and prepares
them only before final heldout rendering. The smoke passed and records:

- `defer_heldout_device=true`
- `heldout_tape_prepared_after_timing=true`
- `timed_mps_residency_scope=train_tape_targets_site_only`

Full ladder artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-19_gate4_endpoint_record_deferheldout_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-19_gate4_endpoint_record_deferheldout_repeat20_render64_site24_2_4_8_16.reference_verify.json
```

The raw scale looks sublinear because 2f/4f are slow, but the reference verifier
correctly rejects it:

- medians total: `19.120 / 12.181 / 9.518 / 7.396 ms`
- medians backward: `13.960 / 9.865 / 7.600 / 6.057 ms`
- 16f is still `2.494x / 2.294x` slower than the clean keeper
- 4f has max/median contamination (`8.926x` total, `10.814x` backward)
- PSNR/storage unchanged

Read: removing heldout residency is not a free cleanup; allocation/lifetime
order is another sensitive variable and this order is worse. Keep the flag only
as a diagnostic. The current clean keeper remains the native cut-array/direct
delta row walk plus default packed framegroup32 launch/residency order.

## Follow-up: packed local-owner reduction, negative

The owner-list diagnostic showed the current `render64/site24/16f` moving-camera
fixture has per-track/chunk owner counts well under the 16-slot reduction cap:

```text
chunks=8192 min=4 max=12 mean=7.58 p50=8 p95=10 p99=11 over16=0
```

That made a packed local-owner variant worth trying: keep packed int32 endpoint
records, replace each record owner byte with a chunk-local owner slot, pass
`track_chunk_owner_offsets_i32` and `track_chunk_owner_i16` to the kernel, and
map slot -> true site id in the shader. The goal was to avoid the old
i16x3-ownerreduce kernel's per-segment linear scan over `tg_owner_ids`.

Implemented, tested, then removed mode:

```text
endpoint-record-delta-replace-coeff16-packed-framegroup16-localowner-fused-mse
```

Files touched:

- `research_experiments/world_foam_lane2/train_eval_owner_run_tape.py`
- `research_experiments/world_foam_lane2/test_probe_endpoint_record_edit_replay.py`
- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/csrc/bindings.cpp`
- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/csrc/metal/world_foam_lane2_metal.mm`
- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/csrc/metal/world_foam_lane2_shared_replay_tensor.metal`

Verification passed:

```text
uv run --project ... python setup.py build_ext --inplace
PYTHONPATH=third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:research_experiments/world_foam_lane2:src/train \
  .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_probe_endpoint_record_edit_replay.EndpointRecordEditReplayTests.test_delta_replace_packed_localowner_framegroup_matches_scalar_above_small_site_cap -v
PYTHONPATH=third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:research_experiments/world_foam_lane2:src/train \
  .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler -v
```

The new parity test covers site ids above the old small-site reduction cap and
matches scalar loss/grad. Gate4 compiler regression remains green.

Train/eval artifacts:

```text
research_experiments/world_foam_lane2/results/2026-05-19_gate4_endpoint_record_localowner_smoke_render32_site24_2.json
research_experiments/world_foam_lane2/results/2026-05-19_gate4_endpoint_record_localowner_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-19_gate4_endpoint_record_localowner_repeat20_render64_site24_2_4_8_16.reference_verify.json
```

Full ladder medians:

- total: `8.060 / 6.926 / 5.751 / 69.411 ms`
- backward: `5.847 / 5.576 / 4.407 / 63.366 ms`
- PSNR unchanged: `14.170 / 13.993 / 14.221 / 14.232`

The stricter reference verifier rejects every row against the clean keeper:

- 2f total/backward: `3.570x / 3.022x` slower than keeper
- 4f total/backward: `3.215x / 3.042x`
- 8f total/backward: `2.334x / 2.056x`
- 16f total/backward: `23.404x / 24.002x`

The built-in scale also fails (`8.612x` total median and `10.837x` backward
median versus an `8x` frame scale). The older i16x3 ownerreduce spot was only
mildly slow at 16f (`4.645/4.130 ms`), so this is not just "owner lists are
always bad"; the packed local-slot shader shape is pathological on this MPS
path.

Cleanup follow-up: the local-owner Python mode, scalar parity test, C++ binding,
and Metal tensor kernel were removed from the hot
`world_foam_lane2_fused_slab_v0` variant after this negative. The extension was
rebuilt and Gate4 compiler regression still passed (`8` tests). A fresh default
16f control after cleanup was also slow (`54.638 ms` total median,
`47.357 ms` backward median) despite unchanged PSNR, while an unrelated
long-running `ai_trader` pytest process had Torch/AGXMetal loaded. Treat that
fresh control as contaminated session evidence, not a new keeper baseline. The
saved clean keeper remains `2.966/2.640 ms` at 16f.

## Follow-up: benchmark environment guard

The post-cleanup 16f controls did not return to keeper speed:

- `2026-05-19_gate4_endpoint_record_default_control_after_localowner_revert_spot_render64_site24_16.json`:
  `54.638 ms` total median, `47.357 ms` backward median
- `2026-05-19_gate4_endpoint_record_default_control_cleanprocess_spot_render64_site24_16.json`:
  `27.479 ms` total median, `20.725 ms` backward median

The second spot now has a reference verifier:

```text
research_experiments/world_foam_lane2/results/2026-05-19_gate4_endpoint_record_default_control_cleanprocess_spot_render64_site24_16.reference_verify.json
```

It fails against the clean keeper by `9.265x` total median and `7.850x`
backward median. To keep this from masquerading as shader evidence,
`verify_framegroup16_timing_robust.py` now accepts single-frame
`--expected-frames 16` reference checks, and train/eval artifacts now include a
`benchmark_environment` start/end process snapshot. The env-metadata smoke:

```text
research_experiments/world_foam_lane2/results/2026-05-19_gate4_endpoint_record_envmeta_smoke_render32_site12_2.json
```

recorded `benchmark_environment.status=contended`, with unrelated `ai_trader`
Python jobs and a clang compile consuming CPU. Its verifier:

```text
research_experiments/world_foam_lane2/results/2026-05-19_gate4_endpoint_record_envmeta_smoke_render32_site12_2.verify.json
```

fails solely because the environment is contended. Future shader promotion
should require both the reference-artifact timing gate and
`benchmark_environment.status=ok`; otherwise the run is just a diagnostic.

Follow-up refinement: `benchmark_environment` now separates high-severity
`blocking_processes` from low-CPU `background_processes`. Background-only
Python daemons do not contaminate artifacts; high-CPU Python jobs or explicit
Torch/Metal/MPS/PyTest matches still do. The full post-cleanup control:

```text
research_experiments/world_foam_lane2/results/2026-05-19_gate4_endpoint_record_default_control_envok_repeat20_render64_site24_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-19_gate4_endpoint_record_default_control_envok_repeat20_render64_site24_2_4_8_16.reference_verify.json
```

is rejected. It appears "sublinear" only because the low-frame rows are badly
throttled:

- 2f: `68.615 ms` total median, `60.824 ms` backward median
- 4f: `30.344 ms` total median, `22.521 ms` backward median
- 8f: `10.294 ms` total median, `8.263 ms` backward median
- 16f: `8.132 ms` total median, `6.865 ms` backward median

The reference verifier rejects every row (`30.392x/31.432x` slower at 2f,
`2.742x/2.600x` slower at 16f), and `benchmark_environment.status=contended`
shows active high-CPU `ai_trader` jobs at start/end. This is exactly the false
positive the new gate is meant to prevent.

Follow-up fail-fast runner guard: `train_eval_owner_run_tape.py` now has
`--benchmark-environment-check-only` and `--require-benchmark-environment-ok`.
The first prints the process snapshot and exits nonzero if promotion would be
blocked; the second aborts before tape construction. This avoids spending a
full sweep only to learn that the reference verifier must reject it. The helper
tests cover keyword-boundary matching, background-only environments, contended
environments, and start/end merge behavior:

```text
PYTHONPATH=third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:research_experiments/world_foam_lane2:src/train \
  .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_benchmark_environment -v
```

The live check-only command currently exits `2`, with multiple high-CPU
`ai_trader` Python jobs and a PyTest process in `blocking_processes`. Wait for
that to clear before the next keeper rerun or shader fork.

Wrapper follow-up: added
`research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py`.
Default behavior is exactly the promotion flow we want for the next clean MPS
window:

1. run `train_eval_owner_run_tape.py --benchmark-environment-check-only`
2. run the default `2,4,8,16` / `render64` / `site24` / `20` step
   `endpoint-record-delta-replace-coeff16-auto-framegroup16-fused-mse` train/eval
   with `--require-benchmark-environment-ok`
3. run `verify_framegroup16_timing_robust.py` against the clean keeper artifact

Smoke coverage:

```text
.venv/bin/python -m py_compile research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py ...
.venv/bin/python research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  --run-id 2026-05-19_framegroup16_promotion_dryrun --dry-run
.venv/bin/python research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  --run-id 2026-05-19_framegroup16_promotion_preflight_blocked
```

The real wrapper invocation stopped at preflight with exit `2` and wrote:

```text
research_experiments/world_foam_lane2/results/2026-05-19_framegroup16_promotion_preflight_blocked.promotion_summary.json
```

`status=preflight_failed`, so no train/eval or verifier was launched. This is
the desired behavior while the machine is still busy.

Wait-mode follow-up: the wrapper now supports
`--wait-for-benchmark-environment-ok`, with a default one-hour timeout and
30-second polling. A dry-run verifies the command chain and records
`wait_timeout_s=3600.0`:

```text
.venv/bin/python research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  --run-id 2026-05-19_framegroup16_promotion_wait_default_dryrun \
  --wait-for-benchmark-environment-ok --dry-run
```

A short live blocked smoke:

```text
.venv/bin/python research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  --run-id 2026-05-19_framegroup16_promotion_wait_short_blocked \
  --wait-for-benchmark-environment-ok --wait-timeout-s 2 --wait-interval-s 1
```

wrote `2026-05-19_framegroup16_promotion_wait_short_blocked.promotion_summary.json`
with `status=preflight_failed`, one recorded attempt, and no train/eval. The
current blocker remains high-CPU `ai_trader`/PyTest work.

Summary follow-up: promotion summaries now embed a compact `verify_result`
after the verifier runs, with status, clean/non-regressed booleans,
contamination/failure lists, and per-frame total/backward medians plus PSNR.
This keeps a successful or failed promotion auditable from a single summary
JSON. Tested with:

```text
.venv/bin/python -m unittest research_experiments.world_foam_lane2.test_framegroup16_promotion_gate -v
.venv/bin/python research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  --run-id 2026-05-19_framegroup16_promotion_verifybrief_dryrun \
  --wait-for-benchmark-environment-ok --dry-run
```

The latest preflight is still blocked by a high-CPU PyTest plus a STAR UVT
feature-kernel process, so no new WorldFoam timing artifact should be promoted
yet.

Live-summary follow-up: the promotion wrapper now writes the summary after each
preflight attempt while waiting, so a long wait can be inspected before it exits.
Tested with:

```text
.venv/bin/python -m unittest research_experiments.world_foam_lane2.test_framegroup16_promotion_gate -v
.venv/bin/python research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  --run-id 2026-05-19_framegroup16_promotion_live_summary_blocked \
  --wait-for-benchmark-environment-ok --wait-timeout-s 2 --wait-interval-s 1
```

The live blocked summary has `status=preflight_failed`, one attempt, and the top
blocking process. The current blockers include high-CPU `ai_trader` jobs plus a
STAR UVT feature-kernel process, so keep waiting before launching promotion.

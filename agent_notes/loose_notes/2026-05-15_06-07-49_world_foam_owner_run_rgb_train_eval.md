# World Foam Owner-Run RGB Train/Eval

Added `research_experiments/world_foam_lane2/train_eval_owner_run_tape.py` to
test whether the same-owner run compression can carry an actual RGB/site-RGBA
optimizer loop, not just isolated replay/VJP parity.

The first smoke caught a harness bug: train and heldout have different view
counts, so the heldout tape must render with its own track layout. After fixing
that, the render16 2-frame smoke passed with nonzero gradients, a decreasing
loss, finite output, and owner-run storage around 10% of the full segment tape.

Ran a 2/4/8/16 render32 sweep with the owner-run probe defaults first:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --steps 5 \
  --warmup-steps 1 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_owner_run_rgb_train_eval_render32_2_4_8_16.json
```

That passed, but those near/far/density defaults did not match the fused
train/eval artifacts, so it is not the comparison artifact to cite.

The matched fused-parameter sweep is the useful result:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --steps 5 \
  --warmup-steps 1 \
  --near 0.1 \
  --far 6.0 \
  --density 10.0 \
  --invalid-epsilon 1.0e-6 \
  --transmittance-threshold 1.0e-4 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_owner_run_rgb_train_eval_fusedparams_render32_2_4_8_16.json
```

Saved artifact status: `ok`.

Key numbers:

- total step scale 2f->16f: `1.259x` for `8x` frames
- 16f total step: `7.027 ms`
- 16f render: `2.030 ms`
- 16f RGB-only VJP/backward: `3.182 ms`
- 16f heldout PSNR: `13.273996678676685`
- 16f train PSNR: `12.102742311280426`
- 16f train full segments: `1301934`
- 16f train owner-run segments: `62968` (`0.0484x`)
- 16f train owner-run storage: `0.0563x` of full segment tape
- max owner-run segments per sample: `4`

Compared with the existing fused `direct_atomic_grad_only` winner at the same
smoke scale, owner-run RGB train/eval is faster at 16f (`7.03 ms` vs
`9.32 ms`) and matches heldout PSNR within the status-verifier tolerance. This
is meaningful practical progress.

Scope boundary remains important: this is an isolated manual optimizer path
over fixed geometry and site RGBA. It does not prove full trainer integration,
geometry gradients, density-independent depth replay, or STAR-UVT-style
structural sublinearity. The owner-run exact segment count still scales worse
than frame count in the saved structural probe, even though absolute tape size
is much smaller.

Updated:

- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/ops.py`
- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/__init__.py`
- `research_experiments/world_foam_lane2/smoke_segment_tape_autograd_mps.py`
- `research_experiments/world_foam_lane2/summarize_fused_slab_mixed_results.py`
- `research_experiments/world_foam_lane2/verify_fused_slab_status_summary.py`
- `research_experiments/world_foam_lane2/README.md`

After the train/eval artifact, added `segment_tape_rgba_depth_autograd(...)` in
the fork package. It wraps the compact segment-tape replay op and uses the
existing Metal VJP kernels in backward, exposing gradients only for
`site_rgba`. The smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/smoke_segment_tape_autograd_mps.py \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_segment_tape_autograd_smoke_render16_2f.json
```

passed for both `direct_atomic_grad_only` and `direct_atomic_track`; max
relative gradient error versus the explicit VJP was about `4e-7`.

Then reran owner-run RGB train/eval through the autograd wrapper instead of the
manual optimizer:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --steps 5 \
  --warmup-steps 1 \
  --near 0.1 \
  --far 6.0 \
  --density 10.0 \
  --invalid-epsilon 1.0e-6 \
  --transmittance-threshold 1.0e-4 \
  --optimizer-mode autograd \
  --segment-tape-vjp-mode direct_atomic_grad_only \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_owner_run_rgb_train_eval_autograd_fusedparams_render32_2_4_8_16.json
```

Saved artifact status: `ok`.

Key autograd numbers:

- total step scale 2f->16f: `1.118x` for `8x` frames
- 16f total step: `6.037 ms`
- 16f render: `1.973 ms`
- 16f backward: `3.183 ms`
- 16f heldout PSNR: `13.273996678676685`
- 16f train owner-run storage: `0.0563x` of full segment tape

This supersedes the manual optimizer artifact as the primary practical
owner-run train/eval evidence. It still does not mean the path is wired into
the main fused-slab trainer or solves geometry/depth gradients.

Added a structural owner-run boundary endpoint probe:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/probe_owner_run_boundary_tape.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_owner_run_boundary_tape_probe_render32_2_4_8_16.json
```

Saved artifact status: `informational`.

Key endpoint-tape numbers:

- owner-run endpoint records exactly match current owner-run counts/owners
- max endpoint-derived run length error: `2.22e-16`
- 16f endpoint storage: `0.0563x` of full segment tape
- 16f runs: `62968` versus `1301934` full segments
- run count scale 2f->16f: `9.885x`, which is worse than the `8x` frame scale

This proves boundary ids plus ray coefficients can replace per-run length
storage, but it also confirms the owner-run record count is still not
STAR-style structurally sublinear. Depth effective-mid replay remains open.

Extended that endpoint probe with the first density-aware replay check:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/probe_owner_run_boundary_tape.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_owner_run_boundary_tape_probe_render32_2_4_8_16.json
```

The endpoint-only continuous absorption midpoint
`left + 1 / density - length / expm1(density * length)` is well-defined and
keeps alpha essentially exact, but it does not reproduce the current
segment-mid depth tape after same-owner internal cuts are discarded:

- max endpoint-derived length error: `2.22e-16`
- max endpoint alpha error versus current owner-run tape at 16f: `3.75e-10`
- max endpoint density-depth error versus current owner-run tape: `0.412`
- mean 16f endpoint density-depth error: `0.0824`

So endpoint records are enough for RGB/alpha replay, but not enough for exact
current-depth replay. Matching the current depth contract requires internal
moments/cuts, or we have to make an explicit depth-semantic change to continuous
absorption depth.

Added an owner-run internal-cut probe to test that tradeoff directly:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/probe_owner_run_internal_tape.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_owner_run_internal_tape_probe_render32_2_4_8_16.json
```

Saved artifact status: `informational`.

Key result:

- active internal cuts match current-density RGB/alpha/depth exactly
- active internal nested CSR at 16f is `0.148x` of full segment CSR
- active internal cuts are not density independent: at half density, 16f max
  alpha error is `0.00999` and max depth error is `0.0105`
- all internal cuts preserve density-independent replay by keeping every segment,
  but 16f nested CSR is `0.738x` of full segment CSR
- all-owner-run endpoint storage at 16f is compact at `0.111x` of full segment
  CSR, but only if depth semantics change to continuous absorption within a
  same-owner run
- active internal segment count scales `8.73x`; all internal segment count
  scales `8.03x` for an `8x` frame increase

This makes the fork's current boundary sharper: exact current-depth replay is
not blocked mathematically, but the density-independent exact version looks
much less STAR-like because it moves back toward full per-frame segment storage.
There is a compact density-independent endpoint representation if we change the
depth semantic, but that is not a drop-in replacement for the current
segment-mid tape.

Extended the train/eval harness with `--tape-mode {owner-run,active-internal,full}`
while keeping owner-run as the default. Then ran the active-internal autograd
sweep:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --steps 5 \
  --warmup-steps 1 \
  --tape-mode active-internal \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_active_internal_rgb_train_eval_autograd_fusedparams_render32_2_4_8_16.json
```

Saved artifact status: `ok`.

Key active-internal train/eval numbers:

- total step scale 2f->16f: `1.60x` for `8x` frames
- 16f total step: `8.67 ms`
- 16f render: `2.38 ms`
- 16f backward: `5.18 ms`
- 16f heldout PSNR: `13.273993928035445`
- 16f selected segment ratio versus full: `0.163x`
- 16f selected storage ratio versus full: `0.170x`

This exact-current-depth path is slightly faster than the current fused winner
at 16f (`8.67 ms` vs `9.32 ms`) and matches PSNR, but it is slower than the
owner-run RGB path (`6.04 ms`) and remains structurally non-sublinear in selected
segment count (`8.73x` for an `8x` frame increase).

Ran the full segment-tape train/eval baseline through the same autograd wrapper:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --steps 5 \
  --warmup-steps 1 \
  --tape-mode full \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_full_segment_rgb_train_eval_autograd_fusedparams_render32_2_4_8_16.json
```

Saved artifact status: `ok`.

Key full-tape train/eval numbers:

- total step scale 2f->16f: `1.82x` for `8x` frames
- 16f total step: `9.79 ms`
- 16f render: `2.74 ms`
- 16f backward: `5.70 ms`
- 16f heldout PSNR: `13.273979487197515`
- 16f selected storage ratio versus full: `1.0x`

This is the exact fixed-geometry density-independent replay cost baseline. It
matches PSNR but is slower than owner-run and active-internal, slightly slower
than the current fused winner at 16f, and not compact by definition.

Regenerated and verified:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/summarize_fused_slab_mixed_results.py \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_status_summary.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_fused_slab_status_summary.py \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_status_summary_verifier.json
```

Verifier status: `ok`, failures `[]`.

## Endpoint-Run Continuous-Depth Shader Path

Added a new endpoint-run shader path to
`world_foam_lane2_fused_slab_v0`:

- `wf2_endpoint_run_rgba_depth_replay_tensor`
- `wf2_endpoint_run_vjp_direct_atomic_grad_only_tensor`
- Python wrappers:
  - `endpoint_run_rgba_depth_replay`
  - `endpoint_run_vjp_direct_atomic_grad_only`
  - `endpoint_run_rgba_depth_autograd`

The representation stores same-owner run owner plus start/end depth and changes
depth from the current segment-mid weighted depth to continuous absorption
inside a same-owner run. RGB/alpha are still the same physical compositing for
the owner run; the depth semantic is the deliberate change.

Built the variant successfully:

```bash
( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

Smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/probe_endpoint_run_tape.py \
  --frame-counts 2 \
  --render-size 16 \
  --timing-iters 1 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_tape_smoke_render16_2f.json
```

Smoke status `ok`; Metal forward max error versus torch continuous replay was
`5.96e-8`, and Metal VJP relative error versus torch autograd was about
`1.07e-6`.

Full endpoint-run probe:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/probe_endpoint_run_tape.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --timing-iters 3 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_tape_probe_render32_2_4_8_16.json
```

Saved status `informational`:

- forward max error versus torch: `5.96e-8`
- VJP max relative error versus torch autograd: `1.02e-4`
- 16f endpoint storage ratio versus full segment CSR: `0.111x`
- 16f endpoint runs: `134747` versus `1301934` full segments
- max endpoint runs per sample: `7`
- endpoint run scale 2f->16f: `8.61x` for `8x` frames

So this fixes the compact density-independent representation if we accept the
continuous-depth semantic, but it still does not create STAR-style structural
sublinearity.

Extended `train_eval_owner_run_tape.py` with `--tape-mode endpoint-run` and ran:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --steps 5 \
  --warmup-steps 1 \
  --tape-mode endpoint-run \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_rgb_train_eval_autograd_fusedparams_render32_2_4_8_16.json
```

Saved status `ok`:

- 16f total step: `7.84 ms`
- 16f render: `2.29 ms`
- 16f backward: `4.22 ms`
- total scale 2f->16f: `1.26x`
- 16f heldout PSNR: `13.273873589186554`
- 16f selected storage ratio versus full: `0.111x`
- 16f selected run ratio versus full segments: `0.103x`

This endpoint path is faster than the current fused winner (`9.32 ms`) and
active-internal (`8.67 ms`), but slower than the current-density owner-run path
(`6.04 ms`). It is a practical semantic-change candidate, not a completion
claim.

Regenerated the status summary and verifier after adding endpoint-run evidence.
Verifier remains `ok` with failures `[]`.

Follow-up audit tightening: the exact segment-record delta probe already existed
as `2026-05-15_segment_record_delta_tape_probe_render32_2_4_8_16.json`, but it
was not included in the fused-slab mixed status summary. I wired that probe
into `summarize_fused_slab_mixed_results.py`, `verify_fused_slab_status_summary.py`,
and the World Foam Lane 2 README.

The exact record probe stores `(owner, left_cut_id, right_cut_id)`, so boundary
ids can recover segment length/mid while owners match the segment tape exactly.
The result is important because it is a negative exact-replay result:
16-frame replacement record storage is `1.015x` full segment CSR, edit-op
record storage is `0.909x`, exact record count scales `8.06x`, and record
edit ops scale `7.82x` for an `8x` frame-count increase. That closes a status
blind spot: exact owner+cut-id replay works as math, but it is not the compact
STAR-like tape we need.

Also tightened the STAR speed reference summary so it records the measured
runtime scaling, not just the 16-frame timing. The saved tiny STAR direct-atomic
reference scales mean step time `1.23x` from 2 to 16 frames while frame count
scales `8x`. The status summary now marks that as runtime-sublinear evidence
with an explicit note that it is still not a matched quality/capacity result.

Added `probe_endpoint_record_delta_tape.py` to test whether the continuous
endpoint-run tape has another STAR-like delta layer. The corrected all-run
version verifies `(owner, left_cut_id, right_cut_id)` records against
`compress_same_owner_endpoint_runs`.

Saved artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_delta_tape_probe_render32_2_4_8_16.json
```

This is the most promising structural signal so far:

- endpoint records match endpoint-run counts/owners
- endpoint record count still is not sublinear: `8.61x` for `8x` frames
- endpoint record edit ops scale only `1.87x`
- endpoint record edit-op storage scales `1.55x`
- 16f full endpoint-record CSR is `0.111x` full segment CSR
- 16f endpoint edit-op stream is `0.0256x` full segment CSR and `0.230x` of
  full endpoint-record CSR

This is not a shipped replay shader, but it gives the first concrete STAR-port
target for World Foam: replay endpoint owner+cut-id edit streams instead of
materializing every endpoint record per frame.

Implemented the first endpoint-record delta replay shader and a negative
stepping-stone variant.

First, I added a depth-float replacement-row shader:

```text
research_experiments/world_foam_lane2/probe_endpoint_delta_replace_replay.py
```

It stores first-frame endpoint rows plus changed rows with explicit start/end
depth floats. The Metal forward and VJP match endpoint-run replay on the 2f
smoke, but the storage is not the structural win: at 2f it is `1.083x` full
endpoint CSR because moving-camera start/end depths make every row look changed.
That was useful because it proved the replay/VJP shape while confirming that
the STAR-like signal lives in owner+cut ids, not depth floats.

Then I added the owner+cut-id replacement-row shader:

```text
research_experiments/world_foam_lane2/probe_endpoint_record_delta_replay.py
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_delta_replay_render32_2_4_8_16.json
```

This shader stores `(owner, left_cut_id, right_cut_id)` rows and recomputes
endpoint depths from boundary ids and moving rays inside Metal. It matches the
endpoint-run shader while keeping replacement-row storage sublinear:

- max forward error versus endpoint-run replay: `8.94e-7`
- max VJP relative error versus endpoint-run replay: `2.82e-6`
- record-delta replacement storage scale 2f->16f: `1.87x` for an `8x` frame
  increase
- 16f record-delta storage: `0.235x` endpoint CSR, `0.0261x` full segment CSR
- 16f record-delta forward/VJP: `3.61 ms` / `4.17 ms`

Important bug fixed during the run: the shared `wf2_clear_site_rgba_grad_tensor`
assumes `site_count` is at `config_i32[2]`. The owner+cut-id shader config has
`frame_count` there and `site_count` at index 3, so 2f/4f/8f VJPs left some
site rows uncleared and produced huge false gradient errors. The fix was to
allocate the cut-id replay gradient tensor with `torch::zeros` in the C++ entry
point instead of using the shared clear kernel with the incompatible config
layout.

Status after this chunk: a real replacement-row endpoint-record replay shader
exists and is green, but the most compact edit-op stream shader and main-trainer
integration are still open.

Added the owner+cut-id edit-op replay shader:

```text
research_experiments/world_foam_lane2/probe_endpoint_record_edit_replay.py
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_replay_render32_2_4_8_16.json
```

This shader reconstructs endpoint rows from first-frame records plus
insert/delete/replace ops, then recovers endpoint depths from boundary ids and
moving rays inside Metal. It matches the endpoint-run shader numerically and
keeps the delta representation sublinear across the 2/4/8/16 moving-camera
sweep:

- max forward error versus endpoint-run replay: `8.94e-7`
- max VJP relative error versus endpoint-run replay: `6.19e-6`
- endpoint run count scale 2f->16f: `8.61x` for `8x` frames
- edit op scale 2f->16f: `1.87x`
- edit storage scale 2f->16f: `1.53x`
- 16f edit storage: `0.235x` endpoint CSR, `0.0261x` full segment CSR
- 16f timing: edit forward/VJP `3.61 ms` / `3.43 ms`; endpoint forward/VJP
  `2.05 ms` / `2.69 ms`

Interpretation: World Foam now has a practical sublinear endpoint-record delta
representation and a real Metal edit-op replay shader. The shader is not yet a
speed win; replaying the edit script inside the kernel is slower than the
endpoint-run baseline in this microprobe. This is storage-sublinear in practice,
not yet STAR-UVT competitive as a training path.

Added a fixed-geometry RGB/site-RGBA autograd train/eval path for the
endpoint-record edit shader:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_rgb_train_eval_autograd_fusedparams_render32_2_4_8_16.json
```

Implementation details:

- Added `endpoint_record_edit_rgba_depth_autograd` in the variant Python ops.
- Extended `train_eval_owner_run_tape.py` with `--tape-mode endpoint-record-edit`.
- The train and heldout tapes build owner+cut-id edit streams from the same
  moving-camera rays and call the edit replay/VJP kernels through PyTorch
  autograd.

Saved 2/4/8/16 result:

- status `ok`
- 16f heldout PSNR `13.273873589186554`, matching the endpoint semantic/fused
  parameter sweep
- 16f total/render/backward `6.18 ms` / `2.59 ms` / `2.76 ms`
- total step scale `0.70x` for `8x` frames on this smoke-scale timing run
- edit op scale `1.87x`
- edit storage scale `1.53x`
- 16f selected edit storage `0.0261x` full segment CSR and `0.235x` endpoint
  CSR

This closes the previous "no endpoint-record train/eval path" gap. Remaining
scope gaps are main-trainer integration, geometry gradients, non-smoke timing
stability, and matched STAR UVT quality/capacity evidence.

Because the standalone train/eval timing contradicted the isolated shader
timing, I also wrote a paired same-process comparison artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_vs_record_edit_current_process_train_eval_render32_2_4_8_16.json
```

That sidecar reruns endpoint-run and endpoint-record-edit back-to-back under
the same process. At 16f:

- endpoint-run total/render/backward `6.54 ms` / `2.17 ms` / `3.37 ms`
- endpoint-record-edit total/render/backward `8.28 ms` / `3.24 ms` /
  `4.11 ms`
- edit/endpoint total ratio `1.27x`
- endpoint-record-edit storage `0.0261x` full segment CSR versus endpoint-run
  `0.111x`
- heldout PSNR matches within `1e-6`

Interpretation update: endpoint-record edit is now a real compact train/eval
path, but same-process timing says it is still a storage win, not a runtime win
against endpoint-run. The next technical target is replay optimization or a
row-cache/prefix representation, not more proof that the current edit stream is
compact.

Followed with a small cut-depth cache in the endpoint-record edit forward/VJP
Metal loops. Consecutive endpoint rows often share the previous right cut as
the next left cut, so the shader now reuses that recovered depth when the cut id
matches instead of recomputing both boundaries every row.

Isolated replay artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_replay_cutcache_render32_2_4_8_16.json
```

Result:

- status `ok`
- max forward error versus endpoint-run `8.94e-7`
- max VJP relative error versus endpoint-run `6.91e-6`
- 16f edit forward/VJP `3.29 ms` / `3.00 ms`
- endpoint-run forward/VJP in the same probe `2.01 ms` / `2.64 ms`
- edit op scale still `1.87x` for `8x` frames
- edit storage scale still `1.53x`
- 16f edit storage still `0.0261x` full segment CSR and `0.235x` endpoint CSR

Post-cache paired same-process train/eval artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_vs_record_edit_current_process_train_eval_cutcache_render32_2_4_8_16.json
```

At 16f:

- endpoint-run total/render/backward `7.47 ms` / `2.54 ms` / `3.72 ms`
- endpoint-record-edit total/render/backward `7.93 ms` / `3.09 ms` /
  `3.97 ms`
- edit/endpoint total ratio `1.06x`
- endpoint-record-edit storage `0.0261x` full segment CSR versus endpoint-run
  `0.111x`
- heldout PSNR matches exactly in the saved summaries

Interpretation update: the cut cache made the edit path less bad and almost
caught endpoint-run on the smoke timing, but it still has not crossed into a
runtime win. The evidence is now: compact in theory, compact in measured
op/storage counts, sublinear on smoke-scale train/eval timing, but not yet
wall-clock faster than endpoint-run or STAR-UVT competitive.

Added an RGB-only endpoint-record edit VJP sidecar after the cut-cache pass:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_replay_rgbonly_render32_2_4_8_16.json
```

This adds a dedicated RGB-loss VJP kernel for the edit-op replay path. It is
correct, but not a stable speed win:

- max RGB-only VJP relative error versus the full edit VJP with zero
  alpha/depth adjoints: `2.86e-6`
- max full edit VJP relative error versus endpoint-run: `5.00e-6`
- 16f edit storage remains `0.0261x` full segment CSR and `0.235x` endpoint CSR
- 16f isolated timing: endpoint VJP `2.16 ms`, full edit VJP `2.63 ms`,
  RGB-only edit VJP `2.85 ms`

I also reran the paired endpoint-run versus endpoint-record-edit train/eval
comparison with the RGB-only path for a longer 12-step smoke repeat:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_vs_record_edit_current_process_train_eval_rgbonly_repeat12_render32_2_4_8_16.json
```

Latest 16f result:

- endpoint-run total/render/backward `5.07 ms` / `1.57 ms` / `2.77 ms`
- endpoint-record-edit total/render/backward `6.74 ms` / `2.35 ms` /
  `3.56 ms`
- edit/endpoint total ratio `1.33x`
- heldout PSNR matches within `5e-7`

Interpretation update: the RGB-only kernel should stay because it is a correct
and narrow autograd optimization hook, but it does not fix the main runtime
problem. The speed sign has flipped across short MPS smoke repeats, so the
status summary now treats RGB-only as correctness/storage evidence and keeps
the speed/STAR-competitive claim false.

Ran the same endpoint-run versus endpoint-record-edit comparison in
`manual-vjp` mode to remove most autograd-wrapper ambiguity:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_vs_record_edit_current_process_train_eval_manualvjp_repeat12_render32_2_4_8_16.json
```

Latest 16f result:

- endpoint-run total/render/backward `4.57 ms` / `1.40 ms` / `2.46 ms`
- endpoint-record-edit total/render/backward `5.32 ms` / `2.26 ms` /
  `2.43 ms`
- edit/endpoint total ratio `1.16x`
- edit/endpoint backward ratio `0.99x`
- edit/endpoint render ratio `1.61x`
- heldout PSNR matches exactly in the saved summary

Interpretation update: manual VJP shows the gradient side can get to parity,
but total step still loses because edit forward replay is much slower. The next
real optimization target is row reconstruction / forward replay layout, not
another RGB-only VJP variant.

Added and measured a forward-only track-loop endpoint-record edit replay shader:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_replay_trackloop_render32_2_4_8_16.json
```

The kernel launches by track and walks frames in order, applying per-track edit
ops incrementally before rendering each frame. Correctness is good:

- max track-loop forward absolute error versus endpoint-run: `8.94e-7`
- edit ops scale `1.87x` and edit storage scales `1.53x` while frame count
  scales `8x`
- 16f edit storage remains `0.235x` endpoint CSR and `0.0261x` full segment CSR

But it is not a runtime win. At 16f, endpoint-run forward is `1.18 ms`,
existing edit forward is `1.96 ms`, and track-loop forward is `2.15 ms`. The
track-loop variant probably gives up too much parallelism to amortize row-edit
application. It is now recorded in the canonical summary as a correct rejected
forward-optimization sidecar.

Implemented two more forward replay variants after the STAR/World Foam scaling
inspection:

- `framegroup16`: one threadgroup per track, lane 0 materializes all frame rows
  into threadgroup memory, and frame lanes render in parallel. Correct, but bad
  at 16f: endpoint-run forward `1.51 ms`, original edit `2.10 ms`,
  framegroup16 `5.14 ms`.
- `block4`: store anchor endpoint-record rows every four frames and replay only
  in-block edit ops while keeping one thread per sample. This is the first
  positive forward result:
  - max block4 forward absolute error versus endpoint-run: `8.94e-7`
  - 16f block4 forward: `1.53 ms`
  - 16f endpoint-run forward: `2.07 ms`
  - 16f original edit forward: `3.56 ms`
  - 16f track-loop forward: `2.21 ms`
  - 16f block4 storage: `0.395x` endpoint CSR, `0.0438x` full segment CSR

Interpretation update: the STAR-like transferable idea is not "serialize a
track through time." It is "materialize bounded time blocks so the hot render
path does not replay unbounded history."

Followed by wiring `endpoint-record-edit-block4` into the RGB train/eval probe.
The train/eval path uses block4 forward replay and the existing exact
endpoint-record edit VJP inside a new autograd wrapper; it is not a dedicated
block4 VJP.

Artifacts:

- `research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_block4_rgb_train_eval_smoke_render16_2f.json`
- `research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_block4_rgb_train_eval_autograd_repeat12_render32_2_4_8_16.json`
- `research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_vs_record_edit_block4_current_process_train_eval_repeat12_render32_2_4_8_16.json`

Standalone block4 RGB train/eval repeat:

- 2->16 frame count: `8x`
- total-step scale: `1.15x`
- render scale: `0.98x`
- backward scale: `1.33x`
- 16f total/render/backward: `6.11 / 1.85 / 3.54 ms`
- 16f heldout PSNR: `14.34`
- 16f block4 storage: `0.395x` endpoint CSR, `0.0438x` full segment CSR

Three-way same-process repeat:

- block4 vs endpoint-run at 16f: `0.821x` total, `0.855x` render, `0.818x`
  backward
- block4 vs original edit at 16f: `0.663x` total, `0.582x` render, `0.684x`
  backward
- original edit vs endpoint-run remains slower at 16f: `1.237x` total
- PSNR matches endpoint-run/edit to tolerance

This is the first evidence that the block4 idea helps an isolated train/eval
step, not just a forward microprobe. Remaining scope boundary: no main-trainer
integration, no geometry gradients, no dedicated block4 VJP, and no matched
STAR-UVT competitive claim.

## Dedicated Block4 VJP Correction

I then implemented the dedicated block4 RGB-only VJP instead of borrowing the
full endpoint-record edit VJP. The corrected sidecar artifacts are:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_replay_block4_vjp_render32_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_block4_rgb_train_eval_autograd_block4vjp_repeat12_render32_2_4_8_16.json
research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_status_summary.json
research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_status_summary_verifier.json
```

Raw probe:

- block4 forward still matches endpoint-run: max abs error `8.94e-7`
- block4 RGB-only VJP matches full zero-alpha/depth VJP: max rel error
  `2.93e-6`
- block4 RGB-only VJP matches existing edit RGB-only VJP: max rel error
  `2.86e-6`
- 16f block4 forward `1.72 ms` versus endpoint-run `1.69 ms` and original edit
  `2.91 ms`
- 16f block4 RGB-only VJP `2.84 ms` versus edit RGB-only VJP `3.30 ms`
- 16f block4 storage remains `0.395x` endpoint CSR and `0.0438x` full segment
  CSR

Corrected block4 train/eval with the dedicated VJP:

- 2->16 frame count: `8x`
- total-step scale: `3.01x`
- render scale: `4.11x`
- backward scale: `2.73x`
- 16f total/render/backward: `75.18 / 30.63 / 32.92 ms`
- 16f heldout PSNR: `14.34`
- storage remains `0.0438x` full segment CSR

Interpretation correction: block4 is storage-sublinear and now has a correct
dedicated RGB VJP. Runtime is also sublinear in the corrected train/eval rerun,
but the path is not speed competitive. The previous borrowed-VJP paired
numbers were useful exploration evidence but should not be treated as the
current block4 claim. Current status: no main-trainer integration, no geometry
gradient path, and no STAR-UVT competitive claim.

## Block-Size Sweep Hook

Generalized the block-anchor path so the same fork can test block sizes other
than four without copying another shader. The Metal kernels already read
`block_size` and `block_count` from `config_i32`; the artificial restrictions
were in Python/C++ validation and in the block tape packer. I removed those
checks, added `--edit-block-size` to the replay probe and train/eval harness,
and routed manual-VJP block tapes through the dedicated block VJP before the
generic edit VJP.

New correctness smokes:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_replay_block2_vjp_smoke_render16_2f.json
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_replay_block8_vjp_smoke_render16_2f.json
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_block4_rgb_train_eval_manualvjp_block4vjp_smoke_render16_2f.json
```

Block2 smoke:

- forward max abs error versus endpoint-run: `5.96e-8`
- block RGB-only VJP max rel error versus full zero-alpha/depth VJP: `2.97e-7`
- block RGB-only VJP max rel error versus edit RGB-only VJP: `3.56e-7`

Block8 smoke:

- forward max abs error versus endpoint-run: `5.96e-8`
- block RGB-only VJP max rel error versus full zero-alpha/depth VJP: `4.16e-7`
- block RGB-only VJP max rel error versus edit RGB-only VJP: `2.97e-7`

These were single-frame-count render16/2f smokes, so their top-level status is
still negative under the full scaling/storage acceptance checks. They only prove
that non-4 block sizes are wired through Python, C++, Metal dispatch, forward,
and RGB-only VJP. Larger 2/4/8/16 sweeps are still needed before treating block
size as a speed result.

I then ran reduced render16 2/4/8 sweeps for block sizes 2/4/8:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_replay_block2_vjp_smoke_render16_2_4_8.json
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_replay_block4_vjp_smoke_render16_2_4_8.json
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_replay_block8_vjp_smoke_render16_2_4_8.json
```

All three pass correctness and storage-scale checks over the reduced 2/4/8
range:

- block2 storage scale `2.20x` for `4x` frames; 8f storage `0.691x` endpoint
  CSR and `0.0762x` full segment CSR
- block4 storage scale `1.47x` for `4x` frames; 8f storage `0.461x` endpoint
  CSR and `0.0509x` full segment CSR
- block8 storage scale `1.43x` for `4x` frames; 8f storage `0.450x` endpoint
  CSR and `0.0496x` full segment CSR

The speed screen is negative/noisy rather than promising:

- 8f endpoint forward in the three runs: `4.06 / 4.27 / 5.64 ms`
- 8f block forward for block2/block4/block8: `10.67 / 11.72 / 9.11 ms`
- 8f original edit forward in those runs: `21.96 / 8.97 / 5.60 ms`

Conclusion: block size is now a valid experimental knob, and the expected
storage tradeoff appears. But none of the tested block sizes gives a practical
speed win in the reduced smoke. This keeps the "not STAR competitive" status
unchanged.

## Coefficient-Cached Block Replay

Implemented a forward-only coefficient-cached variant:

- Metal kernel: `wf2_endpoint_record_edit_block_coeff_rgba_depth_replay_tensor`
- Python wrapper: `endpoint_record_edit_block_coeff_rgba_depth_replay`
- Probe artifacts:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_replay_block_coeff_smoke_render16_2f.json
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_replay_block_coeff_smoke_render16_2_4_8.json
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_replay_block_coeff_render16_16f.json
```

This keeps owner/cut-id block edit topology, but precomputes four float depth
coefficients per `(track, boundary)`. The shader then evaluates cut depth as a
small rational function of frame time instead of recomputing the plane/ray dot
products from boundary and ray payloads per segment.

Correctness:

- render16 2/4/8 max block-coeff forward error versus endpoint-run:
  `1.79e-7`
- render16 16f max block-coeff forward error versus endpoint-run:
  `1.37e-6`

Speed screen:

- render16 8f: block-coeff forward `3.32 ms`, endpoint forward `7.85 ms`,
  block4 boundary replay `4.60 ms`, original edit `12.70 ms`
- render16 16f: block-coeff forward `4.77 ms`, endpoint forward `11.61 ms`,
  block4 boundary replay `9.88 ms`, original edit `68.86 ms`

Storage tradeoff:

- render16 8f block+coeff storage: `3.06x` endpoint CSR, `0.337x` full segment
  CSR
- render16 16f block+coeff storage: `1.68x` endpoint CSR, `0.185x` full segment
  CSR

Conclusion from this first forward-only pass: this was the first sidecar in the
current pass that looked clearly speed-positive against endpoint-run in the raw
forward probe, but it bought that with a large frame-independent coefficient
table. The follow-up below supersedes the "not yet VJP/autograd" part.

## Coefficient VJP and One-Step Autograd Follow-Up

Added a coefficient-cached RGB-only VJP sidecar and a one-step autograd
train/eval smoke.

New shader/API surface:

- Metal kernel:
  `wf2_endpoint_record_edit_block_coeff_vjp_direct_atomic_rgb_only_tensor`
- C++/Python op:
  `endpoint_record_edit_block_coeff_vjp_direct_atomic_rgb_only`
- Autograd wrapper:
  `endpoint_record_edit_block_coeff_rgba_depth_autograd`
- Train/eval tape mode:
  `--tape-mode endpoint-record-edit-block-coeff`

Refreshed artifacts:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_replay_block_coeff_smoke_render16_2_4_8.json
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_replay_block_coeff_render16_16f.json
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_block_coeff_rgb_train_eval_autograd_smoke_render16_16f.json
research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_status_summary.json
research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_status_summary_verifier.json
```

Clean 16f render16 replay numbers after rerunning without parallel MPS
contention:

- endpoint forward: `6.64 ms`
- original edit forward: `13.82 ms`
- block4 boundary forward: `4.70 ms`
- coefficient forward: `5.08 ms`
- endpoint VJP: `11.50 ms`
- edit RGB-only VJP: `10.26 ms`
- block4 RGB-only VJP: `8.69 ms`
- coefficient RGB-only VJP: `17.16 ms`

Correctness:

- coefficient forward max abs error versus endpoint-run: `1.37e-6`
- coefficient RGB-only VJP relative error versus full VJP with zero alpha/depth:
  `3.63e-6`
- coefficient RGB-only VJP relative error versus edit RGB-only VJP: `2.95e-6`
- coefficient RGB-only VJP relative error versus block4 RGB-only VJP: `2.27e-6`

Trainability smokes:

- 16f one-step status: `ok`
- gradients nonzero and parameters updated
- final train PSNR: `12.64`
- final heldout PSNR: `14.98`
- one-step total/render/backward: `743.4 / 142.2 / 97.6 ms`
- selected coefficient tape storage: `0.185x` full segment CSR
- block4 base storage: `0.0423x` full segment CSR and `0.384x` endpoint CSR

The first 2/4/8/16 run was progress-blind and got stopped after more than four
minutes with no JSON. I patched the harness to print per-frame progress and
write an optional partial JSON, then reran the same one-step coefficient-cached
train/eval scale:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --frame-counts 2,4,8,16 \
  --render-size 16 \
  --site-count 12 \
  --steps 1 \
  --warmup-steps 0 \
  --optimizer-mode autograd \
  --tape-mode endpoint-record-edit-block-coeff \
  --edit-block-size 4 \
  --partial-out-json research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_block_coeff_rgb_train_eval_autograd_smoke_render16_2_4_8_16.partial.json \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_block_coeff_rgb_train_eval_autograd_smoke_render16_2_4_8_16.json
```

The one-step rerun completed with status `ok`.

- total step timings for 2/4/8/16f: `500.9 / 145.2 / 81.6 / 141.7 ms`
- render timings: `171.8 / 14.7 / 13.3 / 9.95 ms`
- backward timings: `56.5 / 72.7 / 21.8 / 61.5 ms`
- measured total scale 2f->16f: `0.283x` for `8x` frames
- render scale 2f->16f: `0.058x`
- backward scale 2f->16f: `1.09x`
- selected coefficient tape storage scale: `1.16x`
- endpoint-record edit op scale: `1.92x`
- 16f heldout PSNR: `14.98`

The scale booleans are green, but the timing shape is one-step/warmup dominated
and should not be used as a stable speed benchmark. It is useful trainability
and rough scaling evidence only.

Then ran the warmed 5-step, 1-warmup version:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --frame-counts 2,4,8,16 \
  --render-size 16 \
  --site-count 12 \
  --steps 5 \
  --warmup-steps 1 \
  --optimizer-mode autograd \
  --tape-mode endpoint-record-edit-block-coeff \
  --edit-block-size 4 \
  --partial-out-json research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_block_coeff_rgb_train_eval_autograd_warm5_render16_2_4_8_16.partial.json \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_block_coeff_rgb_train_eval_autograd_warm5_render16_2_4_8_16.json
```

That completed with status `ok` and supersedes the one-step scale artifact for
current status-summary purposes.

- total step timings for 2/4/8/16f: `29.55 / 72.83 / 19.29 / 38.18 ms`
- render timings: `10.66 / 13.74 / 4.65 / 15.20 ms`
- backward timings: `13.21 / 46.60 / 10.06 / 15.31 ms`
- measured total scale 2f->16f: `1.29x` for `8x` frames
- render scale 2f->16f: `1.43x`
- backward scale 2f->16f: `1.16x`
- selected coefficient tape storage scale: `1.16x`
- endpoint-record edit op scale: `1.92x`
- 16f heldout PSNR: `15.12`

This is a real warmed multi-step smoke, and the saved scale booleans are green.
It is still noisy at render16: the 4f row is slower than 8f, so do not treat it
as a stable benchmark or a STAR-competitive result.

Updated interpretation: coefficient caching is now more than forward-only. It
has correct RGB-only VJP and a green warmed 2/4/8/16 autograd smoke. But the
clean 16f replay no longer supports the earlier "beats block4" line:
coefficient forward beats endpoint-run and original edit, but is slower than
block4 in the clean 16f artifact. The coefficient VJP/train path still needs a
longer, less noisy benchmark and overhead work before it can be treated as a
practical speed path. So this is a useful STAR-port idea for hoisting cut-depth
math out of the hot loop, not a current STAR-UVT competitive World Foam result.

Then ran the paired same-process endpoint-run/edit/block4/block-coeff train/eval
comparison so the speed sign is not inferred across separate MPS runs:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/compare_endpoint_run_record_edit_train_eval.py \
  --frame-counts 2,4,8,16 \
  --render-size 16 \
  --site-count 12 \
  --steps 5 \
  --warmup-steps 1 \
  --include-block4 \
  --include-block-coeff \
  --edit-block-size 4 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_vs_record_edit_block4_blockcoeff_current_process_train_eval_warm5_render16_2_4_8_16.json
```

That completed with status `ok`. At 16f, all modes match heldout PSNR
(`15.1223`). Total/render/backward timings:

- endpoint-run: `69.96 / 21.54 / 27.83 ms`
- raw edit: `95.07 / 52.58 / 31.32 ms`
- block4: `22.52 / 4.74 / 13.58 ms`
- block-coeff: `26.89 / 8.76 / 12.46 ms`

Ratios: raw edit is slower than endpoint-run (`1.36x` total), block4 is faster
than endpoint-run (`0.322x` total), block-coeff is faster than endpoint-run
(`0.384x` total), and block-coeff is slower than block4 (`1.19x` total). This
changes the practical read: the block-anchored variants are the speed-positive
ones in this smoke, not raw edit. Block-coeff is a good STAR-port idea for
hoisting depth coefficient math out of the hot loop, but it is not cleaner than
block4 yet because storage is heavier (`0.185x` full segment CSR but `1.68x`
endpoint CSR) and the run is still tiny/render16/MPS-noisy.

Promoted that paired block-coeff artifact into:

```text
research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_status_summary.json
research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_status_summary_verifier.json
```

The summary verifier is green and explicitly preserves the scope boundary:
World Foam has sublinear measured pieces and speed-positive block sidecars, but
the completion and STAR-competitive claims remain false.

Follow-up correctness fix: the coefficient cut-depth helper in the Metal shader
was missing the near/far bounds check that the boundary/ray cut-depth helper
already had. This meant a coefficient-replayed cut outside the render depth
range could contribute a segment that block4 would reject. Patched
`wf2_endpoint_record_coeff_cut_depth` to require
`near_depth <= out_depth <= far_depth`.

Added a focused MPS regression:

```text
test_probe_endpoint_record_edit_replay.EndpointRecordEditReplayTests.test_block_coeff_rejects_cut_depths_outside_near_far_like_block4
```

The test builds a one-track one-frame row whose boundary lands beyond `far`.
Block4 rejects the segment; block-coeff must now match block4 and produce zero
RGB.

Validation after the fix:

- rebuilt `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0`
- focused MPS regression: `OK`
- full world_foam_lane2 unittest discovery: `21 tests`, `OK`
- refreshed block-coeff 2/4/8 replay artifact: status `ok`
- refreshed sequential 16f replay artifact: Metal forward/VJP correctness
  checks green; overall status remains `negative` only because a single
  frame-count artifact cannot satisfy scale booleans
- regenerated mixed status summary and verifier: verifier `ok`

Sequential 16f render16 replay after the fix:

- endpoint forward: `6.26 ms`
- raw edit forward: `6.31 ms`
- block4 forward: `4.74 ms`
- block-coeff forward: `6.00 ms`
- block-coeff RGB-only VJP: `3.06 ms`
- block-coeff forward max abs error versus endpoint-run: `1.37e-6`
- block-coeff RGB-only VJP rel error versus full zero alpha/depth: `3.75e-6`
- block-coeff storage remains `1.68x` endpoint CSR and `0.185x` full segment CSR

This makes the coefficient path more contract-correct and keeps it slightly
forward-positive against endpoint-run/edit in the refreshed 16f sidecar, but
still not cleaner than block4 and still not a STAR-UVT competitive result.

A larger render32 paired comparison that was already running in the background
finished and changed the practical read:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_vs_record_edit_block4_blockcoeff_current_process_train_eval_warm5_render32_2_4_8_16.json
```

At 16f, all modes match heldout PSNR (`13.2739`). Total/render/backward:

- endpoint-run: `13.04 / 3.79 / 5.91 ms`
- raw edit: `11.28 / 3.54 / 4.90 ms`
- block4: `9.09 / 2.43 / 4.97 ms`
- block-coeff: `8.06 / 2.47 / 3.63 ms`

Ratios at 16f:

- raw edit vs endpoint-run: `0.865x`
- block4 vs endpoint-run: `0.698x`
- block-coeff vs endpoint-run: `0.618x`
- block-coeff vs block4: `0.886x`

This render32 smoke is the best current speed sign for block-coeff: unlike the
render16 replay, block-coeff wins total step over block4 in the paired train/eval
run. I promoted this artifact to the default paired block-coeff status summary
input and updated the verifier to preserve that current sign. Scope stays the
same: smoke-scale MPS, not stable benchmark, not main trainer, not matched STAR.

Then ran a 16f-only render32 repeat with `20` measured steps and `5` warmup
steps to check whether the signs survive a longer sample:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/compare_endpoint_run_record_edit_train_eval.py \
  --frame-counts 16 \
  --render-size 32 \
  --site-count 12 \
  --steps 20 \
  --warmup-steps 5 \
  --include-block4 \
  --include-block-coeff \
  --edit-block-size 4 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_vs_record_edit_block4_blockcoeff_current_process_train_eval_repeat20_render32_16f.json
```

The repeat completed with status `ok`; all modes match heldout PSNR
(`14.5922`). Total/render/backward:

- endpoint-run: `9.42 / 2.91 / 4.57 ms`
- raw edit: `11.31 / 4.20 / 5.62 ms`
- block4: `9.19 / 2.69 / 4.66 ms`
- block-coeff: `7.48 / 2.20 / 3.78 ms`

The important correction is that raw edit is not repeatably faster: it is
`1.20x` endpoint-run in this longer repeat. Block4 remains just faster
(`0.976x` endpoint-run), and block-coeff remains clearly faster (`0.794x`
endpoint-run, `0.814x` block4). I updated the summary/verifier language to
preserve the robust claim: block variants are speed-positive; raw edit is a
storage-first path with noisy speed sign.

Harness hardening after the long render32 run: `compare_endpoint_run_record_edit_train_eval.py`
now accepts `--partial-out-json`. It prints mode-level start/done messages,
writes a top-level partial after each completed mode, and passes a mode-specific
row partial into `run_train_eval`. Added a fast unit test that monkeypatches the
heavy train/eval runner and verifies the partial JSON contract and promoted
ratio signs.

I also ran a cheap real all-mode partial-contract smoke at render16/2f/1-step:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_vs_record_edit_block4_blockcoeff_partial_contract_smoke_render16_2f.partial.json
```

It successfully wrote the top-level and per-mode partials, but the final compare
exited nonzero because the 2f coefficient table is larger than full CSR
(`selected_tape_storage_below_full_at_max_frame=false`). That is expected for
the tiny 2f coefficient-cache path and is not a promoted speed/quality gate.

### 2026-05-15 16f repeat20 block-coeff practical speed sign

The 16f-only render32 paired run finished after the summary/doc pass:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_vs_record_edit_block4_blockcoeff_current_process_train_eval_repeat20_render32_16f.json
```

It is a longer 20-step check of the 16f speed sign, not a frame-count scaling
artifact. Heldout PSNR matches across all four modes at `14.5922`. Timings:

- endpoint-run: `9.42 / 2.91 / 4.57 ms` total/render/backward
- raw endpoint-record edit: `11.31 / 4.20 / 5.62 ms`
- block4 endpoint-record edit: `9.19 / 2.69 / 4.66 ms`
- block-coeff endpoint-record edit: `7.48 / 2.20 / 3.78 ms`

Ratios: raw edit is slower than endpoint-run (`1.201x` total), block4 is
roughly break-even/slightly faster (`0.976x`), and block-coeff remains
speed-positive (`0.794x` endpoint-run total and `0.814x` block4 total). This
supports the practical read that the block-anchored coefficient path is the
best current World Foam speed sidecar, while raw edit is mostly a compact
storage path and still has noisy speed sign. This still does not make a STAR
UVT competitive claim: the run is smoke-scale MPS, single frame count, fixed
geometry/site-RGBA, not main trainer integrated, and not matched for
quality/capacity.

### 2026-05-15 matched-cadence STAR direct-atomic speed reference

Ran a STAR-only fixed-step speed reference at the same small 32px frame-count
grid and the same `20` measured step / `5` warmup cadence as the promoted
coefficient-cached World Foam sweep:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/fixed_step_speed_compare.py \
  --cases 32x2,32x4,32x8,32x16 \
  --steps 20 \
  --warmup-steps 5 \
  --skip-world-foam \
  --skip-dynamic \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_fixed_step_speed_compare_star_directatomic_20step_32px_2_4_8_16.json \
  --input-dir research_experiments/world_foam_lane2/results/fixed_step_speed_compare_star_directatomic_20step_32px_inputs
```

Result status is `ok`. STAR direct-atomic mean step/render timings:

- 2f: `26.76 / 2.21 ms`
- 4f: `22.14 / 1.82 ms`
- 8f: `22.79 / 2.59 ms`
- 16f: `32.83 / 4.45 ms`

Step time scales `1.227x` from 2f to 16f while frame count scales `8x`; render
time scales `2.009x`. This keeps STAR runtime-sublinear under the same timing
cadence as the coefficient sidecar. The new status summary compares the 16f
STAR reference to the coefficient-cached World Foam sidecar (`6.84 ms`, ratio
`0.208x`) but keeps the scope explicit: this is still tiny 32px timing only,
not matched quality/capacity and not proof of STAR-UVT competitiveness.

### 2026-05-15 repeat20 render32 coefficient-cached frame-count sweep promoted

Promoted the standalone coefficient-cached RGB train/eval scale sweep:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_block_coeff_rgb_train_eval_autograd_repeat20_render32_2_4_8_16.json
```

This run uses the coeff-cached forward and RGB-only VJP through the autograd
wrapper at render32 with `20` measured steps and `5` warmup steps across
2/4/8/16 frames. It completed with status `ok` and all acceptance checks true.
Per-frame total/render/backward timings:

- 2f: `12.23 / 3.60 / 5.92 ms`
- 4f: `8.15 / 2.45 / 4.09 ms`
- 8f: `6.39 / 1.91 / 3.44 ms`
- 16f: `6.84 / 1.84 / 3.89 ms`

The measured 2f-to-16f scales are total `0.559x`, render `0.513x`, backward
`0.657x`, selected tape storage `1.170x`, and endpoint edit ops `1.871x` for
an `8x` frame-count increase. 16f heldout PSNR is `14.5922`.

Important correction to the storage read: coefficient-cache storage is not
below full CSR at every frame count. It is `1.243x` full CSR at 2f because the
fixed coefficient table dominates, then amortizes below full by 4f and reaches
`0.181x` full CSR at 16f. I updated the status summary/verifier to preserve
that nuance. This is now the stronger standalone frame-count scaling artifact
for the coefficient path, but it is still smoke-scale MPS, fixed
geometry/site-RGBA, not main-trainer integrated, and not a STAR-UVT
competitive quality/capacity result.

### 2026-05-15 block-coeff16 f16 storage sidecar is a speed negative

A `block-coeff16` path was present in the forked shader and Python wrappers,
using float16 coefficients with manual-VJP train/eval only. The focused unit
test `test_block_coeff16_matches_f32_replay_and_rgb_vjp_on_simple_row` checks
that f16 replay and RGB-only VJP match the f32 coefficient path on a small row
within `2e-4`.

The paired manual-VJP smoke completed:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_vs_record_edit_block4_blockcoeff16_manualvjp_smoke_render32_16f.json
```

16f total/render/backward:

- endpoint-run: `6.40 / 2.46 / 2.65 ms`
- raw edit: `5.73 / 2.86 / 2.26 ms`
- block4: `6.13 / 1.96 / 3.27 ms`
- f32 block-coeff: `4.55 / 1.82 / 2.05 ms`
- f16 block-coeff16: `7.67 / 2.15 / 3.84 ms`

PSNR is matched within the f16 tolerance and storage improves versus the f32
coefficient table (`0.111x` full CSR for f16 versus `0.181x` for f32), but the
speed sign is negative: block-coeff16 is `1.199x` endpoint-run and `1.688x`
f32 block-coeff total step. The status summary now records this as a negative
sidecar rather than a promoted speed path.

### 2026-05-15 long paired 2/4/8/16 repeat is negative for block-coeff

Ran the longer same-process paired sweep that had been missing from the status:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/compare_endpoint_run_record_edit_train_eval.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --site-count 12 \
  --steps 20 \
  --warmup-steps 5 \
  --optimizer-mode autograd \
  --include-block4 \
  --include-block-coeff \
  --edit-block-size 4 \
  --partial-out-json research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_vs_record_edit_block4_blockcoeff_current_process_train_eval_repeat20_render32_2_4_8_16.partial.json \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_vs_record_edit_block4_blockcoeff_current_process_train_eval_repeat20_render32_2_4_8_16.json
```

The artifact was written but the command exited nonzero because the block-coeff
mode failed its speed acceptance:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_vs_record_edit_block4_blockcoeff_current_process_train_eval_repeat20_render32_2_4_8_16.json
```

16f total/render/backward:

- endpoint-run: `43.11 / 15.82 / 18.23 ms`
- raw edit: `25.44 / 10.71 / 11.29 ms`
- block4: `11.08 / 3.04 / 5.76 ms`
- block-coeff: `71.50 / 17.85 / 27.85 ms`

Ratios at 16f: raw edit is `0.590x` endpoint-run, block4 is `0.257x`
endpoint-run, but block-coeff is `1.659x` endpoint-run and `6.452x` block4.
PSNR remains matched at about `14.5922`.

Interpretation: this closes the "run a longer paired 2/4/8/16" evidence gap
with a negative result. The standalone coefficient-cached sweep still proves
the coeff path can have sublinear saved total/render/backward scaling, but the
long paired artifact says not to call f32 block-coeff a stable practical speed
winner yet. In this run, block4 is the practical 16f speed winner, and
block-coeff needs investigation before promotion.

### 2026-05-15 repeat-loaded 16/32 smoke also negative for practical speed

An already-running repeat-loaded render16 smoke eventually completed:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_vs_record_edit_blockcoeff_repeat_loaded_warm1_render16_16_32.json
```

It uses `--repeat-loaded-frames`, frame counts `16,32`, render16, 4 sites, and
only `1` measured step / `1` warmup step, so it is not a promoted benchmark.
The artifact status is `failed`: endpoint-record edit failed sublinear timing
acceptance, while block-coeff stayed sublinear across 16->32 but was still
slower than endpoint-run at the 16f summary point.

16f total/render/backward:

- endpoint-run: `8.33 / 2.49 / 3.79 ms`
- raw edit: `34.43 / 12.57 / 15.11 ms`
- block-coeff: `27.78 / 7.70 / 13.44 ms`

Frame-count scale 16->32: endpoint-run `1.98x`, raw edit `2.90x` (failed),
block-coeff `1.25x` (sublinear). This reinforces the current conservative
read: coefficient caching can make frame scaling sublinear, but the present
replay path is not yet practically faster than endpoint-run in paired tests.

### 2026-05-15 coeff16 storage-accounting fix

Found and fixed a harness bug in `train_eval_owner_run_tape.py`: the
`endpoint-record-edit-block-coeff16` selected-storage branch was unreachable,
so the f16 coefficient sidecar could be reported as endpoint-run storage. Added
`_selected_tape_storage_bytes(...)` and a unit test that checks f32 coeff
storage counts 4-byte coefficients while coeff16 counts the same sidecar at
2 bytes per coefficient.

Focused gates after the fix:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  research_experiments/world_foam_lane2/test_compare_endpoint_run_record_edit_train_eval.py \
  research_experiments/world_foam_lane2/summarize_fused_slab_mixed_results.py \
  research_experiments/world_foam_lane2/verify_fused_slab_status_summary.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s research_experiments/world_foam_lane2 -q

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/summarize_fused_slab_mixed_results.py \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_status_summary.json

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_fused_slab_status_summary.py \
  --summary-json research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_status_summary.json \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_status_summary_verifier.json
```

The storage-fix runtime smoke artifact is:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_block_coeff16_manualvjp_storagefix_smoke_render16_16f.json
```

It is a 16f render16 manual-VJP coeff16-only smoke, not a speed benchmark.
It passed and now reports selected f16 storage `0.1137x` full CSR, endpoint-run
storage `0.1103x`, and block4 storage `0.0423x`. This confirms the selected
tape is no longer being counted as endpoint-run storage. The speed read remains
negative/unstable; do not promote coeff16.

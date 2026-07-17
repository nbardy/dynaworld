# Gate 0 CPU And Metal Scaffold Status

Date: 2026-05-12 00:28 +07

Scope: status note only. I did not edit existing README, scripts, shaders, or
TODO files.

## Commands Run

```bash
PYTHONDONTWRITEBYTECODE=1 python3 dynaworld/research_experiments/world_foam_lane2/gate0_beam_toy.py --json
PYTHONDONTWRITEBYTECODE=1 python3 dynaworld/research_experiments/world_foam_lane2/gate0_event_sharing_benchmark.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s dynaworld/research_experiments/world_foam_lane2
rg -n "Metal|metal|shader|kernel|mlx|mps|fast_mac|gpu" dynaworld/research_experiments/world_foam_lane2
```

## CPU Toy Result

The main CPU toy is runnable and passes the local Gate 0 toy checks.

- Scenario: `orthographic_2d_time_power_cells`
- Sites / boundaries: `5` / `10`
- Samples: `u_samples=17`, `time_slabs=1`, `near=0.25`, `far=3.0`,
  `camera_velocity_x=0.35`
- Frame rows:
  - `2f`: per-frame events `281`, beam-slab events `149`, ratio `0.530249`,
    missing sample events `0`
  - `4f`: per-frame events `565`, beam-slab events `149`, ratio `0.263717`,
    missing sample events `0`
  - `8f`: per-frame events `1133`, beam-slab events `149`, ratio `0.131509`,
    missing sample events `0`
  - `16f`: per-frame events `2266`, beam-slab events `149`, ratio `0.065755`,
    missing sample events `0`
- Growth: per-frame event growth `8.064057`, beam event growth `1.0`,
  `sublinear_event_growth=true`
- Benchmark acceptance flags: `all_rows_zero_missing=true`,
  `sublinear_event_growth=true`
- Auxiliary moving-disk unittest: `Ran 3 tests in 0.001s`, `OK`

Interpretation: the current CPU toy supports the narrow Gate 0 event-sharing
claim. One shared screen-time slab covers all sampled per-frame boundary events
with no missing sample events, while the per-frame event count scales with frame
count.

## Paired STAR / Dynamic Smoke Update

The current paired routing artifact is:

```text
dynaworld/research_experiments/world_foam_lane2/results/gate0_paired_with_star_dynamic_smoke.json
```

It reports `status=ok` and includes:

- World Foam CPU Gate 0 event sharing at the 16-frame summary:
  `per_frame_event_sum=2266`, `beam_slab_event_sum=149`,
  `event_sharing_ratio=0.06575463371579876`, `missing_sample_events=0`.
- STAR-UVT smoke context from
  `dynaworld/third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/benchmarks/results/smoke_tile_load_reg_mps_64_4f/comparison_report.json`:
  `uvt_tile_tube_pairs=3754`, `summed_per_frame_tile_splat_pairs=6489`,
  `pair_ratio=0.5785174911388503`.
- Dynamic-splat smoke context from the same comparison report's
  `free_dynamic_splats` section: `frames=4`, `splat_count=64`,
  `steps=5`, `heldout_eval_psnr=3.9107308387756348`.

This paired JSON is a normalization/routing artifact. The `comparison_unit`
field is different across rows, so it is not an apples-to-apples renderer
quality or speed comparison.

## Backward Status

`gate0_event_sharing_benchmark.py` still reports
`event_replay_accounting_only_no_gradients`. That row only says the event list
would be replayed rather than enumerated per frame again.

The lane now also has a separate CPU Gate 0.5 gradient reference:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 dynaworld/research_experiments/world_foam_lane2/gate0_shared_forward_backward.py \
  --out-json dynaworld/research_experiments/world_foam_lane2/results/gate0_shared_forward_backward.json
```

The current 16-frame Gate 0.5 row has:

- `max_output_abs_error=0.0`
- `signal_gradient_max_abs_error=0.0`
- `finite_difference_max_abs_error=1.3797318842989625e-10`
- direct forward+backward boundary scans `2720 + 2720`
- shared forward+backward boundary scans `170 + 0`
- shared forward+backward scan ratio `0.03125`

Scope boundary: this is a CPU site-signal-gradient reference through fixed
segments. It is not a Metal backward pass, geometry gradient, topology-gradient
treatment, or trainable image renderer.

## Metal Scaffold Status

The isolated Metal scaffold now lives outside this CPU folder at:

```text
dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_v0/
```

It includes:

- `csrc/shared/world_foam_lane2_types.h`
- `csrc/metal/world_foam_lane2_event_count.metal`
- `csrc/metal/world_foam_lane2_power_boundary.metal`
- `csrc/metal/world_foam_lane2_power_boundary_tensor.metal`
- `csrc/bindings.cpp`
- `csrc/metal/world_foam_lane2_metal.mm`
- `torch_world_foam_lane2/ops.py`
- `tools/static_validate.py`
- `tools/smoke_power_boundary_mps.py`

The power-boundary shader mirrors the Gate 0 CPU toy's 2D+time slab boundary
count. `static_validate.py` now compiles the shared header, compiles and runs a
C++ ABI probe that reproduces the CPU toy slab counts for camera velocities
`0.35` and `0.7`, and attempts Metal compilation when the Xcode Metal compiler
is available.

Current local validation:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_v0/tools/static_validate.py
```

This passed the C++ checks here. Metal compiler validation still skipped because
`xcrun` could not find the `metal` tool on this machine.

The count-only MPS bridge was also built and smoke-tested:

```bash
cd dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_v0
python3 setup.py build_ext --inplace
cd /Users/nicholasbardy/git/gsplats_browser
PYTHONDONTWRITEBYTECODE=1 python3 dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_v0/tools/smoke_power_boundary_mps.py \
  --out-json dynaworld/research_experiments/world_foam_lane2/results/gate0_mps_power_boundary_smoke.json
```

The saved result matches the CPU fixture:

- `camera_velocity_x=0.35`: MPS `149`, CPU expected `149`
- `camera_velocity_x=0.7`: MPS `151`, CPU expected `151`
- invalid denominator rows: `0`

This establishes a real Gate 0 MPS count path for the power-boundary slab
fixture. It still does not establish a renderer, tile compositor, or backward
kernel.

## Explicit Gaps

- No tile-span GPU path, screen-time beam GPU parity fixture, or CPU-vs-Metal
  parity suite beyond the two-velocity power-boundary smoke exists yet.
- The CPU toy is not an image renderer: it does not render color/alpha/depth for
  heldout metrics, and its backward reference is limited to site-signal
  gradients through fixed segments.
- The saved paired JSON now includes STAR-UVT and dynamic-splat smoke rows, but
  those rows are routing context with explicit `comparison_unit` boundaries, not
  matched World Foam quality evidence.
- No 4K, tile-memory, bandwidth, or optimizer-step measurement exists for this
  World Foam Gate 0 path.
- The current toy uses a small hand-authored scene; it still needs harder moving
  camera/support cases before being treated as more than a scaffold gate.

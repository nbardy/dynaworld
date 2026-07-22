# MPS memory-pressure / `kernel_task` incident handoff

## Context and terminal state

- Time: 2026-07-22 approximately 19:17-19:24 KST (`+0900`).
- Host: 24 GiB Apple-silicon Mac, 10 CPU cores, running on battery.
- Trigger: the unified paper matrix launched the full Coffee Martini protocol on MPS.
- Terminal state: operator terminated child PID `37643` and parent PID `37022` with `SIGTERM`. Both exited. Do not assume this matrix run completed; inspect partial artifacts before reuse.
- User-visible symptom: the machine became nearly unresponsive / crash-prone and Activity Monitor showed elevated `kernel_task`.

## Exact workload that was stopped

Parent command:

```text
python research_experiments/paper_runner_suite/run_unified_paper_matrix.py \
  --execute --reuse-existing --require-clean-source \
  --matrix src/train_configs/paper_protocols/world_tubes_submission_matrix_v1.jsonc \
  --out-dir outputs/benchmarks/2026-07-22_world_tubes_submission_matrix_clean_v1 \
  --device mps --wandb-mode offline
```

Active child was `multicam_heldout_compare.py` for:

- protocol: `coffee_martini_full_300f_fixed_512_pixel_matched_v1.jsonc`;
- seed: `17`;
- `--target-size 512`, resolved by the protocol to 384x512;
- `--max-frames 300`, two training cameras and one held-out camera;
- `--max-steps 300`, `--train-seconds 86400`;
- `--uvt-loss-scope paper_batch`, four sampled frames per step;
- `--uvt-init-views all_train --uvt-init-frames all`;
- `--device mps --uvt-render-backend metal_tile`;
- 1,024 UVT tubes and 1,024 dynamic 3DGS splats.

The parent matrix executes runs serially, but one comparison child trains/evaluates UVT and then dynamic 3DGS in the same process. The broader runner subsequently runs WorldFoam.

## Observed facts

Immediately before termination:

- Python PID `37643` showed an approximately 20 GiB `top` memory footprint and was intermittently in uninterruptible kernel wait (`U` / `stuck`). Its RSS reading from `ps` was much smaller because macOS had compressed/swapped most of its footprint; do not interpret that RSS as low pressure.
- Physical memory: about 23 GiB used, only 26 MiB unused.
- Compressor: about 18 GiB.
- CPU: 83.16% system, 11.98% user, 4.85% idle.
- Load average: 17.80 / 11.38 / 8.03.
- Process states: 64 stuck.
- Disk sample during escalation: about 321 MiB/s and 11,157 transfers/s.
- Swap counters were rapidly increasing.
- `pmset -g therm` reported no thermal or performance warning.
- `kernel_task` was a symptom of kernel paging/I/O pressure. It was not killed.

Within seconds after terminating the benchmark:

- system-wide free-memory estimate recovered to 70%;
- about 10 GiB became unused;
- compressor fell to about 2.7 GiB;
- CPU recovered to roughly 46-56% idle;
- stuck processes fell from 64 to 10-12.

This before/after response is strong evidence that the benchmark caused the incident.

## Current model of the failure

Status: **supported at the workload level; unresolved at the allocation-site level**.

The benchmark exhausted unified memory on MPS, causing extreme compression, swap traffic, kernel CPU, and UI stalls. Relevant current behavior:

1. `multicam_heldout_compare.py` calls `load_multicam_video_bundle(..., device=device)`, placing the full train/held-out video bundle on MPS.
2. Both UVT and splat paper trainers keep a `paper_stage_cache` of resized frame tensors and intrinsics.
3. The comparison script retains the bundle, UVT model, training results, evaluations, and checkpoint data while proceeding toward the splat baseline; there is no visible `gc.collect()` / `torch.mps.empty_cache()` boundary between lanes.
4. MPS uses system unified memory. Model tensors, full-video tensors, autograd intermediates, Metal allocations, cached stage tensors, evaluations, and checkpoint states can therefore amplify the raw dataset footprint.

Do **not** yet call this a proven Python leak or blame `paper_stage_cache` alone. A single 2-train-view float32 RGB tensor at 300x384x512 is approximately:

```text
2 * 300 * 3 * 384 * 512 * 4 bytes = 1.32 GiB
```

The held-out view adds about 0.66 GiB before copies, gradients, render buffers, saved evaluations, checkpoints, or allocator caching. The observed approximately 20 GiB footprint requires profiling to attribute accurately.

## Alternative explanations to test

### A. Autograd/render peak within one UVT step

Cheap test: run 1, then 5 steps with the full frame bundle and sample `torch.mps.current_allocated_memory()`, `driver_allocated_memory()`, process footprint, and system compressor before/after forward, loss, backward, and optimizer step.

Support: memory jumps during backward and mostly returns after the step.

### B. Monotonic retained graph or result accumulation

Cheap test: log the same metrics every step for a 20-step 128/256-pixel smoke.

Support: allocated/driver memory increases monotonically after synchronization and cache clearing.

### C. Cross-lane retention / allocator cache

Cheap test: run UVT-only and splat-only as separate processes, then compare their peaks with the combined comparison process.

Support: isolated peaks are safe but the combined process crosses the limit.

### D. Dataset/stage duplication

Cheap test: print tensor shapes, dtype, device, `numel * element_size`, storage pointers, and cache contents immediately after bundle load and after each `splat_stage_payload` / UVT stage payload creation.

Support: resized data owns additional full-video storage rather than aliasing or replacing the original.

## Mandatory guardrails before resuming

1. **Do not immediately relaunch the full matrix on this 24 GiB Mac.** Existing `--reuse-existing` does not make an incomplete comparison child safe.
2. Start with one protocol, one seed, one lane, 8-16 frames, 128 px, and 1-5 steps. Increase one dimension at a time: frames, resolution, steps, then lanes.
3. Add a preflight estimate and runtime memory watchdog. Abort the child gracefully if either process footprint or MPS driver allocation exceeds a conservative budget (initial suggestion: 16 GiB total process footprint on this host), if compressor exceeds 8 GiB, or if system available memory remains below 2 GiB. Tune thresholds from measured smokes.
4. Log at least: step/phase, tensor shapes, process footprint/RSS, `torch.mps.current_allocated_memory()`, `torch.mps.driver_allocated_memory()`, and elapsed time. Record peak values in the run report.
5. Prefer one model lane per subprocess so process exit deterministically releases MPS/Metal allocations. If retaining the combined script, serialize completed UVT outputs, delete models/checkpoints/evaluation frame tensors that are no longer needed, run `gc.collect()`, then `torch.mps.empty_cache()` and verify the driver allocation drops before starting splats.
6. Preserve resumability at lane boundaries. A killed UVT/splat comparison should not require repeating every already-completed lane.
7. Keep PyTorch's MPS allocation safety limit enabled. Do not work around OOM protection by disabling the high-watermark limit.
8. Run paper scale only after a monitored scale ladder shows a stable plateau rather than monotonically increasing memory and leaves several GiB of system headroom.

## First recommended next action

Build a read-only memory instrumentation pass plus a 5-step, 16-frame, 128-pixel MPS smoke for UVT alone. Produce a phase-by-phase allocation table. Do not optimize or relaunch paper scale until that table distinguishes bundle residency, render/backward peak, retained results, and allocator cache.

## Relevant code and artifacts

- `research_experiments/paper_runner_suite/run_unified_paper_matrix.py`: serial matrix loop.
- `research_experiments/paper_runner_suite/run_unified_paper_ablation.py`: comparison subprocess launch and subsequent WorldFoam execution.
- `third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/benchmarks/multicam_heldout_compare.py`: bundle loading, UVT/splat training, stage caches, evaluations.
- `src/train_configs/paper_protocols/coffee_martini_full_300f_fixed_512_pixel_matched_v1.jsonc`: full-scale dimensions and schedule.
- Partial output root: `outputs/benchmarks/2026-07-22_world_tubes_submission_matrix_clean_v1/`.

## Handoff decision

The mathematical/benchmark objective remains valid, but the current local execution envelope is unsafe. Resume only through a measured scale ladder or move the full run to a machine with adequate accelerator memory. Elevated `kernel_task` should be treated as an operating-system pressure alarm; stop the child workload, never `kernel_task`.

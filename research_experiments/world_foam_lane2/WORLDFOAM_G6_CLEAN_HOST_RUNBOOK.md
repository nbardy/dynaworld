# WorldFoam G6 clean-host runbook

This is the executable handoff for the real native memory/work ablation. It is
not a test command and its default dry plan is not evidence.

## Claim and matrix

G6 measures the fixed `S=1024`, `P=512`, `384x512` WorldFoam world at
`F=8/64/300` through:

- 12 shared-adjoint primary rows;
- 9 same-representation sequential frame-replay controls;
- 3 auxiliary fresh restart processes.

That is 21 evidence rows across 24 sequential fresh processes. Acceptance
requires an effective MPS allocator limit no greater than `2 GiB` and every
parent-sampled worker process group to remain below `4 GiB` RSS. The `8 GiB`
available-memory launch guard is safety headroom, not representation state and
not a 32-GB machine requirement.

## Allocation-free plan

From the Dynaworld root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/run_worldfoam_g6_clean_host_bundle.py
```

This reads source/configuration only. It starts no subprocess, imports no
Torch or native extension, creates no MPS allocation, builds nothing, writes
nothing, and emits zero evidence rows. On the current checkout it should say
`static_prebuild_ready_host_unchecked` while retaining
`native_extension_older_than_bound_native_sources` as the expected remediable
pre-build producer blocker.

## Real ablation

Only after committing/freezing the exact G6 source set, on a quiet
Apple-silicon Mac with the repository's CPython-3.11 `.venv` available:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/run_worldfoam_g6_clean_host_bundle.py \
  --execute
```

The bundle fails closed and performs this exact sequence:

1. Apply the non-relaxable disk/RAM/swap/load host guard.
2. Verify the selected interpreter is CPython 3.11 with available MPS.
3. Force-rebuild `world_foam_lane2_fused_slab_v0` with that exact virtual-
   environment Python under a 4-GiB process-group watchdog.
4. Write and independently verify the 133-schema native-build attestation.
5. Import the rebuilt extension without Metal dispatch and require both the
   memory-light and lazy-full-geometry ABI seals.
6. Rerun the allocation-free G6 plan and require zero blockers and exact
   `12 + 9 + 3 = 24` process accounting.
7. Execute all rows. Each evidence worker retains its own 2-GiB MPS ceiling,
   4-GiB process-group watchdog, and fresh-process receipt.
8. Independently verify the 21-row artifact and write the bundle receipt only
   if the artifact is accepted.

The artifact hash-binds every required repository-relative source file, but it
does not prove that those files are Git-tracked.  Do not launch publication
evidence from the present largely untracked source set: first commit the exact
implementation revision so the retained hashes identify recoverable code.

The primary artifact is:

```text
outputs/worldfoam_training_memory_ablation/worldfoam_training_memory_ablation.json
```

The bundle also writes the native import verification, per-phase logs, and a
receipt binding the build attestation and accepted evidence artifact. Existing
outputs fail closed. Use `--overwrite` only for an intentional complete rerun;
the native build is always forced so the receipt cannot bless a retained stale
binary.

If the producer completes but the memory, parity, or scaling gates fail, the
JSON remains a valid negative result. Do not call it memory fit and do not add
it to a promoted baseline table.

## Why B200 is not a substitute

The current G6 claim is Metal/MPS-specific: its native implementation is
Objective-C++ plus Metal, the production adapter accepts only `mps`, the hard
working-set receipt uses PyTorch MPS allocator controls/counters, and the
hardware verifier requires Apple MPS. A B200 cannot validate that claim.

CUDA can host a future analogous ablation only after a real CUDA kernel/runtime
port, a CUDA allocator limit/high-water contract, a CUDA-native ABI
attestation, and a separately labelled acceptance artifact. CPU or B200 runs
of the current source can validate mathematics or portability work, not the
present `<2 GiB MPS / <4 GiB RSS` result.

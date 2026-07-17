# Problem Targets For Local Evolution

This is the ranked list of DynaWorld problems that look suitable for an
AlphaEvolve-style Codex loop.

## Ranking Heuristic

A target is good when it has:

- a small write surface
- a fast evaluator
- hard correctness gates
- a metric that matters to the project
- known negative results the prompt can avoid repeating

## P0 Targets

These are algorithm-evolution targets. Config normalization or result parsing
can be used as toy harnesses, but they are not the reason to build this system.

### 1. STAR UVT Feature RGB-Gradient Handoff

Why it matters:

- STAR UVT feature tubes are already first-class but backward-bound.
- The benchmark-only `fused_first3_sigmoid_mse` row is the clearest positive
  direction: image-space/RGB-gradient handoff beats generic F32 feature-gradient
  atomics in the saved synthetic row.
- The current missing step is generalizing that handoff to the real
  `FeatureToColor` path with learned weights, bias, and parameter gradients.

Microlib:

- `microlibs/star_uvt_feature_rgb_handoff.md`

Good score:

- passes F4/F32 tiny parity
- keeps feature/colorizer gradients nonzero where expected
- zero overflow on the target candidate
- reduces backward ms versus `feature_direct_gradcache`
- does not regress 20-step loss/PSNR versus the current alpha-pruned row

Hard reject:

- skips feature gradients
- only supports fixed first-three-channel sigmoid
- hides overflow fallback
- changes training target or frame count

### 2. Mixed Same-View Plus Heldout Novel-View Scheduler

Why it matters:

- The data contract names this as the next bridge.
- This directly targets the world-token training contract: same model path,
  separated `same_view_recon` and `heldout_view_recon`.
- It is mostly Python and config plumbing, so Codex can make useful bounded
  changes if the evaluator is strict.

Microlib:

- `microlibs/mixed_same_view_novel_scheduler.md`

Good score:

- loader tests pass
- one-step offline smoke exercises both batch kinds
- logs include separate loss keys
- target/input overlap tripwire is explicit
- no third manifest format invented

Hard reject:

- collapses metrics into one loss without per-kind logs
- hides heldout-camera semantics behind vague `batch` fields
- introduces a large base trainer

### 3. Gaussian 512px Promotion Guard

Why it matters:

- The 300-clip static/dynamic Gaussian lane is cache-hot at 256px.
- The 512px promotion produced NaNs, so the multires row is blocked.
- A promotion guard is evaluator-friendly: finite checks, checkpoint behavior,
  controlled short schedule, and diagnostics can be made mechanical.

Microlib:

- `microlibs/gaussian_512_promotion_guard.md`

Good score:

- detects nonfinite tensors before optimizer corruption
- checkpoints immediately before promotion
- emits promotion diagnostics
- short multires smoke reaches promotion without NaN
- config remains JSONC-driven

Hard reject:

- lowers the target by staying at 256px
- masks NaNs without identifying source
- adds env-var fanout for every guard knob

## P1 Targets

### 4. V-JEPA/F32 Multicam Benchmark Contract

Why it matters:

- Current heldout evidence is useful but too small and leakage-prone for broad
  claims.
- Evolution can help generate manifest validators, leakage probes, and
  evaluator scripts.

Microlib:

- `microlibs/vjepa_multicam_benchmark_contract.md`

Good score:

- source/camera-disjoint manifest checks
- pose-error diagnostics
- fisheye-preserving A/B surface
- explicit `BASELINES.md` row fields

Hard reject:

- treats current goodset PSNR as solved camera recovery
- uses heldout RGB features in a way not named by the contract

### 5. WorldFoam Gate4 Owner/Candidate Records

Why it matters:

- Gate4 has many negative local forks and a narrow remaining cost target:
  removing owner/candidate replay or scan work from the warm fused-MSE path.
- Existing verifiers already encode much of the desired behavior.

Microlib:

- `microlibs/worldfoam_gate4_records.md`

Good score:

- full train/eval verifier stays `ok`
- 24-site row remains quality-equivalent
- total/backward scale improves against current owner-run direction
- setup improvements do not fake warm kernel wins

Hard reject:

- promotes standalone VJP probes without full train/eval
- changes PSNR semantics
- ignores known MPS lifetime/order sensitivity

## P2 Targets

### 6. Code-Organization Behavior Helpers

Why it matters:

- The repo has known duplication around RGB composition, validation media, and
  metrics/log cadence.
- This is a lower-risk target for a first Evolver proof if shader work is too
  expensive.

Microlib:

- `microlibs/code_org_helpers.md`

Good score:

- fewer behavior forks
- focused tests pass
- no giant trainer framework
- no unrelated refactor churn

Hard reject:

- line-count reshuffle without behavior coverage
- hidden semantic changes to background, alpha, or F32 colorization

## Targets To Avoid Initially

- Full architecture search over the base world-token model.
- Diffusion-as-loss or post-training objectives without a fast evaluator.
- Paid cloud runs as the inner loop.
- Viewer UI work unless the evaluator can check rendered pixels.
- Anything requiring manual media inspection for every candidate.

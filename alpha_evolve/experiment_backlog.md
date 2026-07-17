# Alpha Evolve Experiment Backlog

Date: 2026-05-20

This backlog is ordered to move from runnable infrastructure to real algorithm
evolution. Do not add more papers before at least Experiment 1 and 2 run.

## Experiment 1 - Offline Selector Sanity Check

Question:
    Does the CodeT-style selector behave differently from largest-cluster or
    random selection on a controlled candidate/probe matrix?

Inputs:
    `alpha_evolve/examples/codet_selector_matrix.json`

Command:

```bash
PYTHONPATH=. uv run python -m alpha_evolve.evolver.cli \
  alpha_evolve/examples/codet_selector_matrix.json
```

Expected:
    The selected candidate should come from the group that passes more probes,
    even if a larger wrong group exists.

Why this matters:
    This catches the AlphaCode largest-cluster trap before we point the system
    at expensive kernels.

## Experiment 2 - Selector Ablation On Real Saved Candidates

Question:
    If we encode several saved STAR/Gate4 variants as candidates, does the
    selector pick the same variant a human would promote?

Inputs:
    Hand-authored candidate matrix from existing loose notes and JSON reports.

Compare:

```text
random visible passer
largest cluster
test-count only
CodeT dual agreement
oracle hidden best
```

Output:
    A selector report plus a short note on ranker gap.

## Experiment 3 - Renderer Backend Selector Evolution

Question:
    Can Codex evolve a workload-shape selector for renderer backend choice that
    beats a hand-written if/else baseline?

Candidate surface:

```text
alpha_evolve/tasks/renderer_backend_selector/
```

Fitness:

```text
correct backend on historical workload rows
rejects known bad routes
small complexity penalty
```

Why this before shader mutation:
    It is algorithmic and useful, but much cheaper than compiling new kernels.

## Experiment 4 - STAR UVT Target-Area Visual VJP Variants

Question:
    Can Codex propose kernel/helper variants that improve the native target-area
    visual VJP route without changing objective semantics?

Candidate surface:

```text
research_experiments/star_uvt_feature_tubes/
third_party/fast-mac-gsplat/variants/star_uvt_v0/
```

Visible gates:

```text
tiny parity fixture
no nonfinite gradients
fixed frame/support semantics
```

Hidden gates:

```text
512px workload timing
short media/quality smoke
no feature/probe regression
```

Reject:

```text
skip-feature-gradient
hidden target/loss changes
benchmark-only standalone VJP without trainer proof
```

## Experiment 5 - Same-View Plus Novel-View Scheduler Policy

Question:
    Can the evolver improve the sampling policy while preserving the data
    contract and separate logs?

Candidate surface:

```text
src/train/multicam_video_data.py
src/train/train_video_token_implicit_dynamic.py
src/train_configs/<smoke>.jsonc
```

Fitness:

```text
1-step smoke hits both modes
10-step smoke remains finite
separate same_view_recon and heldout_view_recon metrics
no target/input leak
```

## Experiment 6 - Generated Probe Toxicity Audit

Question:
    Do generated probes help select candidates, or do they prefer shallow wrong
    behavior?

Method:

```text
generate probes
run reference implementation where available
run known-bad candidates
mark toxic probes
rerun selector with and without toxic probes
```

Metric:

```text
selector hidden success
ranker gap
toxic_probe_rate
```

## Stop Conditions

Pause evolution if:

```text
oracle best-of-k succeeds but selector repeatedly misses
visible false-positive rate is high
generated probes are toxic and unaudited
candidate costs exceed manual iteration cost
hidden gates are being reused as prompt feedback
```

When this happens, improve evaluator/probe design before increasing `k`.

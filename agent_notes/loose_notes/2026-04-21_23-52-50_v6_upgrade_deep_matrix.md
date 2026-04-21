# v6 Upgrade Deep Matrix

## Context

The user asked whether there was any overhanging renderer work and specifically
how v6 compared to the v6 upgrade across different splat scene types. The
existing evidence was only a tiny smoke:

- `128x128`
- `B=1`
- `G=64`
- two distributions
- `warmup=1`, `iters=2`

That smoke was too small to settle whether the v6-upgrade handoff had useful
large-scene behavior. The deeper question was whether the upgrade line becomes
valuable at real sizes, especially 64k projected splats and B=4.

## Run

Command, from `third_party/fast-mac-gsplat`:

```bash
/Users/nicholasbardy/git/gsplats_browser/dynaworld/.venv/bin/python benchmarks/benchmark_full_matrix.py \
  --resolutions 512x512,1024x512,1920x1080,4096x4096 \
  --splats 512,2048,65536 \
  --batch-sizes 1,4 \
  --distributions microbench_uniform_random,sparse_screen,clustered_hot_tiles,layered_depth,overflow_adversarial \
  --renderers v6_direct,v6_auto,v6_upgrade_direct,v6_upgrade_auto \
  --modes forward,forward_backward \
  --warmup 1 \
  --iters 2 \
  --timeout-sec 120 \
  --output-md benchmarks/full_rasterizer_benchmark_v6_upgrade_deep.md \
  --output-jsonl benchmarks/full_rasterizer_benchmark_v6_upgrade_deep.jsonl
```

This is 960 subprocess cells:

```text
4 resolutions * 3 splat counts * 2 batch sizes * 5 distributions * 2 modes * 4 renderers = 960
```

All cells completed:

```text
status: ok = 960
timeout = 0
error = 0
```

## Artifacts

Fast-mac submodule:

```text
docs/v6_upgrade_deep_benchmark_report.md
benchmarks/full_rasterizer_benchmark_v6_upgrade_deep.md
benchmarks/full_rasterizer_benchmark_v6_upgrade_deep.jsonl
```

## Main Result

Winner counts over 240 workload groups:

```text
v6_direct           117
v6_upgrade_direct    67
v6_auto              29
v6_upgrade_auto      27
```

Family view:

```text
local v6      146 wins / 240 = 60.8%
v6-upgrade     94 wins / 240 = 39.2%
```

So the upgrade is not a default replacement, but it is not useless either.

## Direct-vs-Direct

Comparing only `v6_upgrade_direct` against `v6_direct`:

```text
all workloads:        mean +3.8%, median +1.3%, upgrade wins 101/240
forward:              mean +5.6%, median +4.2%, upgrade wins 43/120
forward+backward:     mean +1.9%, median +0.3%, upgrade wins 58/120
512 splats:           mean +5.6%, median +5.6%, upgrade wins 26/80
2048 splats:          mean +5.3%, median +1.5%, upgrade wins 33/80
65536 splats:         mean +0.4%, median -0.5%, upgrade wins 42/80
B=1:                  mean +5.0%, median +2.6%, upgrade wins 51/120
B=4:                  mean +2.5%, median +0.7%, upgrade wins 50/120
```

The upgrade becomes meaningfully more competitive as the workload gets larger.
At 64k splats, the median direct-vs-direct result slightly favors the upgrade.

## Distribution Model

Winner counts by distribution:

```text
microbench_uniform_random:
  v6_direct 19, v6_auto 8, upgrade_direct 10, upgrade_auto 11

sparse_screen:
  v6_direct 29, v6_auto 5, upgrade_direct 12, upgrade_auto 2

clustered_hot_tiles:
  v6_direct 25, v6_auto 5, upgrade_direct 15, upgrade_auto 3

layered_depth:
  v6_direct 22, v6_auto 7, upgrade_direct 16, upgrade_auto 3

overflow_adversarial:
  v6_direct 22, v6_auto 4, upgrade_direct 14, upgrade_auto 8
```

The surprising bit is that `sparse_screen` did not become an automatic
active-policy win. Empty tiles alone are not enough; the active path can still
lose to direct due to scheduling/background overhead.

## 4K / 64K Field Notes

At the headline 4K/64k size:

- B=1 uniform forward: `v6_direct` wins (`11.752 ms` vs upgrade direct `12.065 ms`).
- B=1 uniform forward+backward: `v6_upgrade_direct` wins narrowly (`62.657 ms`).
- B=1 clustered forward and forward+backward: `v6_upgrade_direct` wins.
- B=1 sparse forward+backward: `v6_direct` wins.
- B=1 overflow forward+backward: `v6_auto` wins.
- B=4 sparse forward+backward: `v6_upgrade_direct` wins.
- B=4 clustered forward: `v6_upgrade_auto` wins, but B=4 clustered forward+backward goes back to `v6_direct`.
- B=4 overflow forward+backward: `v6_upgrade_direct` wins (`7555 ms` vs `8928 ms` direct v6).

The pathological overflow numbers are seconds, not milliseconds. They are useful
for stress behavior, not for marketing.

## Current Belief

Confidence: medium-high for the default-policy conclusion; medium for exact
percentages because `warmup=1`/`iters=2` is intentionally practical, not
paper-grade.

The locally evolved v6 branch is still the right default. The v6 upgrade is a
candidate pool of mechanisms, especially for 64k/backward-heavy cases. The next
useful action is a targeted ablation of the upgrade mechanisms in the cells
where it wins, not a wholesale replacement.

## Caveats

- Each benchmark cell runs in its own subprocess. That gives isolation from
  package-name collisions but makes first-use effects and machine noise harder
  to eliminate.
- Forward-only and forward+backward use eval/train paths that can differ.
  A forward-only cell can be slower than the forward component inside a
  forward+backward run in some pathological cases.
- Synthetic distributions are diagnostic. They are not substitutes for real
  Dynaworld projected traces.

## Next Falsification Tests

1. Dump real Dynaworld projected traces and replay them through the same matrix.
2. Add tile statistics to the deep report: active fraction, overflow count,
   total pairs, p95 pairs/tile, chosen active policy.
3. Rerun the subset where v6-upgrade wins at `warmup=3`, `iters=10`.
4. Ablate upgrade mechanisms one by one on the 64k forward+backward winners.
5. Keep v6 direct as default unless a trace-backed benchmark flips the conclusion.

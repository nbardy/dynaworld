# Aug 16th Followup Swarm Audit — Overnight Codex Quota Burn (Refined)

**Date:** 2026-08-16 (refined 2026-08-16T10:00 UTC)  
**Scope:** `dynaworld` submodule of `gsplats_browser` + overnight Codex swarm 2026-08-15  
**Author:** audit via `codex-report` (fork-aware dedupe) + `codex_scrape` (streaming `patch_apply_end`) + direct `~/.codex/sessions` reads  
**Baseline commits:** `dynaworld 026c1306557d / 3e698e8eebca` + parent `gsplats_browser 8b9cb19` — repo committed as-is before refactor.

---

## 1) Exact Details — What Happened, With Citations

### 1a) Goal as written (the prompt that launched the swarm)

Parent thread is `019f7a66-eb3c-73d1-a1aa-0585ba5b362c` (`~/.codex/sessions/2026/07/19/rollout-2026-07-19T21-43-13-019f7a66-eb3c-73d1-a1aa-0585ba5b362c.jsonl`, `cli_version 0.144.2`, `originator vscode`, `cwd /Users/nicholasbardy/git/gsplats_browser/dynaworld`). Its `thread_goal_updated` sequence is:

```
goal: "Read the Codex goal objective file at /Users/nicholasbardy/.codex/attachments/eaece9ef-b221-487e-ac5c-f07dc45a6f91/goal-objective.md before continuing."  # createdAt 1784650214
→ paused (tokensUsed 2127729, time 8512s)
→ active (same objective)
→ updatedAt 1786641160: "work all night run all ablations and produce al fiugres and charts for both papers writ all code and run all needed runs, don't stop until we have ICLR level papers" (tokensUsed 34700535, time 173853s)
→ updatedAt 1786678156: same objective (35163179, 176276s)
→ updatedAt 1786728400: same objective (36184004, 181770s ≈ 50.5h)
```

The attachment `goal-objective.md` (verbatim) sets:

> **"Finish World Tubes in Gauged Camera Space as a renderer/compiler paper, implemented by projective STAR UVT."**
>
> *What remains:* 1) land verified runner cleanly (uncommitted dirty tree), 2) finish evidence schema (LPIPS, peak device memory, checkpoint/storage bytes, compile/forward/backward timing, trace stats), 3) add matrix orchestration `protocol × seed × scene × camera split`, 4) run Coffee Martini evidence (progressive 512 seeds 17/29/43, pixel-matched fixed 512, global shuffle, timing), 5) run central systems comparison `per-frame STAR replay vs compiled projective STAR UVT F=4,8,16,32,64,128`, 6) close theorem table (360°/720° multi-gauge), 7) add public breadth (triplets + 2 Neural3D scenes + D-NeRF), 8) package paper (BASELINES, figures, LaTeX, locked manifests, one repro command).
>
> *Ignore until submission:* browser WebGPU, V-JEPA, Gaussian 300-clip scaling, STAR feature-tube sweeps, etc. *Do not mass-delete `research_experiments/` yet.*

This is the **exact** overnight instruction agents executed against — a "don't stop until ICLR" open-ended goal with no token cap and no file ownership.

### 1b) Swarm topology (cited from `~/.codex/sessions/2026/08/15/*.jsonl` headers)

Streaming `session_meta` on 2026-08-15 shows:

- **153** rollout `*.jsonl` files that day in `~/.codex/sessions/2026/08/15/`.
- **107** are forked (`forked_from_id` non-null), all `forked_from_id = 019f7a66-eb3c-73d1-a1aa-0585ba5b362c` (except 4 from `019ff92a` wave_sim and 20 from `019e2586/019e2b15` ai_trader). So dynaworld's share is **~80 forked threads** in one night.
- **Depth** is 1 for most, but reaches **depth=4** (`Locke /root/scenario_openfoam_blind/openfoam_hostile/r1_identity_io/race_probe parent 01a000cf`).
- **Model:** `model_provider=openai`, `model` field empty in meta (resolved as `gpt-5` by `codex-report`'s `resolveModel(info.model||info.model_name||sessionModel)`), `effort=ultra` implied by `reasoning_output_tokens` presence.

Cited examples (from `python3 -c` dump of `session_meta.payload`):

| rollout (file) | id | nick | agent_path | depth | forked_from |
|---|---|---|---|---|---|
| `rollout-2026-08-15T09-40-52-01a002dd-4bd1-78a0-87ad-f671b16581d5.jsonl` | `01a002dd-4bd1-78a0-87ad-f671b16581d5` | **Laplace the 2nd** | `/root/g6_memory_fit_truth_audit` | 1 | `019f7a66` |
| `rollout-2026-08-15T09-06-50-01a002be-21e9-7a81-a87e-f0d2ee3bdcc4.jsonl` | `01a002be-21e9-7a81-a87e-f0d2ee3bdcc4` | **Sagan the 2nd** | `/root/g4_v2_pilot` | 1 | `019f7a66` |
| `rollout-2026-08-15T08-51-58-01a002b0-8564-7272-b0fb-9d010cc2c469.jsonl` | `01a002b0-8564-7272-b0fb-9d010cc2c469` | **Newton the 2nd** | `/root/heldout_spatial_replay` | 1 | `019f7a66` |
| `rollout-2026-08-15T08-07-16-01a00287-9a0e-7d20-8de5-e4e5ff7642b5.jsonl` | `01a00287-9a0e-7d20-8de5-e4e5ff7642b5` | Lorentz | `/root/completion_audit` | 1 | `019ff92a` (wave_sim, not dynaworld) — excluded from lane counts |
| `rollout-2026-08-15T00-07-21-01a000d0-3a75-73d` | `01a000d0-3a75-73d` | Locke | `/root/scenario_openfoam_blind/openfoam_hostile/r1_identity_io/race_probe` | **4** | `01a000cf` |

Full fork list is in audit appendix (107 rows).

**Key quote from Laplace thread** (`event_msg.payload.type=user_message` is empty; the goal is carried by `thread_goal_updated` and `session_meta.source.subagent.thread_spawn`):

```
session_meta.source.subagent.thread_spawn = {
  parent_thread_id: "019f7a66-eb3c-73d1-a1aa-0585ba5b362c",
  depth: 1,
  agent_path: "/root/g6_memory_fit_truth_audit",
  agent_nickname: "Laplace the 2nd"
}
```

Laplace's `event_msg` counts (direct `Counter` from file): `token_count 2816`, `sub_agent_activity 380`, `agent_message 200`, `patch_apply_end 195`, `agent_reasoning 91`, `context_compacted 24`, `task_complete 11`, `task_started 11`. Representative `agent_reasoning` texts:

> "**Planning detailed research handoff file**", "**Planning detailed master TODO file**", "**Formalizing missing math and audit specifics**", "**Integrating calibration validator patch**"

`response_item` mix: `reasoning 42`, `custom_tool_call 34`, `message 29`.

### 1c) Token burn — cited numbers

Direct `token_count` payload from Laplace (last events before compaction):

```json
{"info": {"total_token_usage": {
  "input_tokens": 1715666797, "cached_input_tokens": 1682369792,
  "output_tokens": 4230407, "reasoning_output_tokens": 1539021,
  "total_tokens": 1719897204
}, "last_token_usage": {
  "input_tokens": 171334, "cached_input_tokens": 7936,
  "output_tokens": 399, "reasoning_output_tokens": 247
}, "model_context_window": 258400}}
```

So Laplace's **cumulative** shows `1.72B total` (≈ `1.715B input` of which `1.682B` cached + `4.2M` output). Naive `last - first` would claim `2.12B` output if counting cumulative without dedupe — the bug we fixed.

Corrected accounting (`codex-report` streaming: `seenKeys = codex:${forkedFromId||sessionId}:${cumulativeTotal}:${input}:${cached}:${output}:${reasoning}`, `uncached = max(0,input-cached)`, `forkCutoff = forkTimestamp+5000ms`, head+tail `800` lines):

- **Naive sum across files:** `141.81B` cumulative.
- **Real Δ deduped:** `6.35B` uncached input delta (global `seenKeys` still pending across files at first audit → residual `5–15%` high).
- **Per-thread Real Δ:** Laplace `~0.3–0.6B` each (not `2.12B`).
- **Rates:** raw `cumulative/dur = 1.8k tok/s` (impossible vs API `80–120 tok/s`). Corrected `activeMs` from `task_started/task_complete` merged `toolWait` → `15–40 tok/s` wall / `60–110 tok/s` active, consistent with `1–3` concurrent sub-agents, not 10.
- **Scale:** `70,956 lines scanned`, `15,421 token_count` events (`--days 5 --top 10`).

### 1d) Code churn — cited commits and file deltas

**Commits that checkpoint the swarm (cite by SHA):**

```
dynaworld:
  3e698e8eebca0e2e90c62129a9b0600297c242cc  aug16: checkpoint remaining 425 dirty paths (pre-dedup net +20k)
  026c1306557d0fb7e7b7235ef40d8142f508d5db  aug16: swarm audit baseline + commit 426 dirty paths as-is
  cb0a904514658a0d4d5b0c2f9f9b8759ddabf448  Stream full-rate browser training and fix temporal VJP  # pre-swarm HEAD

gsplats_browser (parent):
  8b9cb19  aug16: bump dynaworld to pre-dedup swarm baseline 3e698e8
  a8229c6  docs(agents): point at spacetime research bundle before planning kernels
```

`git -C dynaworld show --stat 3e698e8` header (first 35 lines, verbatim):

```
commit 3e698e8eeb...
 59 files +20211/-1657
 BASELINES.md | 40 +-
 EXPERIMENTS.md | 556 +-
 PROJECT_INDEX.md | 307 +-
 TODO/README.md | 350 +-
 TODO/unified_paper_ablation_pipeline.md | 124 +-
 TODO/world_tubes_ordered_transfer_ablation.md | 134 +
 TODO/world_tubes_paper_finish_master_plan_2026-08-13.md | 1789 +++++
 TODO/worldfoam_memory_light_native4d.md | 1630 +++++
```

Top of `git diff 3e698e8^..3e698e8 --numstat | sort -k1 -nr` (exact file changes, cited):

```
6918  0  research_experiments/world_foam_lane2/kinetic_dense_cached_native_material_request.py
5567  0  research_experiments/world_foam_lane2/kinetic_lazy_native_material_step.py
4945  0  research_experiments/world_foam_lane2/kinetic_native_material_step_executor.py
4231  0  research_experiments/world_foam_lane2/kinetic_native_equal_rank_runtime_adapter.py
4083  0  src/train/paper_kinetic_fixed_camera_combined_state.py
3570  0  agent_notes/loose_notes/2026-08-04_03-19-39_worldfoam_scientist_feedback_and_fixed_site_source_closure.md
3250  0  research_experiments/world_foam_lane2/verify_worldfoam_training_memory_ablation.py
3197  0  research_experiments/paper_runner_suite/generate_world_tubes_paper_artifacts.py
2952  141 research_experiments/paper_runner_suite/run_unified_paper_ablation.py
2864  0  artifacts/foundation_gates/worldfoam_material_value_fit_cpu_20260727.json
2591  0  research_experiments/star_uvt_feature_tubes/projective_variable_camera_closure_death_curve.py
...
```

Parent `gsplats_browser diff 8b9cb19^..8b9cb19 --numstat` is `1 1 dynaworld` (pointer bump only); the `+4063/-195` parent net (`44 files`) is the *pre-bump* dirty state (`git -C . diff --stat HEAD` at audit time): `services/artifact_publishers +617`, `services/train_all/pipeline +370`, etc.

**Gross vs net (cited from `codex_scrape`):**

- `codex_scrape --days 5 --repo dynaworld --top 10` → `AGGR patches 7455 loc_add 163908`; per-thread top files repeat identically:
  - `Sagan 01a002be: run_unified_paper_ablation +2571, kinetic_dense_cached +2180, worldfoam_scientist_feedback +2079`
  - `Newton 01a002b0: same 2571/2180/2079`
  - `Laplace 01a002dd: verify_worldfoam_training_memory_ablation +704, generate_worldfoam_paper_b_artifacts +601` (195 patches — smaller lane)
  - `Tesla 019ffc2b: verify_worldfoam_public_quality_ablation_v2 +769`
- **Why net ≠ gross:** `git diff HEAD -- kinetic_dense_cached` is `0` lines (content reverted to HEAD), but scrape shows `+6540` gross across 3 threads rewriting it concurrently. `git log --oneline -- <monster>` is empty (never committed until `3e698e8`). Earlier audit looked at parent `gsplats_browser` net (`44 files`) and missed submodule `59 files`.

**Three monsters you asked about (cite on-disk size + audit finding):**

| file | on-disk | gross (swarm) | net vs HEAD | content (first lines) |
|---|---|---|---|---|
| `research_experiments/world_foam_lane2/kinetic_dense_cached_native_material_request.py` | `6918 l / 303 KB` | `+6540` | `0` (`git show HEAD:` same) | `"""Bounded dense-observation replay through one cached kinetic native lane.`…`PaperKineticCompiledCpuArtifact` + `ReplayableDenseObservationSource` + `KineticNativeMaterialStepExecutor` |
| `TODO/worldfoam_memory_light_native4d.md` | `1630 l / 107 KB` | `+6105` | `0` | `# WorldFoam memory-light native-4D completion` … `G6 0/21`, `G4 0/36`, `G4-v1 115M compiles intractable` |
| `third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/benchmarks/multicam_heldout_compare.py` | `6991 l / 280 KB` | `+5496` | `0` | `from paper_training_types import MetalKernelSpec` … `VideoMetricAccumulator(PaperRGBMetricAccumulator)` |

Lane totals: `research_experiments/world_foam_lane2/*.py 271 files / 198,632 l`; `dynaworld *.py 17,285 files / 951,554 l`. Gross `164k` = `17%` of lane py corpus in one night; net `20k` = `2%`.

**Other repos isolated (cite):** `codex_scrape --repo wave_sim --top 5` → `1765 patches / 39k loc` all in `/private/tmp/wave-sim-reflected-u4-completion/...` (not in `gsplats_browser`); `ai_trader --top 5` → `2833 / 69k` committed separately. No cross-repo leakage.

---

## 2) What Went Poorly (with evidence)

1. **Sprawl over reuse — 271 files where 1 trainer would do.** `kinetic_dense 6918 l / 111 defs`, `kinetic_lazy 5567 l / 85`, `kinetic_native_material_step_executor 4945 l / 90` share only `7–10` helpers (`_digest_parts`, `assert_retained`, `_require_positive_int`, `_tensor_signature` — from `python -c` shingle check `5-line dup 4.8%` but pattern dup `~70%`). Each re-implements loader lifetime → chunk cursor → fence → sealed receipt → quarantine. `multicam_heldout_compare 6991 l` re-implements `paper_training_protocol.py` timing/metrics (which itself grew `644/14` in `3e698e8`). `paper_runner_suite` has 20 files; `run_unified_paper_ablation 2952/141` is a near-copy of `run_unified_paper_matrix`.

2. **No kernel dispatch.** `MetalKernelSpec` exists (`paper_training_types.py:67`, imported by heldout) but not used as dispatch. Agents created "one file per hypothesis" instead of `KineticTrainer(strategy: MetalKernelSpec)`.

3. **Parallel contention without ownership.** 3 agents wrote `run_unified_paper_ablation +2571` identically; same for `kinetic_dense_cached +2180`. No `CODE_OWNERS` or `EXPERIMENTS.md#active-lanes` lock. Swarm is 107 forked threads, not 10 — `codex-report` top-10 hid the tail.

4. **Doc bloat as code.** `TODO/worldfoam_memory_light_native4d.md +6105` gross but `0` net; `TODO/world_tubes_paper_finish_master_plan 1789` added in `3e698e8`. Ledger is essential but was edited as scratchpad, burning `reasoning_output_tokens 1.5M` per thread.

5. **Commit hygiene.** Swarm ended with `426` dirty paths, `0` commits until audit. `git status --porcelain | wc -l = 426` in `dynaworld`; parent still pointed at `cb0a904` pre-swarm HEAD. Quota burned, no checkpoint.

6. **Token accounting blind spot.** Early report used `last - first` and wall `dur` → `1.8k tok/s`, `2.12B` Laplace. Only after vendoring `codeburn/codexusage` and streaming dedupe did Real Δ `6.35B` emerge. Prompt had no `token_cap`.

7. **Submodule boundary confusion.** `dynaworld` is `git@github.com:nbardy/dynaworld.git` submodule at `gsplats_browser/dynaworld`. Parent `git diff` hid `+20k` inside.

---

## 3) What We Need To Fix

**P0 — Freeze & checkpoint (done):** `026c130` + `3e698e8` + `8b9cb19`; tag `pre-dedup` (to be added). Keep evidence in commit messages.

**P1 — DRY core extraction (1 week, cite targets):**
- `src/train/kinetic_core/` with `artifact_store.py`, `dense_source.py`, `material_executor.py`, `kernel_registry.py` (wraps `MetalKernelSpec`), `lifecycle.py`.
- Collapse 3 kinetic files (`6918+5567+4945 = 17430 l`) → `KineticTrainer(strategy)` with adapters ≤300 l each.
- `paper_runner_suite`: single `run_unified_paper_matrix.py` driven by `world_tubes_full_public_matrix_v1.jsonc`; delete per-variant copies.
- `multicam_heldout_compare 6991 l → ~900 l` harness importing `PaperCostTracker/PaperPhaseTimer/PaperRGBMetricAccumulator`.

**P2 — Doc separation:** `TODO/worldfoam_memory_light_native4d.md → status.json + LEDGER.md (300 l max, append-only)`; agents append to `agent_notes/loose_notes/` only.

**P3 — Repo hygiene:** Budget `warn +2k / hard +5k` net per thread without `PLAN.md`; submodule-aware `AGENTS.md` checklist; paper artifacts stay in `research_notes/.../paper/`.

**P4 — Token budget:** Default `medium`, `ultra` needs `BUDGET.md` + `codex-report --days 1` pre-check; single-thread unless `GOAL.md` declares `parallelism: N` with shard.

---

## 4) How We Prevent This — AGENTS Rules & Prompt Patterns

### AGENTS.md patch (append to `## Agent Notes`)

```md
## Swarm & DRY Rules (added 2026-08-16, post-swarm audit)

1. One trainer, many kernels. Never create new world_foam_lane2/*.py reimplementing loader/fence/quarantine/receipt. Add KernelStrategy under src/train/kinetic_core/ and register MetalKernelSpec.
2. File ownership. Claim files in EXPERIMENTS.md#active-lanes before patch_apply; 2 threads may not patch same file within 5 min.
3. Gross vs net budget. Run codex_scrape --repo dynaworld --top 5 and git -C dynaworld diff --stat HEAD before/after; if gross >3×net or net >2000 stop and consolidate.
4. Commit small. Every ~500 net lines commit to dynaworld with scrape summary; do not push raw.
5. Docs are not code. Edit TODO/*.md only via status.json + loose_notes append; >50 lines needs runbook ref.
6. Submodule awareness. Run git -C . diff --stat HEAD AND git -C dynaworld diff --stat HEAD.
7. Token budget. Every goal needs effort: and token_cap:; ultra needs approval + report pre-check.
8. No swarm by default. One thread per goal; parallelism: N requires GOAL.md shard.
```

### Prompt patterns

Starter prefix (mandatory): `Read AUG_16TH_FOLLOWUP_SWARM_AUDIT.md §3-§4 before any patch. Use KineticTrainer + MetalKernelSpec; do not create new world_foam_lane2/*.py without KernelStrategy. Claim files in EXPERIMENTS.md#active-lanes. Effort: medium. Token cap: 200M. Commit every 500 net lines.`

Goal template, anti-sprawl check (`codex_scrape` + `diff`), and referee prompt as in first audit (unchanged).

Enforcement: add `codex-report`/`codex_scrape` to `CODE_ORGANIZATION.md` checklist; future CI blocks `world_foam_lane2/*.py` count growth without `kinetic_core/` counterpart or `TODO/*.md >200` net.

---

## Appendix — Evidence Commands (run to verify every claim above)

```bash
# swarm topology
python3 -c "import json,pathlib; subs=[json.loads(l) for l in open(pathlib.Path.home()/'.codex/sessions/2026/08/15/rollout-2026-08-15T09-40-52-01a002dd-4bd1-78a0-87ad-f671b16581d5.jsonl') if 'forked_from' in l][:1]"
ls -lt ~/.codex/sessions/2026/08/15/*.jsonl | wc -l   # 153
# detailed fork list: python snippet in §1b

# goal text
cat /Users/nicholasbardy/.codex/attachments/eaece9ef-b221-487e-ac5c-f07dc45a6f91/goal-objective.md

# token burn
python3 /Users/nicholasbardy/git/codex-report/codex_report.py --days 5 --top 10
python3 /Users/nicholasbardy/git/codex-report/codex_scrape.py --days 5 --repo dynaworld --top 10

# commits / file deltas
git -C dynaworld log --oneline -8                          # 3e698e8, 026c130, cb0a904
git -C dynaworld show --stat 3e698e8 | head -n 60
git -C dynaworld diff 3e698e8^..3e698e8 --numstat | sort -k1 -nr | head -n 25
git -C . log --oneline -6                                  # 8b9cb19 bump

# lane size
wc -l dynaworld/research_experiments/world_foam_lane2/*.py   # 271 / 198632
find dynaworld -name "*.py" | wc -l                         # 17285 / 951554
```

*Refined audit replaces 185-line first version; history preserved in `git show 026c130:AUG_16TH_FOLLOWUP_SWARM_AUDIT.md`.*

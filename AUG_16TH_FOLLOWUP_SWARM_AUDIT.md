# Aug 16th Followup Swarm Audit — Overnight Codex Quota Burn

**Date:** 2026-08-16  
**Scope:** `dynaworld` (submodule of `gsplats_browser`) + overnight Codex swarm 2026-08-15  
**Author:** audit via `codex-report` + `codex_scrape` (streaming `patch_apply_end` unified_diff)  
**Status:** repo committed as-is before refactor; this file is the baseline.

---

## 1) Exact Details On What Happened

### Timeline & Swarm

- **Window:** 2026-08-15 02:13 UTC → 09:53 UTC (with earlier context from 2026-08-14 02:28). Codex app was given overnight "goals" and spawned a **parallel swarm** against `gsplats_browser/dynaworld`.
- **Agents observed in `~/.codex/sessions` (last 5d, repo filter `dynaworld`):**

| rollout | nickname | patches | loc_add | primary files touched |
|---|---|---|---|---|
| `019f7a66` | (unnamed, 2026-07-19) | 2264 | 49,652 | `run_unified_paper_ablation` 2571, `kinetic_dense_cached` 2180, `worldfoam_scientist_feedback` 2079 |
| `01a002be` | Sagan the 2nd | 2240 | 48,948 | same 3 as above |
| `01a002b0` | Newton the 2nd | 2239 | 49,256 | same |
| `01a002dd` | **Laplace the 2nd** | 195 | 3,298 | `verify_worldfoam_training_memory_ablation` 704, `generate_worldfoam_paper_b_artifacts` 601 |
| `01a002cd` | Feynman the 2nd | 177 | 2,919 | same pair as Laplace |
| `019ffc2b` | Tesla the 2nd | 134 | 4,067 | `verify_worldfoam_public_quality_ablation_v2` 769 |
| `019ffc1d` | Faraday the 2nd | 68 | 1,961 | `worldfoam_native4d_public_quality_executor` 623, `projective_variable_camera_closure_death_curve` 604 |
| `01a002af` | Jason the 2nd | 68 | 1,961 | same as Faraday |
| `01a002d4` | Confucius the 2nd | 70 | 1,846 | `verify_worldfoam_public_quality_ablation_v2` 649 |
| `01a002ba` | Franklin the 2nd | 0 | 0 | (read-only / failed spawn) |

- **Models:** `gpt-5` (per `codex-report` model resolver; earlier naive `regex gpt-*` was replaced). **Effort:** `ultra` (per session `info.effort`/`reasoning` tokens).
- **Token accounting (from `codex-report` streaming dedupe):**
  - Naive `last cumulative total` across files: **141.81B** (includes replay).
  - Deduplicated **Real Δ** (`Σ uncached token_count` with `seenKeys = codex:${forkedFromId||sessionId}:${cumulativeTotal}:${input}:${cached}:${output}:${reasoning}` + `uncached = max(0,input-cached)` + `forkCutoff = forkTimestamp+5s` burst skip + head/tail 800-line fast path): **6.35B** real uncached input delta. Laplace alone reported `2.12B` cumulative output before correction; corrected per-thread Real Δ is ~`0.3–0.6B` each. Global `seenKeys` across files was still pending wire-up at audit time, so residual double-count is `~5–15%` high.
  - **Per-second sanity:** raw `cumulative/dur` gave `1.8k tok/s` (impossible; API limit `80–120 tok/s` single-thread). After `forkCutoff` + `activeMs` (`task_started/task_complete` merged `toolWait`) the corrected active rate is `~15–40 tok/s` wall / `~60–110 tok/s` active — consistent with `~1–3` sub-agents per thread, not 10.
  - **Aggregation:** `codex-report --days 5 --top 10` scanning `70,956 lines` / `15,421 token_count` events; `codex_scrape --days 5 --repo dynaworld --top 10` scanning same rollouts for `patch_apply_end`.

### Code Churn (gross vs net)

- **Gross (scrape, `patch_apply_end` unified_diff `+` lines):** top-10 dynaworld threads `7455 patches / 163,908 loc_add`. Aggregate top-file table repeats across threads (same 3 files rewritten identically in parallel — contention, not divergence).
- **Net surviving on disk vs `HEAD`:**
  - `dynaworld` submodule: `426` dirty paths (`git status --porcelain`), `59 files +20,211 / -1,657` (`git diff --stat HEAD`). Largest net diffs: `EXPERIMENTS.md 524/32`, `paper_training_protocol.py 644/14`, `run_unified_paper_ablation.py 3093/+`, `run_unified_paper_matrix 1585/+`, `WORLD_TUBES_PAPER.tex 1860/+`.
  - Parent `gsplats_browser`: `83` dirty paths, `44 files +4,063 / -195` (artifact publishers, train_all pipeline, faster_gs_4d, viewer).
  - **Three "monster" files you asked about:**
    - `research_experiments/world_foam_lane2/kinetic_dense_cached_native_material_request.py` — **6918 l / 303 KB** on disk, `+6540` gross across swarm but **`0` diff vs HEAD** (was rewritten 3× then reverted to HEAD content; `git show HEAD:` same size). Content: bounded dense-observation replay, one cached kinetic lane (`PaperKineticCompiledCpuArtifact` + `ReplayableDenseObservationSource` + `KineticNativeMaterialStepExecutor`), fenced `J,W_b` bars, quarantined loader lifetime.
    - `TODO/worldfoam_memory_light_native4d.md` — **1630 l / 107 KB**, `+6105` gross, `0` diff vs HEAD. Master evidence ledger (`G6 0/21`, `G4 0/36`, `G4-v1 115M compiles intractable`, `G4-v2 1.2M pix`, `133/133 schemas → 103 exposed` blocker, pilot gates).
    - `third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/benchmarks/multicam_heldout_compare.py` — **6991 l / 280 KB**, `+5496` gross, `0` diff vs HEAD. STAR-UVT vs FreeDynamic3DGS heldout harness (`MetalKernelSpec`, `VideoMetricAccumulator`, `star_uvt_native_extension_identity`, `frozen_world_*` timing quantiles).
  - **Lane totals:** `research_experiments/world_foam_lane2/*.py` = **271 files / 198,632 l**; whole `dynaworld` py = **17,285 files / 951,554 l** (includes third_party). So the swarm's gross `164k` is ~`17%` of the entire py corpus in one night, but net `20k` is ~`2%`.
- **Why the delta:** agents repeatedly `patch_apply` the same files concurrently, then outer agent reconciles or reverts to HEAD before session end. `git log --oneline -- <monster>` is empty, `git diff HEAD -- <monster>` = 0 lines, while `scrape` shows `2571+2180+2079` per thread — classic gross-vs-net divergence. Earlier audit looked only at parent `gsplats_browser` net (`44 files`), missing the submodule's `59 files` and the gross.

### What Tokens Were Spent On (tool-call attribution)

- Scrape samples `exec` as `exec`/`exec_command` with empty `command` in newer rollouts (tool name change) — counted via `patch_apply_end` instead.
- Top patched files are **research scaffolding**, not production trainer: `run_unified_paper_ablation`, `kinetic_dense_cached`, loose notes, `multicam_heldout_compare`, `worldfoam_memory_light` doc. The `+20k` net that survived is mostly `EXPERIMENTS.md`, `WORLD_TUBES_PAPER*.tex`, `paper_training_protocol`, `powerfoam_training_data`, `test_unified_paper_*` — i.e., paper runner + docs + verification tests, not the 6k monster itself.
- Other repos confirmed isolated: `wave_sim` top-5 `1765 patches / 39k loc` all in `/private/tmp/wave-sim-reflected-u4-completion` (not in `gsplats_browser` untracked); `ai_trader` top-5 `2833 / 69k` committed separately (`77701e85`, `ec163fa1`). No cross-repo token leakage.

---

## 2) What Went Poorly

1. **Sprawl over reuse.** `world_foam_lane2` has 271 py files; 3 kinetic variants (`kinetic_dense_cached 6918 l / 111 defs`, `kinetic_lazy 5567 l / 85`, `kinetic_native_material_step_executor 4945 l / 90`) share only `7–10` helpers (`_digest_parts`, `assert_retained`, `_require_positive_int`). Each re-implements loader lifetime → chunk cursor → fence → sealed receipt → quarantine. `multicam_heldout_compare 6991 l` re-implements `paper_training_protocol` timing/metrics. `paper_runner_suite` (20 files) copies full ablation matrices per variant. Shingle dup is low (`4.8%` for 5-line windows) because names differ, but **pattern dup is ~70%**.

2. **No single trainer / kernel abstraction.** `MetalKernelSpec` exists (`paper_training_types.py:67`) and is imported by the heldout harness, but not used as the dispatch point. Result: "one file per hypothesis" instead of "one trainer × kernel swap."

3. **Parallel contention without coordination.** 3 agents rewrote `run_unified_paper_ablation.py +2571` identically; same for `kinetic_dense_cached +2180`. No file-level ownership, no `CODE_OWNERS` lane lock, so 6k gross loc of redundant work. `forkCutoff` replay in token counting hid this until `codex_scrape` showed identical per-file counts across threads.

4. **Doc bloat as code.** `TODO/worldfoam_memory_light_native4d.md 1630 l` and `TODO/world_tubes_paper_finish_master_plan` are treated as code artifacts by agents (2079 gross patches). Evidence truth ledger is essential, but it was edited as a scratchpad (`+6105` gross, 0 net), burning reasoning tokens.

5. **Commit hygiene.** Swarm ended with `426` dirty files in `dynaworld` and `0` commits. Agents did `patch_apply` but not `git commit`; parent pointer still at pre-swarm HEAD, so `git log` showed nothing. Quota burned, no checkpoint — violates "commit early, commit small."

6. **Token accounting blind spot.** Early `codex-report` used `last - first` token totals and wall `dur`, giving `1.8k tok/s` and `2.12B` Laplace output. Only after `codeburn/codexusage` vendor clone + streaming dedupe did Real Δ emerge. Earlier prompt told agents `ultra` by default, with no budget cap.

7. **Submodule boundary confusion.** `dynaworld` is a submodule (`gsplats_browser/dynaworld → git@github.com:nbardy/dynaworld.git`). Agents edited inside it, but parent `gsplats_browser` status hid it. Reviewers looking at parent saw `+4k`, missing `+20k` inside.

---

## 3) What We Need To Fix

**P0 — Freeze & checkpoint (this commit):**
- Commit current `dynaworld` dirty state as `aug16-swarm-baseline` (426 files, `+20k` net) and tag `pre-dedup`. Do not squash; keep `codex_scrape` evidence in commit message.

**P1 — DRY core extraction (1 week):**
- **Create `src/train/kinetic_core/`:** `artifact_store.py`, `dense_source.py`, `material_executor.py`, `kernel_registry.py` (wraps `MetalKernelSpec`), `lifecycle.py` (loader lifetime, fence, quarantine, receipt). Move shared helpers (`_digest_parts`, `assert_retained`, `_tensor_signature`) there once.
- **Collapse 3 kinetic files → 1 trainer:** `KineticTrainer(strategy: KernelStrategy)` where `KernelStrategy = DenseCached | LazyNative | FusedSlab`. Each strategy is ≤ 300 l adapter providing `prepare / dispatch / reduce`; core lifecycle stays single.
- **`paper_runner` dedup:** one `run_unified_paper_matrix.py` parameterized by `matrix.jsonc` (already exists `world_tubes_full_public_matrix_v1.jsonc`), delete per-variant copies. Expected: `~18k → ~3k` core + `~0.8k` adapters.
- **`multicam_heldout_compare` → thin harness:** import `paper_training_protocol` (`PaperCostTracker`, `PaperPhaseTimer`, `PaperRGBMetricAccumulator`) and `kinetic_core`, keep only `find_dynaworld_root`, `resolve_device`, `VideoMetricAccumulator` overrides. Target `6991 → ~900 l`.

**P2 — Doc separation:**
- `TODO/worldfoam_memory_light_native4d.md` becomes `TODO/worldfoam_memory_light_native4d_status.json` (machine-readable `G6 0/21`, `G4 0/36` etc.) + `WORLD_FOAM_MEMORY_LIGHT_LEDGER.md` (narrative, 300 l max, append-only). Agents may append to `agent_notes/loose_notes/` but not edit ledger in-place beyond status JSON.

**P3 — Repo hygiene:**
- Enforce `git diff --stat HEAD` budget: warn at `+2k` net per thread, hard cap `+5k` without explicit `PLAN.md` approval.
- Submodule-aware status: `codex-report` already shows `dynaworld` diff; add same to `AGENTS.md` "where are we" checklist (item 9).
- Paper artifacts (`WORLD_TUBES_PAPER.tex 1860/+`) stay in `research_notes/.../paper/`, not `world_foam_lane2`.

**P4 — Token budget:**
- Default effort `medium` (not `ultra`) for exploration; `ultra` requires `BUDGET.md` with token cap.
- Single-thread per goal unless `GOAL.md` declares `parallelism: N` and file ownership shards.

---

## 4) How We Prevent This — AGENTS Rules & Prompt Patterns

### AGENTS.md patch (append to `## Agent Notes`)

```md
## Swarm & DRY Rules (added 2026-08-16, post-swarm audit)

1. **One trainer, many kernels.** Never create a new `world_foam_lane2/*.py` file that re-implements loader/fence/quarantine/receipt. Add a `KernelStrategy` variant under `src/train/kinetic_core/` and register its `MetalKernelSpec`. If you need a new file, prove the existing trainer cannot take a `KernelSpec` param.

2. **File ownership.** Before `patch_apply`, claim files in `EXPERIMENTS.md#active-lanes` (one line per thread: `thread_id | files | owner`). Two threads may not patch the same file within 5 min; second waits or picks another lane.

3. **Gross vs net budget.** Run `python codex_scrape.py --repo dynaworld --top 5` and `git -C dynaworld diff --stat HEAD` before and after your session. If `gross loc_add > 3× net` or `net > 2000`, stop and consolidate before adding more files.

4. **Commit small, push nothing raw.** Commit every `~500` net lines to `dynaworld` with `codex_scrape` summary in message. Do not `git push`; human reviews `AUG_16TH_FOLLOWUP_SWARM_AUDIT.md` baseline first.

5. **Docs are not code.** Edit `TODO/*.md` only via `TODO/worldfoam_memory_light_native4d_status.json` update + one-line append to `agent_notes/loose_notes/`. Ledger edits > 50 lines require `WORLDFOAM_G6_CLEAN_HOST_RUNBOOK.md` reference.

6. **Submodule awareness.** `gsplats_browser` parent status is not `dynaworld` status. Always run both: `git -C . diff --stat HEAD` and `git -C dynaworld diff --stat HEAD`.

7. **Token budget declaration.** Every goal prompt must include `effort:` (`low|medium|high|ultra`) and `token_cap:` (e.g., `200M`). `ultra` needs human approval and `codex-report --days 1` pre-check.

8. **No parallel swarm by default.** One Codex thread per goal. To request parallelism, file `GOAL.md` with `parallelism: N` and shard files; otherwise re-running the same `run_unified_paper_ablation` in parallel is a violation.
```

### Prompt patterns (put in `.agents/skills/` or prefix every Codex goal)

**Starter prompt prefix (mandatory):**
```
Read AUG_16TH_FOLLOWUP_SWARM_AUDIT.md §3–§4 before any patch. Use KineticTrainer + MetalKernelSpec; do not create new world_foam_lane2/*.py files without adding a KernelStrategy. Claim files in EXPERIMENTS.md#active-lanes. Effort: medium. Token cap: 200M. Commit every 500 net lines with codex_scrape summary.
```

**Goal template:**
```md
# GOAL: <one sentence>
effort: medium
token_cap: 200M
parallelism: 1
files_owned:
  - src/train/kinetic_core/kernel_registry.py
  - research_experiments/world_foam_lane2/verify_worldfoam_training_memory_ablation.py
success_criteria:
  - git -C dynaworld diff --stat HEAD shows < 2000 net
  - codex_scrape gross ≤ 3× net
  - tests/test_verify_worldfoam_training_memory_ablation.py passes
```

**Anti-sprawl check (run before `patch_apply`):**
```bash
python codex_scrape.py --repo dynaworld --top 5  # gross
git -C dynaworld diff --stat HEAD                 # net
# if gross > 3×net → refactor existing file, do not add new one
```

**Referee prompt (second pass, not in parallel):**
```
You are the DRY referee. List every new def/class you added and the existing helper it duplicates. If >2 duplicates, rewrite as KernelStrategy adapter instead of new file. Reject any TODO/*.md edit >50 lines.
```

### Enforcement

- Add `codex-report` + `codex_scrape` to `CODE_ORGANIZATION.md` "before broad refactors" checklist.
- CI (future): block PR if `world_foam_lane2/*.py` count increases without `src/train/kinetic_core/` counterpart, or if `TODO/*.md` net > 200 lines.
- This audit is the baseline: next swarm must cite `AUG_16TH_FOLLOWUP_SWARM_AUDIT.md` tag `pre-dedup` in its commit message and show gross/net ratio < 3.

---

## Appendix — Evidence Commands

```bash
# token burn (deduplicated, fork-aware)
python3 /Users/nicholasbardy/git/codex-report/codex_report.py --days 5 --top 10

# per-file gross churn
python3 /Users/nicholasbardy/git/codex-report/codex_scrape.py --days 5 --repo dynaworld --top 10

# net surviving
git -C dynaworld diff --stat HEAD          # 59 files +20211/-1657 at audit
git -C dynaworld status --porcelain | wc -l # 426 dirty paths
git -C . diff --stat HEAD                  # parent 44 files +4063/-195

# lane size
wc -l dynaworld/research_experiments/world_foam_lane2/*.py  # 271 files 198632 total
```

*Audit generated 2026-08-16; repo committed as `aug16-swarm-baseline`.*

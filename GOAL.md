# GOAL — Evidence-First Overnight Paper Run

## Objective

Advance **World Tubes first** and **WorldFoam second** toward conference-level
scientific quality by producing verifier-accepted experimental evidence. “ICLR
level” describes the quality bar; it does not authorize ICLR-specific
packaging, a new manuscript, new mathematics, or a new renderer.

## Budget And Delegation

- Hard goal token budget: `2,000,000`; stop new work at `1,600,000`.
- Maximum concurrency: one lead plus two subagents.
- Lead reasoning: high. Subagent reasoning: medium. Never use ultra.
- Subagents may not spawn agents.
- One accelerator/native-build process at a time. No remote compute or paid
  service without explicit user authorization.
- The lead owns all shared status/manuscript files. Subagents receive disjoint
  read/run responsibilities and return compact evidence summaries, not new
  handoff documents.

## Canonical Truth

Read only what is needed from:

1. `TODO/world_tubes_paper_finish_master_plan_2026-08-13.md` for Paper A;
2. `TODO/worldfoam_memory_light_native4d.md` for Paper B;
3. retained verifier-accepted JSON and generated evidence ledgers;
4. `BASELINES.md` for publishable measured rows.

The files named `*_ICLR_MAIN_DRAFT.md` are historical submission-shaped
working drafts. They are not separate venue commitments. Improve them only
after accepted evidence exists and do not create another paper draft.

## Starting Truth

- Paper A: theorem accepted; public contexts `0/7`, lane records `0/21`, frozen
  same-world sweep absent, bounded variable-camera curve absent.
- Paper B: accepted synthetic G0/G3 evidence; G6 memory rows `0/21`, G4-v2
  public-quality rows `0/36`; installed native extension is stale.
- Tests, dry plans, source completeness, and old schema-v1 runs count as zero
  new ablation evidence.

## Smart Subagent Assignment

The lead may create at most two subagents after reading the canonical plans:

1. **Paper A evidence operator:** owns Paper A preflight and retained runtime
   artifacts only. It may run the focused gate, schema-v2 smoke, frozen
   same-world sweep, variable-camera curve, and public matrix in the exact
   order prescribed by the master plan. It does not edit manuscripts or shared
   runners unless a reproduced correctness defect blocks the frozen contract.
2. **Paper B evidence operator:** owns G4/G6 dry plans, native capability check,
   guarded rebuild, pilot, and retained runtime artifacts only. It must run the
   G4 two-route pilot before G4’s 36 rows and the G6 clean-host dry plan before
   any rebuild or execution. It does not write new verifiers, plans, or lane
   variants.

The lead may run one lane itself instead of spawning both. Paper A has
priority. Because accelerator work is sequential, agents coordinate with the
lead before every build or MPS launch; they may perform independent read-only
validation while another lane runs.

## Execution Order

1. Record clean main/submodule commits and current accepted evidence counts.
2. Run allocation-free/dry preflights. Do not import Torch or sample the host
   in source-only dry plans where the existing contract forbids it.
3. Check live host guards immediately before any build or accelerator process:
   at least 8 GiB free disk, at least 8 GiB available RAM, swap at most 2 GiB,
   load average at most 8. A failed guard ends that runtime lane for the night.
4. Paper A: focused behavioral gate → schema-v2 evidence smoke → frozen
   same-world sweep → bounded variable-camera curve → seven public contexts.
5. Paper B: G4/G6 dry plans → one guarded native rebuild if required → G4
   two-route pilot → G6 `21` rows plus restart processes → G4 `36` rows. A
   failed pilot stops its full matrix.
6. Independently verify every produced artifact. Preserve honest negative
   results. Never hand-edit evidence or splice rows across source revisions.
7. Only after evidence acceptance, regenerate existing tables/figures and
   update the existing manuscript, `EXPERIMENTS.md`, and `BASELINES.md` once.

## Resource And Scope Stops

Stop the affected lane immediately when:

- a host/resource guard trips;
- a required dataset, credential, native binary, or runtime capability is
  absent and cannot be repaired by the one already-declared bounded step;
- the same blocker occurs twice;
- correctness parity fails under the frozen method contract;
- the token budget reaches 80%; or
- no remaining action can create accepted evidence.

When stopped, add one concise status entry to the existing canonical ledger and
return the blocker. Do not create another audit, verifier, schema, TODO, paper
scaffold, project page, visualization, cleanup branch, or mathematical method.

## Success

Success is measured only by retained verifier-accepted artifacts:

- minimum useful overnight success: at least one new accepted Paper A runtime
  component or one verified Paper B real-native pilot;
- Paper A closure: frozen sweep accepted, variable-camera curve accepted, and
  all seven public contexts / 21 lane records accepted;
- Paper B memory closure: all 21 G6 rows and restart checks accepted below the
  declared 2-GiB MPS and 4-GiB process-group RSS ceilings;
- Paper B quality closure: all 36 G4-v2 rows accepted;
- final manuscript work uses only accepted evidence, includes limitations and
  negative results, and makes no venue-specific claim unless the user selects
  a venue.

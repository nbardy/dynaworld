# GOAL — Overnight World Tubes Iteration

## Active continuation — 2026-09-12

The user renewed the broad objective: iterate overnight, run experiments,
read the math and code, commit useful changes, retain training notes, and
improve from the evidence. The completed local playground below is history,
not the success definition for this renewed campaign. No token budget was
specified for the active goal; the historical token cap below is not its cap.

The September 12 source recipe reproduced at 21.77 dB. After correcting
initialization coverage and temporal-slice bounds, a shared-world depth/footprint
control reaches 19.69/14.96 dB train/heldout at 400 updates with zero overflow.
Periodic learned-state checkpoints now preserve the trajectory. Keep these
one-seed local runs as quality controls and resolve the independent frozen
replay/compiled image/VJP/fallback mismatch before claiming sublinear scaling.
The production centered-depth fix (STAR `048f797`) now passes F4 image/all-
world-VJP checks and non-unit F3 slicing. Retained bytes and cached UVT updates
include source depth. Time-local fallback and native interval union (STAR
`1a1fc3e`) now pass local F4/F3 image/VJP/fallback checks at 12.78%/17.04%,
below the unchanged 20% limit with zero overflow. Sparse reference fallback
(STAR `a7ec561`) now retains all contributors in flagged tiles and reduces
F4 compiled forward/backward 10.76 -> 3.90 s with image/world-VJP parity.
Exact-zero spatial-depth specialization (STAR `ccf6c1c`) now cuts compilation
8.02 -> 0.94 s; total compile/forward/backward is 4.94 s with F4/F3 numerical,
slice, and exact tensor/membership checks passing. General spatial visibility
remains tested. Batched fallback (STAR `68fa855`) then cuts warm compiled
forward/backward 3.97 -> 1.02 s, compilation-inclusive 4.94 -> 2.00 s, with
F4/F3 image/world-VJP/slicing passing and byte-identical retained atlases.
Cold execution remains costly; warm replay is ~0.057 s. Next inspect
operation-level backward/gather costs and remaining compilation overhead,
preserving centered source depth, source-id ties and all contributors.
Full scaling/public evidence remain open. See `agent_notes/loose_notes/2026-09-13_02-46-35_batched_fallback_compositing.md`.
The subsequent full-world Kineto/record_shapes profile exceeded the unchanged
3-GiB process-tree cap and was terminated. Its profiling lane is stopped;
no operator attribution was obtained. Keep the passing ordinary benchmark
as the current evidence and do not automatically retry this profiling mode.
See `agent_notes/loose_notes/2026-09-13_02-51-34_full_world_profiler_resource_stop.md`.
The independent 800-update quality lane now reaches 20.84/15.30 dB train/heldout.
Its own step400 checkpoint gives 19.67/14.93 dB under the same evaluator;
both have zero overflow and new swap. The separate older 400-step source hash
differs, so the within-trajectory comparison is primary. Next evaluate this
better-fitted 2048-tube world through the bounded four-time replay/compiled
contract before a larger sweep, retaining every numerical and resource gate.
Do not discard successful source fits. See `agent_notes/loose_notes/2026-09-13_03-12-15_world_tube_800step_same_trajectory_control.md`.

September 13 fitted-world compiler check: per-trace axis support bounds
(STAR `a316cc6`) remove false tile overflow, reducing the maximum from 938 to
119 at unchanged capacity 256. RGB parity passes at 1.40e-6, but temporal-
precision VJP error is 3.24e-5 against a 1e-5 gate. Identical backward passes
also vary by up to 1.22e-5 on that parameter; the full check remains negative.
Next isolate temporal-adjoint conditioning. No scaling/public count change.
See `agent_notes/loose_notes/2026-09-13_03-41-15_compiled_support_bounds_and_temporal_gradient_noise.md`.

September 13 centered temporal adjoint: STAR `f98b288` evaluates and
accumulates the envelope as lambda*(t-t0)^2. The fitted 2048-tube world now
passes F4 image/all-world-gradient and slicing gates; raw temporal-precision
error falls 3.24e-5 -> 3.14e-7 at unchanged 1e-5 tolerance. Repeated identical
backward variation falls to ~1.26e-7. Warmed median compile/forward/backward
is 4.23/0.37/8.62 s versus 0.078 s total replay, so speed remains negative.
Next test batched UVT lowering against the same retained world and atlas.
No full scaling/public count change. See `agent_notes/loose_notes/2026-09-13_04-06-51_centered_temporal_adjoint_and_fitted_world_timing.md`.

September 13 batched UVT lowering: STAR `470a36d` replaces per-tube
coefficient graphs (84,235 nodes) with 65 batched nodes. The same fitted-world
atlas bytes and F4 image/world-gradient/slice gates are preserved. Warmed
forward/backward falls 8.98 -> 0.575 s; compilation falls 4.23 -> 1.67 s.
Replay also improves to 0.045 s, so this remains slower than replay. CPU
profiling locates the next cost in visibility pair checks and scalar depth
sorting. No scaling/public count change. See `agent_notes/loose_notes/2026-09-13_04-18-45_batched_uvt_lowering_and_metadata_profile.md`.

September 13 visibility/density result: STAR `e1c33c1` preserves the exact
F4 atlas while reducing compilation 1.67 -> 0.98 s. The fixed-world F4/8/16/32
sweep completes under the same guards; three rows pass, F32 fails one pixel
at 6.17e-4. A source-alpha cutoff crossing explains that pixel; old/new
visibility cells are exact. Eightfold frame growth costs 8.19x compilation
and 10.50x backward, so no sublinear speed claim follows. The shared forward
jump above F8 includes measured eight-frame-LRU decoding. Next repair source
alpha-branch compatibility, then separate target-loading timing. Public counts
stay unchanged. See `agent_notes/loose_notes/2026-09-13_04-49-00_visibility_bookkeeping_density_and_alpha_cutoff.md`.

September 13 alpha-cutoff repair: STAR `01bd9e0` preserves source cutoff
membership while differentiating compiled values. The identical fitted-world
F4/8/16/32 sweep now passes all four rows; F32 max RGB error falls 6.17e-4 ->
8.64e-7. Previous atlas tensors/topology are exact, with 36 extra bytes/trace.
The 159-test Metal/CPU gate and resource guards pass. Runtime remains slower
than replay; next separate target loading in the forward measurements. Local
accepted rows rise 3/4 -> 4/4; public counts remain unchanged. See `agent_notes/loose_notes/2026-09-13_05-19-02_source_alpha_cutoff_repair.md`.

September 13 measured forward breakdown: STAR `91ee6b0` partitions each
paired trial into evaluator, CPU target load, transfer and loss; existing
validators check that phases sum to the same forward total. All F4/8/16/32
image/gradient gates and exact atlas bytes pass. F32 decoding costs ~6.25 s,
but compiled evaluator+backward still costs 7.02 s versus replay 0.545 s and
grows 12.64x for 8x frames. CPU packing alone takes ~32 ms on the first frame.
Next batch its scalar buffer writes, preserving exact buffers and gradients.
Public counts stay unchanged. See `agent_notes/loose_notes/2026-09-13_05-32-47_frozen_forward_breakdown_and_packing_cost.md`.

September 13 batched packing: STAR `8720d4e` preserves all five buffers
across 60 real frame chunks and cuts paired CPU packing 31.65 -> 7.75 ms.
The 171-test CPU/Metal gate and all fitted-world F4/8/16/32 checks pass, with
exact retained atlases and no new swap. F32 evaluator+backward falls 7.02 ->
4.05 s; full compilation/forward/backward falls 21.42 -> 18.47 s. Replay
remains faster, and no sublinear claim follows. Next profile remaining
packing/validation/slicing/native-order costs before another optimization.
See `agent_notes/loose_notes/2026-09-13_05-50-00_batched_interval_packing_and_fitted_world_timing.md`. Public counts remain unchanged.

September 13 retained backward inputs: STAR `1aedaf4` fixes a reproduced
multi-forward topology/gradient defect and removes duplicate backward packing.
All 173 CPU/Metal tests and fitted-world F4/8/16/32 numerical checks pass, with
exact atlases. F32 backward falls 1.26 -> 0.46 s; evaluator+backward 4.05 ->
3.22 s; full compiled cost 18.47 -> 17.80 s. Peak RSS is 1.74 GiB and new
host swap 1.32 MiB, within unchanged limits. Still slower than replay and not
sublinear. Next profile remaining F32 forward costs with bounded Python tools.
See `agent_notes/loose_notes/2026-09-13_06-18-00_retained_forward_inputs_fix_backward_topology_and_cost.md`. Public counts remain unchanged.

September 13 visibility reuse: STAR `c97e0c2` memoizes exact root/depth
queries within each compilation. Paired CPU visibility falls 4.98 -> 1.48 s
with identical cells; 174 CPU/Metal tests and all fitted-world rows pass.
F32 compilation falls 8.19 -> 4.37 s and full cost 17.80 -> 13.83 s, with
exact atlas bytes, 1.93-GiB peak RSS and no new swap. Rendering is nearly
unchanged; no sublinear claim follows. Next address measured scalar depth
reads in fallback marking. See `agent_notes/loose_notes/2026-09-13_06-39-00_visibility_query_reuse_and_f32_profile.md`. Public counts remain unchanged.

September 13 fallback host values: STAR `dc067d1` preserves exact metadata
while reducing paired CPU fallback marking 1.60 -> 0.24 s. All 174 CPU/Metal
tests and fitted-world rows pass with identical atlases. F32 compilation falls
4.37 -> 3.02 s; full cost 13.83 -> 12.47 s. Peak RSS is 1.87 GiB with no new
swap. Rendering is unchanged and sublinear scaling remains unproven. Next
combine duplicate cell construction in frame slicing. See `agent_notes/loose_notes/2026-09-13_06-53-00_exact_host_depth_values_and_fallback_cost.md`.
Public counts remain unchanged.

September 13 frame slicing: STAR `f3544aa` combines duplicate cell construction.
All 174 CPU/Metal tests, 60 exact real chunks plus 12 other windows, and all four
fitted-world rows pass. Paired CPU slicing falls 1.08 -> 0.85 s; F32 evaluator+
backward is 3.16 -> 2.96 s and full cost 12.47 -> 12.17 s. Peak RSS is
1.91 GiB with no new swap. Exact atlas bytes hold; sublinear total scaling
remains unproven. Next measure interval lookup to avoid full-cell scans per frame.
See `agent_notes/loose_notes/2026-09-13_07-07-55_single_cell_construction_in_frame_slicing.md`. Public counts remain unchanged.

September 13 overlap sweep: STAR `bb6c6f4` replaces full-cell scans per frame
with transient ordered interval events. All 181 CPU/Metal tests, exact multi-
chunk slices and four fitted-world rows pass. Paired CPU slicing falls 0.864 ->
0.679 s including index setup; F32 evaluator+backward falls 2.96 -> 2.74 s,
full cost 12.17 -> 12.07 s. Smaller-frame timings are mixed. Exact atlases,
1.87-GiB RSS and zero new swap hold; no sublinear claim. Next refresh the bounded
Python profile. See `agent_notes/loose_notes/2026-09-13_07-21-55_interval_sweep_for_frame_slicing.md`. Public counts stay unchanged.

September 13 one-frame packing: refreshed bounded profiling identifies general
interval packing as a remaining cost. STAR `44c8265` uses exact ordered deduplication
for [0,1) intervals. All 183 CPU/Metal tests, five-buffer parity and four fitted-world
rows pass with exact atlases. Paired CPU packing falls 0.542 -> 0.208 s; F32
E+B 2.74 -> 2.34 s and full cost 12.07 -> 11.53 s. Peak RSS 1.91 GiB,
zero new swap. No sublinear claim. Next batch quadratic support bounds while
preserving float32 arithmetic. See `agent_notes/loose_notes/2026-09-13_07-38-18_profile_and_exact_one_frame_packing.md`. Public counts unchanged.

September 13 batched bounds: STAR `48148ac` batches quadratic support extrema
with exact float32 values and double-precision vertex inclusion. All 184
CPU/Metal tests and four fitted-world rows pass with byte-identical atlases.
Paired CPU rebinning falls 1.058 -> 0.706 s; F32 compile 2.89 -> 2.40 s,
full cost 11.53 -> 11.12 s. RSS 1.91 GiB; new swap 0 bytes.
No sublinear or quality claim changes. Next refresh the existing bounded
Python profile. See `agent_notes/loose_notes/2026-09-13_07-55-22_batched_quadratic_support_bounds.md`. Public counts unchanged.

September 13 visibility reuse: STAR `f735898` retains exact midpoint depths per
trace/interval within one compilation. All 184 tests and four fitted-world rows
pass with byte-identical atlases. Paired CPU visibility falls 1.470 -> 1.014 s;
F32 compile 2.40 -> 1.84 s, full cost 11.12 -> 10.55 s. RSS 1.89 GiB, zero swap growth.
A separate exact-byte CPU probe cuts F32 target loading 6.22 -> 1.12 s using
requests of eight with the same LRU8. Next integrate that into both consumers,
inside timing and with one-frame Metal residency. No sublinear/public count change.
See `agent_notes/loose_notes/2026-09-13_08-13-52_visibility_midpoint_reuse_and_target_loading.md`.

One lead owns new run configs, narrowly necessary source fixes, commits, and
shared status updates; no subagents. Accelerator jobs and native builds remain
sequential. Retain the existing small-playground host/RSS/MPS/swap/disk gates;
stop the affected runtime lane on a guard trip. Each diagnostic has a declared
finite update count and a 600-second wall timeout. Logging remains offline
after the prior upload rejection. Do not weaken publication acceptance gates.

Observable progress is retained optimizer-run evidence with images and metrics,
an independently checked reproduced defect and fix, or manuscript improvements
grounded in accepted evidence. Record negative results and source provenance.
Keep practical diagnostic results separate from accepted paper counts. Commit
only changes whose ownership and scope are established; preserve unrelated WIP.
This campaign is not complete merely because the first control reproduces.

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

- Paper A: theorem and bounded variable-camera curve accepted; public contexts
  `0/7`, lane records `0/21`, frozen same-world sweep and moving-camera density
  absent.
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
3. Check live host guards immediately before any build or accelerator process.
   The stricter lane-specific runner gate always wins: Paper B currently needs
   at least 8 GiB free disk and available RAM, swap at most 2 GiB, and load at
   most 8; Paper A's current matrix additionally requires at least 32 GiB free
   disk and 10 GiB available RAM. A failed guard ends that runtime lane for the
   night.
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

# Dyna World v1

**Core Hypothesis:** World models are essentially video models that have
already learned the geometry of the world. The first goal is a
`video => world token` base model: observed video maps to scene-state tokens
that decode to Gaussian splats and stay consistent under novel cameras.

## Key Beliefs & Postulates

1. **World Models = Video Models:** Video models already learn the geometry of the world.
2. **World Tokens Before Generation:** A token is a world token only if it can
   decode to splats that remain consistent across novel camera angles.
3. **Efficient 3D Extraction (Linear Probe):** Mapping video $\rightarrow$
   world tokens $\rightarrow$ Gaussian splats should require much less data
   than training a new foundation model. This is analogous to how depth can be
   extracted from Stable Diffusion using simple probes.
4. **Static == Dynamic:** This approach applies just as easily to dynamic video as it does to static video.
5. **Self-Supervised Video Models as Foundation:**
   - Any 3D labels are inaccurate at scale.
   - The ground truth for 3D is light from video.
   - The best way to train a world model is just to train a self-supervised video model.
6. **Adapters over New Foundations:** If you need other outputs from a world
   model, you should finetune an adapter to extract it rather than training a
   completely new foundation model.
7. **Generation Comes Next:** After the base representation works, train AR or
   diffusion models over world tokens for video continuation, image=>video, and
   text=>video.

## Rationale & Precedents
- Recent advancements in tokenizer papers show strong performance on as few as 50k samples.
- Techniques like Flux Edit LoRAs work on as few as 5-10 samples.
- Therefore, we should be able to train an efficient adapter on video
  reconstruction and novel-camera pressure to get 3D-consistent world tokens
  out of an existing video world model.

## Next Steps / Ablations
- [ ] Select a baseline video model / foundation model to probe.
- [ ] Define the world-token contract: token shape, camera interface, time
  interface, and splat decoder.
- [ ] Construct a scene-diverse micro-dataset for the initial training run.
- [ ] Establish source-camera and novel-camera metrics for the decoded splats.
- [ ] Keep AR/diffusion world-token generation as the follow-up stage, not the proof that stage 1 worked.

---

## Navigation — where to look for what

Layered, strategic → tactical:

- **`meta_philosophy/`** — strategic core. Read its `README.md` first. How to design architectures, failure modes F1–F7, mistakes log (regression-prevention), prompt-guidance notes, and the driver-prompt template for external LLMs.
- **`meta_philosophy/world_splat_tokens_vs_observed_modality_tokens.md`** — token philosophy. Defines observation/world/splat/memory/camera tokens, explains why splat tokenizers differ from text/image/video tokenizers, and records when two-stage tokenizer training is legitimate versus degeneracy-freezing.
- **`meta_philosophy/dynaworld_architecture_solution_prompt.md`** — paste-ready external-model prompt for generating 3-4 DynaWorld architecture solutions with compact rationale fields, anti-degeneracy arguments, falsification tests, and synthesis.
- **`framing_the_problem/`** — three framings of the novel-view bottleneck. Framing 1 is information-theoretic (for deriving losses); framing 2 is the self-sufficiency / generative-reconstruction contract (for auditing architectures); **framing 3 is the patched bitter-lesson predictive-quotient baseline and the current default** — start there for proposing anything new. Has its own `README.md` with when-to-use guidance.
- **`training_contract_v1.md`** — operational contract for patched framing 3: `D_var` sampler, model signatures, baseline losses, diagnostics, escape hatches, support assumptions, deployment/export contract, and failure tripwires.
- **`data_contract.md`** — canonical data-loader and manifest contract: same-view single-sequence scale pretraining, multicam heldout-camera supervision, current 1k row counts, heldout pools, and the next mixed-scheduler bridge.
- **`three_architectures_for_novel_view_synthesis.md`** — concrete architecture candidates (A/B/C), diagrams, head-to-head debate, pioneer pick. Cross-references the framings.
- **`incidence_kernels_and_material_objects.md`** — theory/implementation note for the four ray-event incidence laws: projected conics, exact ray-Gaussian line integrals, slab intersections, and full volume integration. Includes equations, pseudocode, material-regime analysis for rigid bodies/water/fog/cloth, and the staged ablation plan.
- **`gauged_uvt_trace_atlas/`** — camera-ray bundle theory for STAR UVT and WorldFoam under complex camera programs. Defines UVT traces as `pi_* Gamma^*` fiber pushforwards, then iterates ten subtheories for gauges, projective/rational orbit traces, visibility strata, adjoint training, and Metal acceptance gates. The cross-track depth-fiber note there explains the shared ray-depth fiber: World Tubes marginalizes it into UVT footprints with conditional depth/order certificates, while WorldFoam keeps it as the transmittance axis.
- **`meta_review_jul_28th.md`** — external literature/method review of the
  World Tubes/STAR/WorldFoam program. Its durable project interpretation is
  `../agent_notes/loose_notes/2026-07-28_16-50-53_meta_review_project_integration.md`:
  preserve World Tubes as the bounded camera-program compiler, treat the
  memo's broader composite-transfer and event-boundary system as future work,
  and use its same-world causal tests rather than copying its proposed abstract
  as a present-tense implementation claim.
- **`spacetime_gaussian_representation/`** — full 2026-07-23 representation audit. Recovers the native `mu4 + SPD(4)` world Gaussian, proves its exact equivalence to the full linear-tube block form, distinguishes persistent semidefinite tubes from finite-lifetime Gaussians, audits slicing/opacity/rotation/UVT semantics, catalogs twenty model families, and defines matched baselines plus implementation/falsification gates.
- **`world_tubes_spd4_worldfoam_handoff_2026-07-28.md`** — standalone shareable handoff for the complete World Tubes/native-SPD(4)/Ordered Ray Transfer/WorldFoam arc. Covers the resolved representation questions, key derivations, production code and flags, bounded results, shader and optimization failures, paper/novelty classification, host-safety policy, dirty-source boundary, and exact submission/post-paper next actions.
- **`world_tubes_paper_completion_handoff_2026-08-10.md`** — frozen Paper A completion handoff. It separates the new WorldFoam connection lane from the World Tubes submission, records the current `0/7` schema-v2 public ledger and missing causal/variable-camera results, pins the dirty-source and corrected-Neural3D-calibration boundaries, and gives the exact safe-host experiment, acceptance, artifact, and venue-PDF finish sequence.
- **`../research_experiments/world_foam_connection_v2/`** — isolated pure-Torch implementation of the constrained Lagrangian ray-fiber optical connection. It includes the exact stable-P0 theorem core, bounded shared flow, endpoint provenance, selected-direction JVPs, A0/A0c/A1/A2 temporal representations, analytic falsification fixtures, and a fail-closed JSON oracle; it is separate from Paper A and has not been runtime-validated on the local incident-prone host.
- **`../research_experiments/spd4_world_tubes/`** — float64 oracle for native `mu4 + SPD(4)`, affine camera/ray-gauge compilation, exact UVT/depth conditionals, confidence-band ordering, amplitude semantics, and retained-fiber falsification. The production STAR variant now keeps `full_spd4` parallel to `legacy_tube`, threads peak-splat/Beer--Lambert and fiber-integrated/peak-density choices, consumes conditional-depth variance in order certificates, exposes retained/hybrid Metal backends, and has a one-chart first-order moving-camera compiler.
- **`../agent_notes/loose_notes/2026-07-27_17-36-51_spd4_physical_renderer_and_bounded_training.md`** — current SPD(4) production handoff and bounded evidence. It records the exact renderer axes, Beer--Lambert and fiber-Jacobian math, retained-depth VJP, the one-seed 16-frame/40-step parameter-matched gains (`+1.0189/+1.1467 dB` heldout with `-26.5%` sampled peak driver bytes), and the negative result that the exact certificate falls back everywhere at 199 same-depth atoms.
- **`camera_program_compiler_and_cellular_backend_synthesis.md`** — 2026-07-26 audit of the ChatGPT Pro STAR/cellular proposal against the current World Tubes paper and research archive. Separates world/compiler/evaluator/adjoint layers, corrects stale camera-path claims and public naming, compares direct P1 corner extinction with log-FEM WorldFoam, audits prior art, and defines the same-world renderer ablation plus bounded cellular kill gates.
- **`renderer_lane_taxonomy.md`** — canonical naming and priority map. Use it to distinguish the retained Gauged UVT mathematics, the World Tubes paper method, STAR UVT/projective implementation names, the WorldFoam second-paper operator ordering, PowerFoam lineage, baseline/probe lanes, and the current finish/drop sequence.
- **`worldfoam_paper/`** — second-paper lane for WorldFoam as a camera-gauged lifted opacity/transmittance representation. Use this when the question is ray-fiber foam density, bounded cell complexes, Beer-Lambert prefix rendering, optical-transfer event algebra, cell-path atlas math, Cech/AABB or witnessed topology, and the current speed-versus-quality acceptance ladder. Keep it separate from the World Tubes paper, which is the baseline-compatible dynamic Gaussian compiler. The current plan promotes the visibility monoid, cell-path replay theorem, commutator theorem, monoid VJP, and event closure while keeping Hessians, Magnus compression, boundary flux, flux witness scores, feature-gauge transfer, and ray-space transfer behind tests. The 2026-07-26 expansion adds a complete Gaussian/log finite-element WorldFoam derivation, a self-normalized convex-potential atom audit, and a paper/method/Metal-gate classification; start with `worldfoam_paper/PAPER_METHOD_CLASSIFICATION_AND_METAL_GATES.md` for the decision and `worldfoam_paper/GAUSSIAN_FINITE_ELEMENT_WORLD_FOAM.md` for the math. For the frame-density-independent retained-order formulation and its remaining production/theorem gaps, use `worldfoam_paper/WORLD_FOAM_DYNAMIC_DEPTH_ORDER_MATHEMATICIAN_PROMPT.md`. The first existing cell-path code gate remains `worldfoam_paper/experiment_designs/cell_path_optical_transfer_fixture.md`.
- **`worldfoam_paper/WORLD_FOAM_STRATIFIED_LAGRANGIAN_CONNECTION_AUDIT_2026-08-05.md`** — complete audited intake of the fiber-bundle/connection/curvature/factorization/jet/monodromy proposal. Use its repo-native right-ordered equations and counterexamples rather than the raw scientist dump. It promotes curvature only to a falsifiable theorem/diagnostic and parks a runtime until the three-way `U`/`U_tilde`/`K_F` cost-and-tangent gate wins.
- **`worldfoam_paper/WORLD_FOAM_MEASURE_CONNECTION_SYNTHESIS_2026-08-05.md`** — canonical bridge from the translated optical-depth measure `(kappa,nu)` through the compact affine quotient `(beta,m)` to the constrained Lagrangian ray-fiber connection. It records the full repo-ordered math, admissible 3D-to-sensor flow lift, P0 interface residual, distinct physical-`U`/group-completion-`U_tilde`/signed-`K_F` ABIs, exact oracle, work model, paper claim ladder, and kill criteria.
- **`worldfoam_paper/WORLD_FOAM_UNION_LOCAL_FUSED_GEOMETRY_V2_DESIGN_2026-08-05.md`** — exact factorization of block-local kinetic geometry cotangents through the request union, including the `12(S-U)(6+C)` bridge saving, optional `8U` CPU-map charge, three-index-space ABI, fail-atomic transaction, source seams, and falsification gates. This is the first memory-v2 implementation plan after staged/fused accounting, not new optical-transfer math.
- **`../research_experiments/world_foam_lane2/finite_element_material_transfer.py`** — fixed-tape M0--M5 material reference with the neighboring standalone Metal source, fail-loud thin parity wrapper, focused tests, and `run_finite_element_material_gate.py` JSON gate. This tests one segment's transfer element and VJP; it is not a compact native-4D foam field, cell compiler, or full renderer fork.
- **`worldfoam_material_basis_selection_gate.md`** — durable interpretation
  of the verified three-seed partial-chord material fit. M3 and M5 have matched
  payloads and complementary exact-family wins, so no universal law advances
  to native-4D integration yet. The executable and verifier are
  `../research_experiments/world_foam_lane2/finite_element_material_fit.py`
  and `verify_finite_element_material_fit.py`.
- **`worldfoam_paper/scientist_notes/2026-07-26_gauge_invariant_ray_holonomy_intake_and_paper_split.md`** — full normalized intake for the moving-camera gauge-invariant ray-holonomy proposal. Records the camera-program/gauge distinction, noncommutation theorem, path-ordered transfer, self-normalized convex-potential atoms, one-interval ray theorem, certified discriminant compiler, Duhamel VJP, conditional complexity, red-team branches, and the decision to keep this as a strengthened WorldFoam Paper B rather than absorb it wholesale into STAR UVT Paper A.
- **`blur_dof_motion_paper_review.md`** — finite-aperture / depth-of-field and motion-blur paper review for NeRF and 3DGS, with the renderer-state contract for focus distance, aperture / CoC strength, exposure trajectory, and dynamic-object blur.
- **`video_token_overfit_next_plan.md`** — current tiny dog-clip video-token overfit plan: RGB/init diagnostics, static/dynamic split evidence, V-JEPA feature baseline, missing foreground/motion metrics, and the next experiment queue.
- **`blur_dof_motion_papers/paper_index.md`** — local 55-paper corpus index with downloaded PDFs, extracted text, priorities, tags, and search commands.
- **`gaussian_splatting_papers/paper_index.md`** — local papers on Gaussian-splat rendering/compositing/visibility. Currently includes Softmax-GS with PDF, arXiv source, converted Markdown, DynaWorld integration notes, and the short/long plans for Softmax-GS versus STAR UVT/WorldFoam.
- **`potential_directions_index.md`** — routing map for all research threads. Status labels (Now / Probe / Background / Speculative) per direction. Start here when scoping a new experiment.
- **`synthetic_3d_render_data/`** — synthetic 3D rendering as a *complement* to the real `video <=> video` track, not a substitute. Pipelines we already own (Nova v1/v2/v3 Blender/Unity), tiered scene-source catalog, framework picks (BlenderProc over Kubric), human-motion datasets (BEDLAM etc.), and the four legitimate uses inside DynaWorld's contract (probes, camera-leakage tests, pretraining priors, Nova app features). Read its `README.md` for orientation.
- **`../agent_notes/key_learnings.md`** — dense bank of surprising technical lessons. Tactical, not strategic.
- **`../agent_notes/loose_notes/`** — raw session chronology. Go here when you need the why behind a decision, not just the outcome.

When to use which:

- *New agent, cold start:* read `meta_philosophy/README.md`, then `potential_directions_index.md`, then this file.
- *Proposing an architecture:* run through the checklist in `meta_philosophy/how_to_think_about_architecture.md` before writing anything.
- *Driving an external LLM (ChatGPT Pro, Gemini, etc.):* paste `meta_philosophy/chatgpt_pro_prompt_for_expert_divergent_web_of_thought_model_architecture_development.md` as the system brief and attach the problem doc.
- *Proposing anything new:* start with `framing_the_problem/framing_3.md`. It was patched, not replaced; do not mint `framing_4.md` for the predictive-quotient correction.
- *Implementing the baseline:* use `training_contract_v1.md` for sampler/signature/loss/export details.
- *Working on physical blur / DoF:* start with `blur_dof_motion_paper_review.md`, then use `blur_dof_motion_papers/paper_index.md` and `blur_dof_motion_papers/extraction_notes.md` for local PDFs/text and formula anchors.
- *Auditing an existing architecture for frame-local state leaks:* use framing 2's constraints C1, C3, C5, C6, C7 as the audit checklist.
- *Deriving a new loss:* use framing 1's information-theoretic view.

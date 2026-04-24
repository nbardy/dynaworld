# DynaWorld Architecture Solution Prompt

Reusable prompt for asking a strong reasoning model to propose **3-4 concrete
architecture solutions** for DynaWorld without falling into generic tokenizer,
scene/camera factorization, or "plausible module list" answers.

Use this when the goal is:

```text
Given our philosophy and problem definition, generate candidate architectures
that make non-degenerate world/splat tokens from source videos.
```

This prompt is a narrower, DynaWorld-specific sibling of
`chatgpt_pro_prompt_for_expert_divergent_web_of_thought_model_architecture_development.md`.
It should be used when the bottleneck is representation design, token
definitions, held-out prediction, and splat degeneracy.

Important: do **not** ask the model to expose unrestricted private
chain-of-thought. Ask for compact, auditable rationale fields, branch-local
arguments, failure cases, and falsification tests. The goal is visible
scientific reasoning, not hidden token dumps.

---

## How To Use

1. Paste the **System Brief** below into the model.
2. Paste the **DynaWorld Context Pack** below as the companion document.
3. Add one specific question, e.g.:

```text
Design 3-4 candidate architectures for training non-degenerate world/splat
tokens from single moving-camera source videos. We care most about preventing
source-view billboard splats while preserving maximum expressivity.
```

4. Require the XML output exactly.
5. Reject outputs that:
   - give generic transformer scaffolding,
   - propose splat tokenizers without solving the no-ground-truth-splats
     problem,
   - feed query camera into a learned decoder without defending the leakage
     risk,
   - skip held-out input/target set definitions,
   - lack branch-specific falsification tests.

---

## System Brief

Paste this section verbatim.

```text
You are an expert model researcher designing architectures for DynaWorld.

You are not in brainstorming-list mode. You are in constraint-satisfaction
architecture mode. Your job is to propose 3-4 concrete candidate solutions,
critique them, and synthesize a recommendation.

Do not reveal private chain-of-thought. Instead, provide compact, auditable
rationale fields before each architecture: what invariant the design targets,
why the supervision reaches that invariant, what degeneracy it rules out, and
what would falsify it.

The target system turns source video observations into an exported dynamic
world/splat asset. The primary failure to avoid is source-view-degenerate
splats: splats that render the input camera well but fail as persistent 3D/4D
world structure under novel or held-out queries.

You must reason from the provided DynaWorld Context Pack. Treat it as
load-bearing. If you disagree with a premise, say so explicitly in the
`<premise_challenges>` section, but do not silently ignore it.

Your output must contain 3-4 architecture branches. Each branch must be a real
solution, not a parameter variant. Across the set, cover at least three
different mechanism families:

- predictive-objective / held-out observation contract
- augmentation / variable-budget data support
- architectural-seam / representation-boundary design
- two-stage tokenizer or amortization, but only if non-degeneracy is solved
- post-training adversarial/diffusion prior
- emergent/minimal-structure scaling bet

Do not force all families into the final recommendation. Branch for coverage;
synthesize for commitment.

Strict output hygiene:

- Return valid, parseable XML only. The first character must be `<` and the
  last character must be `>`.
- Do not add Markdown fences, headings, bullets, commentary, or separators
  outside the XML.
- Do not wrap any branch or pseudocode in triple backticks.
- Do not include citations, footnotes, `:contentReference[...]` markers, source
  annotations, or URLs unless explicitly requested.
- Do not include raw control characters or terminal escape sequences.
- Use plain ASCII punctuation.
- If pseudocode contains `<`, `>`, or `&`, place it inside CDATA:
  `<pseudocode><![CDATA[...]]></pseudocode>`.
- Before finalizing, check tag balance and repair malformed XML.
```

---

## DynaWorld Context Pack

Paste this section after the system brief.

```text
# DynaWorld Problem Definition

DynaWorld is a video <-> splats modality shift for camera-path editing and
filmmaking. The deployed system should accept source video observations and
export a compact dynamic 3D/4D Gaussian splat world that can be rendered from
user-specified cameras and times.

The MVP use case is not text-to-world generation. It is:

    source video -> dynamic splat world -> edit camera path -> rendered video

Video diffusion is complementary. It may refine final pixels or provide a
post-training prior, but the DynaWorld asset should remain an exploratory,
renderable world representation.

# Core Deployment Contract

The clean deployed function is:

    W       = E(O)
    S_tau   = G(W, tau_q)
    image_q = R_fixed(S_tau, camera_q)

where:

    O          = observation set shown to the encoder
    W          = exported world asset / world tokens
    tau_q      = query time
    camera_q   = query camera
    S_tau      = time-conditioned splat tokens / Gaussian parameters
    R_fixed    = fixed differentiable splat rasterizer

Rules:

    - Observation camera/time may enter encoder ingestion.
    - Query time may enter G(W, tau_q).
    - Query camera should enter only R_fixed by default.
    - Deployed rendering reads only W, tau_q, camera_q, fixed weights, and the
      fixed rasterizer.
    - No source frames, target frames, encoder caches, teacher state, or
      frame-local rescue tensors at render time.

# Training Signal

We do not have ground-truth splats or ground-truth world tokens.

The scalable supervision is video pixels:

    render predicted asset under query camera/time
    compare rendered light to real video frames

Therefore splat/world tokens are latent predictive assets, not labels.

The baseline training sample is:

    V = {(I_i, camera_i, tau_i)}
    O = input observations the encoder sees
    H = held-out target observations, H not_subset O
    Q = {(camera_h, tau_h) for h in H}

Training objective:

    W       = E(O)
    S_h     = G(W, tau_h)
    I_hat_h = R_fixed(S_h, camera_h)
    L_pred  = sum_h ell(I_hat_h, I_h)
    L       = L_pred + beta * Rate(W)

Hard rule:

    Exact target observations must not be visible to the encoder.

Same source time is allowed only if the query is genuinely different, e.g.
different crop/intrinsics/camera and target pixels were not already visible.

# Token Philosophy

Text/image/video tokenizers tokenize observed data. Splats/worlds are not
observed. A splat tokenizer trained as a same-view autoencoder can learn
source-view billboards. Training a predictor to imitate those tokens only
distills the failure.

Definitions:

    Observation tokens:
        Input-side image/ray/time evidence. Allowed to contain observation
        camera and time. Not exported.

    World tokens W:
        Exported predictive asset. Abstract allowed. Must be self-sufficient
        for render under query time/camera through fixed generator/rasterizer.

    Splat tokens S_tau:
        Typed renderer-facing commitments derived from W and tau_q. Eventually
        become Gaussian parameters: xyz, rotation, scale/covariance, opacity,
        color/features, time basis/residual.

    Memory tokens:
        Valid only if exported, re-derived from W, or part of streaming asset
        state. Invalid if hidden frame-local rescue.

    Camera tokens:
        Observation-side camera estimates are allowed. Query camera belongs at
        the fixed rasterizer boundary by default.

The recommended split is not "world tokens OR splat tokens":

    W_slots = E(O)
    S_slots_tau = G_slots(W_slots, tau_q)
    GaussianParams_tau = Heads(S_slots_tau)
    image = R_fixed(GaussianParams_tau, camera_q)

One exported W, typed time-conditioned splat readout.

# Primary Degeneracy

The biggest failure is:

    The model emits splats that render the source camera well but do not
    represent persistent 3D/4D world structure.

Variants:

    - source-view billboards
    - camera-conditioned splat/image decoder
    - source camera stored in W and cancelled elsewhere
    - frame-local caches outside W
    - dynamic billboard tracks over time
    - tokenizer trained to freeze a degenerate splat code

Batch-level red-team question:

    Could a source-view billboard pass this training step?

If yes, the training step is weak for non-degeneracy.

# Source Video Setting

We want this to work on single source videos, ideally with moving cameras.
Moving camera helps because held-out frames have different camera/time queries
and provide parallax. But a single moving-camera clip still does not give true
same-time multi-view supervision for dynamic objects. Claims must be qualified
by query support.

Useful D_var budgets:

    - sparse O -> nearby held-out H
    - rich O -> sparse held-out H
    - early frames -> later held-out frames
    - odd frames -> even held-out frames
    - cropped/ray-shifted O -> uncropped or differently cropped H
    - multi-view or synthetic observations when available

# Current Default Philosophy

Start with the simplest architecture + objective whose supervision mechanism
reaches the invariants the data cannot directly constrain.

Framing 3 default:

    one encoder
    one exported world asset
    time-conditioned scene/splat generator
    fixed rasterizer
    held-out omitted-observation supervision under D_var
    rate/minimality to choose a useful representative

Auxiliary losses and mechanisms should not ship unless they:

    - encode unavoidable physics or the inference contract,
    - select a lower-rate representative,
    - are diagnostics with no gradient,
    - are measured optimization accelerators with a retirement condition,
    - or are explicit post-training priors.

# Diagnostics

Every proposal should include how it handles these:

    Export-purity test:
        render with only W, camera_q, tau_q, fixed weights. Null all hidden
        training-time tensors.

    Camera-leakage probe:
        freeze W and train a probe to recover source camera/crop/focal.

    World-dependence matrix:
        token drop, token shuffle, zero time basis, wrong-world swap.

    Billboard stress test:
        compare against explicit source-view billboard baseline on source and
        held-out query support.

    Rate-distortion by query type:
        source camera vs held-out time/crop/camera support.

# Known Failure Modes

F1 camera leakage into world/image tokens.
F2 cheating splats: source-view-good, novel-view-bad.
F3 trajectory-geometry ambiguity in implicit camera mode.
F4 long-horizon drift.
F5 latent cheating: hidden memory/features carry geometry instead of splats.
F6 low-rank motion assumptions break for water, smoke, crowds, topology change.
F7 agreement-collapse: student/teacher token agreement can collapse without
   photometric anchor.

# Forbidden Moves

Do not:

    - propose a splat tokenizer as if ground-truth splat tokens exist,
    - rely on same-view autoencoding as proof of world tokens,
    - feed query camera into learned splat generation without a rigorous
      anti-degeneracy argument,
    - use raw token L2 across non-canonical slot sets as the primary target,
    - introduce a second AR/view generator stage that patches broken W,
    - stack weak losses without saying which degeneracy each rules out,
    - answer with generic ViT + cross-attention + MLP heads,
    - cite papers or claims that are not verified,
    - claim arbitrary novel-view correctness outside Q_train support.
```

---

## User Question Template

Paste after the context pack and edit the bracketed parts.

```text
QUESTION:

Design 3-4 candidate architectures for [specific target].

Specific target:
    [Example: training non-degenerate world/splat tokens from single moving-
    camera source videos using held-out frame supervision.]

Main bottleneck:
    [Example: source-view billboard splats can satisfy same-view loss.]

Must address:
    - how input observations O are encoded,
    - how query time tau_q conditions the model,
    - where query camera camera_q enters,
    - what token primitives exist and what each is allowed to contain,
    - how H not_subset O is enforced,
    - why source-view billboard solutions fail,
    - whether the design is single-stage or two-stage,
    - if two-stage, why the tokenizer target is non-degenerate,
    - what rate/minimality pressure is used,
    - what diagnostics would falsify the branch.

Output exactly the XML schema requested below. Do not include prose outside XML.
```

---

## Required XML Output

Paste this after the user question.

Strictly require **well-formed XML**. The model must not include Markdown code
fences inside the returned XML. Any pseudocode should be either plain text or
wrapped in CDATA.

```xml
<dynaworld_architecture_exploration>

  <task>
    <goal>One sentence.</goal>
    <bottleneck>Specific degeneracy or architecture uncertainty being solved.</bottleneck>
    <answering_right_question>Yes/no plus compact justification.</answering_right_question>
  </task>

  <premise_challenges>
    <challenge>
      <premise>Any provided premise the model thinks may be wrong. If none, say none.</premise>
      <why_it_matters>What architecture choice would change if this premise is wrong.</why_it_matters>
    </challenge>
  </premise_challenges>

  <shared_definitions>
    <deployed_signature>Explicit E/G/R signature.</deployed_signature>
    <token_primitives>
      <primitive name="observation_tokens">
        <allowed_information>...</allowed_information>
        <forbidden_information>...</forbidden_information>
      </primitive>
      <primitive name="world_tokens">
        <allowed_information>...</allowed_information>
        <forbidden_information>...</forbidden_information>
      </primitive>
      <primitive name="splat_tokens">
        <allowed_information>...</allowed_information>
        <forbidden_information>...</forbidden_information>
      </primitive>
      <primitive name="memory_tokens">
        <allowed_information>...</allowed_information>
        <forbidden_information>...</forbidden_information>
      </primitive>
      <primitive name="camera_tokens">
        <allowed_information>...</allowed_information>
        <forbidden_information>...</forbidden_information>
      </primitive>
    </token_primitives>
    <query_support>
      <q_train>What the proposed training distribution supports.</q_train>
      <q_deploy>What deployment wants.</q_deploy>
      <unsupported_claims>What must not be claimed.</unsupported_claims>
    </query_support>
  </shared_definitions>

  <architecture_branches count="3-4">

    <branch id="A">
      <rationale_summary>
        Compact branch-local reasoning. What invariant this branch targets and why this mechanism should reach it.
      </rationale_summary>
      <mechanism_family>
        predictive-objective | augmentation | architectural-seam | tokenizer-two-stage | post-training | emergent | hybrid
      </mechanism_family>
      <name>Short descriptive name.</name>
      <core_hypothesis>...</core_hypothesis>
      <architecture>
        <input_encoding>How O becomes observation tokens and W.</input_encoding>
        <world_representation>Shape/role of W.</world_representation>
        <splat_readout>How W and tau_q become S_tau/Gaussian params.</splat_readout>
        <time_conditioning>Where tau_i and tau_q enter.</time_conditioning>
        <camera_boundary>Where observation camera and query camera enter.</camera_boundary>
        <memory_or_streaming_state>Whether memory exists and how export purity is preserved.</memory_or_streaming_state>
      </architecture>
      <training_contract>
        <sampler>Define O, H, Q and budgets.</sampler>
        <losses>List only losses that earn their cost. Mark diagnostics separately.</losses>
        <rate_minimality>How W capacity/rate is constrained.</rate_minimality>
        <pseudocode><![CDATA[Small pseudocode block for the training step.]]></pseudocode>
      </training_contract>
      <anti_degeneracy_argument>
        <billboard_test>Why a source-view billboard fails or where it still passes.</billboard_test>
        <camera_leakage>How F1 is prevented, penalized, or diagnosed.</camera_leakage>
        <frame_local_state>How hidden frame-local rescue is forbidden.</frame_local_state>
        <dynamic_degeneracy>How dynamic billboard tracks or time collapse are handled.</dynamic_degeneracy>
      </anti_degeneracy_argument>
      <what_it_rules_out>
        <failure_mode>Concrete solution family removed from the hypothesis space.</failure_mode>
      </what_it_rules_out>
      <what_it_does_not_rule_out>
        <failure_mode>Concrete remaining failure.</failure_mode>
      </what_it_does_not_rule_out>
      <costs>
        <compute>...</compute>
        <data>...</data>
        <implementation_complexity>...</implementation_complexity>
      </costs>
      <falsification_tests>
        <test>
          <measurement>...</measurement>
          <supporting_result>...</supporting_result>
          <falsifying_result>...</falsifying_result>
          <backtrack_if_falsified>...</backtrack_if_falsified>
        </test>
      </falsification_tests>
    </branch>

    <!-- Repeat for 3-4 total branches. Branches must differ structurally, not by hyperparameter only. -->

  </architecture_branches>

  <cross_branch_comparison>
    <axis name="single_stage_vs_two_stage">...</axis>
    <axis name="where_non_degeneracy_pressure_lives">...</axis>
    <axis name="expressivity_vs_export_purity">...</axis>
    <axis name="camera_support_required">...</axis>
    <axis name="optimization_risk">...</axis>
  </cross_branch_comparison>

  <synthesis>
    <recommended_branch_id>Pick one branch, or one branch plus a clearly bounded diagnostic/post-training add-on.</recommended_branch_id>
    <why>
      Compact justification referencing training signal, unavoidable structure, deployment contract, and degeneracy prevention.
    </why>
    <what_this_gives_up>...</what_this_gives_up>
    <minimum_viable_experiment>Smallest experiment to decide whether the recommendation is right.</minimum_viable_experiment>
    <tripwires>
      <tripwire>Concrete signal that means stop and backtrack.</tripwire>
    </tripwires>
  </synthesis>

  <final_architecture_spec>
    <tokens>Final recommended token primitives and shapes at a high level.</tokens>
    <forward_pass>Step-by-step forward pass.</forward_pass>
    <training_step>Step-by-step training step.</training_step>
    <diagnostics>Diagnostics to run from day one.</diagnostics>
    <open_questions>Questions not settled by this response.</open_questions>
  </final_architecture_spec>

  <self_audit>
    <possible_pattern_matching>Where the answer may have imported generic habits.</possible_pattern_matching>
    <weakest_claim>Most load-bearing claim that needs evidence.</weakest_claim>
    <missing_information>Information that would materially improve the design.</missing_information>
  </self_audit>

</dynaworld_architecture_exploration>
```

---

## Follow-Up Prompt For Bad Outputs

Use when the model gives generic or weak answers.

```text
Redo. The response failed the DynaWorld contract.

Specific failures:
    - [list failures]

You must:
    - define O, H, Q for every branch,
    - state whether H is excluded from O,
    - explain why a source-view billboard fails,
    - define world tokens vs splat tokens vs observation tokens,
    - keep query camera out of learned splat generation unless explicitly
      defended,
    - separate training losses from diagnostics,
    - give 3-4 structurally different branches,
    - synthesize to one recommendation.
    - return well-formed XML only,
    - remove all Markdown fences, citations, contentReference markers, and
      malformed/control characters.

Return only the XML schema.
```

---

## Reviewer Checklist

After receiving the XML, reject or send back if:

- The architecture can render without using `W`.
- The tokenizer branch has no answer to "where do non-degenerate token targets
  come from?"
- Query camera enters a learned module before `R_fixed` and the branch does not
  prove why that does not reintroduce view-conditioned billboards.
- The branch uses token equality for non-canonical splat/world slots.
- No branch can explain why held-out frames help on single moving-camera clips.
- The recommendation combines every mechanism without explaining why the added
  complexity earns its cost.
- Diagnostics are promoted into permanent gradient losses without admission
  criteria or retirement conditions.

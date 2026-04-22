# How To Think About DynaWorld Architecture

Debug log of predictable slop patterns observed in agent responses while
working on this project. Each entry names a mistake I (or another agent)
actually made, and the principle that mistake violated.

Read this **before** proposing architecture, evaluating a paper, or
responding to a scientist's suggestion. These mistakes are regressions to
familiar-shaped bad answers. They're easy to repeat.

## Mistakes (named, with the principle they broke)

1. **Pre-committed to factorization as *the* supervision mechanism.**
   Shipped "simplest architecture whose factorization seams sit where data
   cannot supervise" as a north star. Forced every proposal into the
   architectural-seam family.
   - Principle: *supervision mechanism is a family axis, not a singleton.*
     Families: objective, augmentation, architectural-seam, post-training,
     emergent-at-scale. Factorization is one. A clean objective on an
     unfactored model often beats a factored model with a trivial objective.

2. **Treated "camera parameterization" as a real design axis.**
   Listed 6DoF vs Plücker vs quaternion+t as if the choice mattered.
   - Principle: *parameterization is a bikeshed. Commitment timing is the
     real axis.* For any learned token, the design choice is where it
     commits from latent to explicit. Cameras and splats commit at the
     rasterizer boundary. Anything earlier forces the rest of the network
     to reason over explicit reps unnecessarily.

3. **Conflated "AR drift" with "query-distribution shift."**
   Called novel-view generalization "AR drift" when there is no AR loop in
   the MVP.
   - Principle: *novel views are train/inference distribution shift on the
     decoder query, not temporal AR.* Forcing-family training applies to
     both — same mechanism, different axis — but name the axis correctly.

4. **Split forcing and novel-view as different problems.**
   Dismissed self-forcing / rolling-forcing as irrelevant to novel views.
   - Principle: *if two problems share a mechanism, they are the same
     problem.* Both are train/inference gaps. If the rollout perturbs
     cameras, forcing supervises novel views directly.

5. **Overstated "parameter space isn't rotationally symmetric" as a
   diffusion blocker.**
   Claimed rasterizer fragility was why diffusion-on-splats fails.
   - Principle: *don't invent blockers.* The real blocker is that splats
     are derived from tokens — no x_0 to noise. xyz, logit-opacity, RGB
     tolerate Gaussian noise fine. Scales and rotations need a
     parameterization, not an abandonment.

6. **Separated renderer work from model work.**
   Treated kernel tuning and novel-view architecture as independent concerns.
   - Principle: *every failure mode in this project traces back to "what
     happens when the model is evaluated under a condition it wasn't
     optimized for."* Wide-depth z, 128px NaN, small-init, novel-view,
     opacity saturation — one axis. Renderer envelope and model
     representation design are coupled.

7. **Called self-forcing and rolling-forcing unrelated.**
   Labeled them "unrelated papers glued into a narrative."
   - Principle: *check citations before calling papers unrelated.* They
     cite each other — one research thread, not pattern-matched.

8. **Suggested a separate AR generator stage.**
   Proposed two-stage "frozen world tokens → AR novel-view generator" as
   a candidate.
   - Principle: *two-stage designs inherit stage-1 leakage.* Freezing a
     broken representation enshrines the break. Stage 2 either expensively
     patches it or ignores it and becomes a standalone model. One-stage
     with a loss on the true boundary is usually cleaner.

9. **Sloppy about on-manifold vs averaging losses.**
   Treated L1/MSE/LPIPS/SSIM as if they supervised plausibility.
   - Principle: *averaging losses measure distance to a target, not
     likelihood under the data distribution.* Off-manifold supervision
     requires score-based (diffusion), adversarial, or contrastive losses.
     Averaging losses let the model hedge to the mean on hard multi-modal
     data.

10. **Suggested camera as an input token.**
    Concatenated camera to video patch tokens in one pioneer iteration,
    after user had already said camera must be emergent.
    - Principle: *re-read the user's constraints on each pass.* User
      explicitly wants camera predicted, not supplied. Camera lives as an
      output of the encoder, grounded only through rendering. At inference
      the user supplies the edit-path camera; predicted cameras are
      discarded or used as trajectory origin.

11. **Listed ideas as a flat menu.**
    Early "ideas surfaced" section had factorization, augmentation, and
    objective candidates in one undifferentiated list.
    - Principle: *index ideas by supervision-mechanism family.* Mechanism
      family is the useful axis for judging compose-vs-substitute.

12. **Answered the wrong question.**
    Gave a long critique of an unposed-SLAM framing when the user's
    bottleneck is novel-view from posed video.
    - Principle: *check whether the target matches the user's bottleneck
      before engaging.* If it doesn't, say so and reframe. Named in the
      prompt template as the `<am_i_answering_the_right_question>` step;
      do it every time.

13. **Dropped insights surfaced earlier in the same thread.**
    Forgot the "static and dynamic are the same problem" core belief and
    proposed temporal-evolution decoders as if static were solved.
    - Principle: *the project's core beliefs (in the dynaworld README) are
      load-bearing.* Re-check them before proposing anything that looks
      like it recomposes the problem.

14. **Claimed an encoder could be "blind" to camera by not feeding camera
    as input.**
    Proposed a "Blind World Encoder" architecture whose primary mechanism
    was input-denial for camera. User pointed out: the encoder sees
    video, and video contains camera implicitly (motion parallax,
    perspective, vanishing points). Inferring camera from video is an
    easy task; denying the explicit channel does not prevent inference.
    - Principle: *information bottleneck via input denial does not work
      when the input is a rich signal from which the hidden variable is
      easily derivable.* For video-input models, there is no way to hide
      camera from the encoder without destroying the video signal
      itself. The only real leakage-prevention mechanisms are active
      pressure downstream (adversarial recovery, multi-condition
      consistency, augmentation invariance), not encoder-side denial.
      Capacity asymmetry is a routing hope, not a bottleneck.

15. **Imported a product-management frame into research work.**
    Pushed hard for evaluation protocol, candidate scorecard, tripwires,
    and experimental cadence as "Tier 1 blockers" when the user was
    doing theoretical research. User corrected: "calm down on pushing
    for evals you're not a manager, this is research. it's easy to
    write validation scores, we need theory of information systems atm."
    - Principle: *research bottleneck is understanding, not measurement.*
      In research mode, you write theory before you write evals, because
      you don't know what to measure until you know what the information
      system is doing. Ship-mode artifacts (scorecards, tripwires,
      experimental protocols) are cheap to write once the theory is
      settled; writing them first anchors the work on the wrong axis.
      - How to apply: if the user is working through a framing problem
        (what does the representation *mean*), do not pivot to "here's
        how we'd score candidates." Stay in the framing layer until the
        user signals they are ready to operationalize. Evaluation is
        downstream of theory.

16. **Treated noise alone as generative pressure.**
    Proposed rolling-noise, dropout, or denoising regularizers over
    visible frames with visible targets as if they enforced generative
    reconstruction.
    - Principle: *observation omission is the supervision move, not
      noise.* If the encoder sees `O` and the target is `O` (even
      corrupted), the model can still learn to copy. Generative pressure
      requires target observations the encoder did not see. Noise is an
      augmentation; omission is the contract.
      - How to apply: for each proposed loss term, identify input-set
        and target-set. If target ⊆ input, the term is autoencoding and
        does not earn its novel-view budget. See framing_2 philosophy
        bullet 9 and constraint C9.

17. **Bolted on a student-teacher agreement loss without a photometric anchor.**
    Proposed `L_agree(W_s, sg(W_r))` or similar distillation as the
    primary supervision for self-sufficiency.
    - Principle: *agreement losses have a collapse mode.* Student and
      teacher can both drift to a constant `W` that satisfies agreement
      with zero scene content — this is F7 from the problem doc. An
      agreement term only earns its cost when it sits on top of a
      strong photometric anchor on held-out real frames, with bounded
      `λ_agree` and (optionally) an EMA teacher or diversity term.
      Prefer render-agreement over token-L2 (framing_2 philosophy 11).
    - Successor framing: under patched framing_3, token/world agreement is
      replaced by predictive agreement modulo gauge on supported queries,
      plus rate/minimality for representative selection. It should not be
      added as a baseline gradient source. Keep the principle here for
      the case where someone proposes it again.

18. **Reached for a mechanism before checking the predictive quotient.**
    Proposed an agreement / adversarial / cross-view / factorization
    mechanism without first asking "is this target already covered by
    supported-query prediction, rate/minimality, or the export contract?"
    - Principle: *predictive quotient before mechanism.* Patched
      framing_3: the held-out predictive loss identifies behavior only
      on supported `(c, τ)` queries, modulo gauge/reparameterization.
      If the invariant you want to enforce has no consequence beyond
      supported-query rendering behavior, a dedicated semantic loss is
      decorative and does not ship. If the issue is choosing among
      behavior-equivalent worlds, use rate/minimality rather than a new
      invariant loss.
      - How to apply: before writing a new head, loss, or stage, run
        the framing_3 evaluation checklist. If the target invariant is
        already covered by supported-query prediction or representative
        pressure, demote the mechanism to a diagnostic probe (fits
        post-hoc, no gradient) rather than a training signal.

19. **Trusted fabricated citations from exploration-style LLM outputs.**
    Treated 2025–2026 paper citations returned by a divergent-thinking
    XML prompt (Gemini, ChatGPT Pro, etc.) as real prior art without
    verification. Built framing around them.
    - Principle: *verify every citation before admitting it.* The
      divergent XML driver encourages plausible-sounding future-tense
      citations. Confirmed hallucinations in one session: "V-JEPA 2.1",
      "Lyra 2.0 (Shen 2026)", "PERSIST (Garcin 2026)", "Relax Forcing
      (Zhao 2026)", "Causal Forcing (Zhu 2026)", "TokenGS (Ren 2026)".
      See `../../agent_notes/loose_notes/2026-04-22_13-26-00_gemini_exploration_audit.md`
      for the full audit.
      - How to apply: any paper cited with a 2026 date, or with no
        arXiv ID, gets an independent search before the claim enters
        any framing doc. If it cannot be verified in one search, flag
        it as unverified in-line (never silently drop the uncertainty).

20. **Wrote a loss term whose first proposed ablation is to remove it.**
    Specced a training objective (e.g., `L_agree` agreement term,
    `L_cam` crop-seam cross-render, `L_motion` weak prior) then
    immediately wrote "the first ablation I'd run: remove these
    losses." That is a tell.
    - Principle: *if the author's own sanity check is "does this term
      matter," the term failed the predictive-quotient check before it
      was written.* Under framing_3, a loss ships only if (a) its target
      is not covered by supported-query prediction plus rate/minimality,
      (b) it encodes unavoidable physics or the inference/export
      contract, or (c) it's a decaying-to-zero accelerator justified by
      measurement. A term whose first ablation is removal matches none
      of these; it is a mechanism stacked on a hope.
      - How to apply: before adding any loss, write the ablation
        result you expect. If the expected result is "model works
        about as well without it," do not add it. Admit it as a
        diagnostic probe (no gradient) if you want to measure the
        property. See framing_3 evaluation checklist + tripwires.

## Checklist before proposing

Run through these before saying "here's an architecture" or "here's a fix":

1. **Predictive quotient first.** Is the invariant this proposal targets
   already covered by framing_3's supported-query predictive objective,
   rate/minimality pressure, or export contract? If yes, drop or demote
   to diagnostic. If no, continue.
2. What invariant needs to be supervised?
3. Does the data-loss contract cover that invariant, or is it one of the
   axes the data cannot reach?
4. Which mechanism family supplies the cheapest valid supervision for that
   invariant? (objective / augmentation / seam / post-training / emergent)
5. Does the proposal pre-commit to factorization, or does it leave room for
   non-factored solutions?
6. For every learned token: where does it commit? If it never commits, where
   does its supervision hook live?
7. If the proposal is two-stage, does stage 1 enforce the invariants the
   whole pipeline needs, or does it silently violate them?
8. Is the loss averaging or on-manifold? If averaging, does the task
   tolerate blurry means?
9. Does the proposal answer the user's actual bottleneck, or a different one?
10. What class of failures does it rule out? (Use F1–F7 from the problem
    doc.) If none, why is it adding complexity?
11. What would it look like at 10x, 100x, 1000x scale? Does it compose with
    scale or fight it?
12. **Verified citations.** If the proposal cites papers, has every
    citation been independently verified? Suspect 2026-dated or
    arXiv-IDless citations until confirmed.
13. **Retirement condition.** If the proposal ships as an escape hatch
    (framing_3 EH-*), does it have a concrete trigger and retirement
    condition, or is it being smuggled in as permanent infrastructure?

## Failure-mode lookup (F1–F7 from the problem doc)

Before defending a proposal, label which of F1–F7 it rules out.

- F1 camera leakage into scene/image tokens
- F2 cheating splats (overfit to encoded view)
- F3 trajectory-geometry ambiguity (implicit camera mode)
- F4 long-horizon drift
- F5 latent cheating (scene memory absorbs geometry)
- F6 low-rank motion assumption breaks
- F7 world-agreement collapse (student-teacher distillation without anchor)

If a proposal rules out zero of them, it is not earning its cost.

## Meta-rules

- This doc is append-only for new mistake classes. Edit existing entries to
  refine the principle, don't delete them.
- A new mistake pattern noticed in a future session should add an entry
  here, not start a fresh doc.
- If a checklist item in this doc starts feeling pro-forma, rewrite it.
  Checklists that are ignored are worse than no checklist.
- The right heuristic when uncertain: *re-read the problem doc and the
  north star* before generating a new branch. Slop comes from forgetting
  constraints, not from missing ideas.

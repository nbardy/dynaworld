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

## Checklist before proposing

Run through these before saying "here's an architecture" or "here's a fix":

1. What invariant needs to be supervised?
2. Does the data-loss contract cover that invariant, or is it one of the
   axes the data cannot reach?
3. Which mechanism family supplies the cheapest valid supervision for that
   invariant? (objective / augmentation / seam / post-training / emergent)
4. Does the proposal pre-commit to factorization, or does it leave room for
   non-factored solutions?
5. For every learned token: where does it commit? If it never commits, where
   does its supervision hook live?
6. If the proposal is two-stage, does stage 1 enforce the invariants the
   whole pipeline needs, or does it silently violate them?
7. Is the loss averaging or on-manifold? If averaging, does the task
   tolerate blurry means?
8. Does the proposal answer the user's actual bottleneck, or a different one?
9. What class of failures does it rule out? (Use F1–F6 from the problem
   doc.) If none, why is it adding complexity?
10. What would it look like at 10x, 100x, 1000x scale? Does it compose with
    scale or fight it?

## Failure-mode lookup (F1–F6 from the problem doc)

Before defending a proposal, label which of F1–F6 it rules out.

- F1 camera leakage into scene/image tokens
- F2 cheating splats (overfit to encoded view)
- F3 trajectory-geometry ambiguity (implicit camera mode)
- F4 long-horizon drift
- F5 latent cheating (scene memory absorbs geometry)
- F6 low-rank motion assumption breaks

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

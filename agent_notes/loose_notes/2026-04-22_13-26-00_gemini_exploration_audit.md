# Gemini XML-exploration audit — what to trust, what to discount

Session on 2026-04-22. Ran the divergent-thinking XML driver prompt
(from `research_notes/meta_philosophy/chatgpt_pro_prompt...`) against
Gemini across three iterations. Each iteration returned ~6 branches in
the schema, with claims about various forcing / world-model / diffusion
papers. Audit below.

This note exists so a future session running the same prompt does not
re-derive the audit and does not anchor on the slop.

## The slop pattern

**Fabricated / unverifiable citations.** Across three passes, Gemini
cited the following papers with recent (2025–2026) dates:

- V-JEPA 2.1 (Mur-Labadia et al., 2025)
- Lyra 2.0: Explorable Generative 3D Worlds (Shen et al., 2026)
- PERSIST: Beyond Pixel Histories: World Models with Persistent 3D State
  (Garcin et al., 2026)
- Relax Forcing: Relaxed KV-Memory for Consistent Long Video Generation
  (Zhao et al., 2026)
- Reward Forcing (Zhang et al., 2025)
- Causal Forcing: Autoregressive Diffusion Distillation Done Right
  (Zhu et al., 2026)
- ReconPhys: Reconstruct Appearance and Physical Attributes from Single
  Video (Wang et al., 2026)
- TokenGS: Decoupling 3D Gaussian Prediction from Pixels with Learnable
  Tokens (Ren et al., 2026)
- 4D Latent World Model (OpenReview, 2024)
- EMERALD: Efficient MaskEd latent tRAnsformer worLD model

Some of these are real papers, some are plausible extrapolations, some
are likely hallucinated. Treat **any 2026 citation as suspect by
default**; any paper with a plausible-sounding title attributed to a
specific author with no arXiv ID is suspect.

**Verified real** (these stood up to spot-checks):

- Geometry Forcing (Wu et al. 2025, arXiv 2507.07982) — aligns video
  diffusion features with VGGT. EH-1 in framing_3.
- Diffusion Forcing (Chen et al. 2024) — per-token independent noise
  schedules.
- Self-Forcing (Huang et al. 2025, TencentARC) — closes train/inference
  exposure bias for AR rollout.
- Rolling Forcing (TencentARC) — sliding-window denoising + attention
  sink for long-horizon AR.
- Score Jacobian Chaining / DiffRep / "Diffusing Differentiable
  Representations" (Wang et al. 2022; Savani et al. 2024) — rigorous
  pullback of 2D diffusion scores through a differentiable renderer.
  EH-2 math in framing_3.
- V-JEPA 2 (Meta, 2024). The "2.1" version is not verified; treat the
  underlying technique as real and any "2.1" claims as potentially
  hallucinated specifics.
- WVD — World-Consistent Video Diffusion with Explicit 3D Modeling
  (Zhang et al. 2025) — generates joint RGB + XYZ. Real; silhouette
  comparator.
- DGS-LRM (Lin et al. 2025, arXiv 2506.09997). Real.

## The mechanism-stacking bias

Across every iteration, Gemini's synthesis converged on:

- Freeze a TokenGS-style encoder/decoder (stage 1)
- Train a V-JEPA-style latent transition model on frozen tokens
  (stage 2)
- Add Self-Forcing on the rollout
- Add Geometry Forcing / VGGT alignment as auxiliary
- Optionally add SJC / diffusion-as-loss

This is explicitly **mistake #8** ("suggested a separate AR generator
stage") from `how_to_think_about_architecture.md`. Two-stage designs
with a frozen stage 1 inherit whatever stage 1 got wrong. Every
iteration of the prompt re-proposed this trap. Gemini does not apply
the bitter-lesson filter; it stacks mechanisms instead of asking which
are subsumed by the primary loss.

**Rule for future sessions:** the XML exploration driver will, by
default, return mechanism stacks. Apply framing_3's subsumption
principle to every returned branch before adopting anything.

## Real extractable ideas

Worth keeping from the output:

1. **Geometry Forcing as a warmup auxiliary.** Real paper, real
   technique (angular + scale alignment against VGGT features). Admitted
   in framing_3 as EH-1 with a decay schedule.
2. **Score Jacobian Chaining over naive SDS.** If we ever do diffusion-
   as-loss (EH-2), the pullback formulation is the mathematically
   correct version. Not naive SDS.
3. **WVD's joint RGB + XYZ generation.** Principle — emit explicit 3D
   alongside pixels to bind them — informs decoder design even though
   we do not adopt WVD's architecture.
4. **Diffusion Forcing's per-token independent noise schedules.** If
   we ever do diffusion over W (regime 3 / EH-3), the formulation is
   worth borrowing. Not used in the baseline.

## Ideas explicitly rejected (under framing_3)

- Two-stage freeze-then-AR training. Mistake #8.
- Dense voxel world-frames (PERSIST-style). VRAM-unbounded; we use
  world tokens.
- Implicit MLP world (test-time token tuning). Too slow for real-time
  streaming; the meta-learned-optimizer escape is speculative.
- Self-Forcing / Rolling Forcing as load-bearing in the baseline. AR
  over rolling state is not the contract; they are relevant only if
  the project later moves in that direction.
- JEPA agreement on `W` as a dedicated loss. Subsumed by the primary
  photometric loss via transitivity (framing_3 subsumption theorem);
  adding it invites F7 without benefit.

## Operating rules for next time

When running the XML exploration prompt against any external LLM:

1. Before extracting a branch as an idea, verify every citation with
   an independent search. Papers with 2026 dates get extra scrutiny.
2. Apply the framing_3 evaluation checklist to every branch: does it
   encode unavoidable physics / the inference contract? Is the target
   invariant subsumed by `L`? Is it diagnostic-only? Is it a decaying
   accelerator?
3. If the branch passes the checklist, record it as a potential
   escape hatch with trigger + retirement condition. Never as a
   permanent baseline component.
4. If the branch fails, record *why* in this file (or its successor),
   so the next session knows it was considered and rejected.

## One-line takeaway

> Gemini's XML exploration tends toward mechanism stacking, two-stage
> freezing, and fabricated 2026 citations. Real extractable ideas
> (Geometry Forcing, SJC, Diffusion Forcing, WVD) are worth keeping;
> the synthesis pattern is not. Apply framing_3 subsumption before
> trusting anything.

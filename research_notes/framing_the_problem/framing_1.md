# Framing 1 — Information-Systems View of the Novel-View Problem

Fair. Switching modes.

Attempting theory. Offering this as a framing to argue with, not a conclusion.

## The model as a channel

The architecture is a channel composed of sub-channels:

```
V ──[encode]──► (W, C, T) ──[decode]──► S(C_q, t_q) ──[render]──► I(C_q, t_q)
```

Where V is video, W is world tokens, C/T are camera/time tokens, S is the splat set, I is the rendered image. Each arrow is an information channel with its own fidelity and distortion properties.

The desired properties are not "loss goes down" but:

- **I(V; W) is high** — W carries the scene content.
- **I(W; C_true) ≈ 0** — W does not carry the encode-time camera. This is the leakage property.
- **I(W; T_true) ≈ 0** — W is time-invariant identity, not time-conditioned state.
- **W is a sufficient statistic** for predicting V_at_t given the query (C_q, t_q). Formally, P(V_at_t | W, C_q, t_q) = P(V_at_t | scene_true, C_q, t_q).
- **The channel is C-equivariant**: an intervention on C_q at decode time produces the render-from-a-different-camera transformation, not some other arbitrary function.

Note that 2 and 4 together make W a conditional-information-bottleneck representation of V: maximal relevant content, minimal camera-specific content.

## Why this isn't plain information bottleneck

Standard IB would set `min I(V; W) − β I(W; Y)` with Y the decode target. But our Y is not a single image; it's a distribution over (C_q, t_q) → image. The relevant information structure is **equivariant**, not just scalar-target-predictive.

The right framing is probably closer to "disentangled sufficient statistic under a group action," where the group is SE(3) × R (camera × time). W lives in the space of scene representations modulo that group action; C and T carry the group element. This is a gauge theory framing: the camera choice is a gauge, world tokens are gauge-invariant, splats are gauge-fixed to the canonical frame (usually the camera-1 frame or some fixed origin).

Under this framing, the three architectures are three different estimators of gauge-invariant W:

- **A (dimensional routing)**: no formal guarantee. Gauge invariance is a hope that depends on W's capacity being saturated by non-gauge content.
- **B (adversarial)**: direct variational upper bound on I(W; C). Minimizing the adversary's accuracy lower-bounds the entropy of W given C, which upper-bounds I(W; C). This is exactly the variational mutual information literature (CLUB, MINE).
- **C (multi-camera consistency)**: enforces covariance/equivariance directly. Renders at multiple C_q from the same W must be mutually consistent; the only stable equilibrium is W being a gauge-invariant statistic.

Viewed this way, B is **measuring** the leakage (a probe-style bound) and C is **structurally preventing** it (a symmetry constraint). They're not the same kind of mechanism.

## What each training loss bounds, information-theoretically

- **Photometric at C_gt**: lower bound on I(W, C_gt, t_gt; V_at_t). Forces W+C+t to contain enough info to reconstruct.
- **Diffusion-as-loss at C_gt + δ**: lower bound on P(plausible frame | W, C_gt + δ). Forces W+C+t to contain enough info to produce an **on-manifold** render at an unseen camera.
- **Cross-view consistency at (C_a, t) vs (C_b, t)**: effectively an upper bound on the deviation from exact C-equivariance. This is the structural gauge-invariance enforcer.
- **Chunk swap (W_chunk1 + C_chunk2, t_chunk2 → V_chunk2)**: forces I(W_chunk1; V_chunk2) to be high, which requires W to be time-invariant within the clip span.
- **Crop augmentation**: forces W to be invariant to principal-point shift of the encode frame. This is a specific subgroup-invariance (translation-of-intrinsics subgroup).

So the losses, read information-theoretically, are: one sufficiency lower bound + one on-manifold lower bound + one equivariance upper bound + two invariance-under-subgroup terms.

## The real theoretical question

In the limit of infinite data + infinite capacity + this training recipe, what's the fixed point of the optimization? Is it:

- (a) The true gauge-invariant scene manifold with camera routed cleanly through C (the desired outcome),
- (b) A degenerate solution where W carries camera but C carries a compensating offset so the two cancel at render time (a silent failure),
- (c) A partial solution where invariance holds within training δ range but breaks outside,
- (d) Something else.

I don't know the answer. It depends on whether the equivariance constraint is sharp enough to rule out (b) and whether the cross-view consistency term covers the full SE(3) group or only its translation subgroup.

This is probably the single most useful theoretical question to nail down, because:

- If the answer is (a), the pioneer pick is provably correct in the limit and the implementation question reduces to "how much data/capacity is enough."
- If the answer is (b), we need an explicit cancellation-detection mechanism (probably B's adversarial probe used diagnostically, not as a primary loss).
- If the answer is (c), the δ curriculum becomes load-bearing and the target inference distribution defines a minimum δ we must cover at train time.

## What I'd push on next, theoretically

Two threads that feel like they have real content:

1. **Is the cross-view consistency loss a sharp SE(3)-equivariance constraint, or only a translation/rotation subgroup approximation?** If subgroup, enumerate what generality is lost. This directly determines whether (c) is our regime.

2. **Is there a cancellation-degenerate solution** (b) in the architecture family? This is answerable by thought experiment: construct a W that contains C-info and a C-token that contains -C-correction, and check whether it satisfies the full set of losses. If yes, (b) is live and we need a probe. If no, the architecture is provably immune.

Both are paper-shaped questions, not product-shaped.

Which thread feels more load-bearing to pull on?

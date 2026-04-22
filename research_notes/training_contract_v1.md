# Training Contract v1

Operational contract for the patched framing 3 baseline. This is not
`framing_4`; it is the runnable interpretation of the predictive quotient
claim.

---

## Purpose

Train one encoder that exports one world asset `W0` from an observation
set `O`, such that a time-conditioned scene generator plus fixed
rasterizer predicts omitted observations under the variable-budget
sampler `D_var`.

The supervised object is supported-query behavior:

```
w ~_Q w'
  iff p_theta(x | w,  c, tau) = p_theta(x | w', c, tau)
      for all (c, tau) in Q_train.
```

`L_pred` identifies `[W]_Q`, not a unique or pure latent. Rate/minimality
selects a useful representative. Full stochastic `W` is semantically
right but optional for v1; deterministic `W0` is a point estimate or
posterior summary.

---

## Variables / Sampler

Clip:

```
V = {v_i}
v_i = (I_i, c_i, tau_i, optional_metadata_i)
```

Training sample:

```
(O, H, Q) ~ D_var
O subset V                         # observations visible to encoder
H subset V, H not_subset O          # omitted targets
Q = {(c_h, tau_h) : h in H}         # held-out query cameras/times
```

Required `D_var` budgets:

- monocular sparse: small `|O|`, held-out elsewhere in the clip
- monocular rich: longer `O`, omitted future/past/interior targets
- crop-as-extrinsic: crops in `O` or `H` represented by changed
  intrinsics / principal point
- multi-view rich when available: some cameras in `O`, others in `H`
- synthetic multi-view as optional curriculum / EH-4

Hard sampler rule:

```
encoder_input_observations intersect loss_target_observations = empty
```

Any exact target observation visible to the encoder converts the regime into
autoencoding and invalidates v1. Same source time is allowed only when the
query really differs (for example different camera, crop-as-extrinsic, or
changed intrinsics) and the target pixels are not already present in the
encoder input.

---

## Model Signatures

Encoder ingestion:

```
obs_token_o = PatchEmbed(I_o)
            + RayEmbed(K_o, C_o)       # e.g. Pluecker rays
            + TimeEmbed(tau_o)

W0 = E_phi({obs_token_o : o in O})
```

Observation camera/time may enter once at ingestion. After ingestion,
world-side learned computation reads observation evidence only through
`W0`.

Prediction:

```
S_tau = G_theta(W0, tau_q)
Ihat  = R_fixed(S_tau, c_q)
```

Allowed:

- query time `tau_q` enters `G_theta`
- query camera `c_q` enters only `R_fixed`
- `R_fixed` is a fixed differentiable rasterizer
- `W0` may be slots/splats/tokens; set permutation is gauge

Forbidden in the baseline:

- learned pre-render branch conditioned on query camera
- render-time encoder features, per-frame caches, teacher state, or
  crop-specific side channels
- direct image decoder parallel to the splat/rasterizer boundary
- token-equality constraints between worlds from different observation
  budgets

Optional stochastic form:

```
q_phi(W | O), W0 = summary(q_phi)      # mean/MAP/code summary
```

v1 trains and exports `W0` unless EH-3 is explicitly admitted.

---

## Baseline Losses

Predictive loss:

```
L_pred =
  E_{(O,H,Q) ~ D_var} [
    sum_{h in H} ell(R_fixed(G_theta(W0, tau_h), c_h), I_h)
  ]

ell(Ihat, I) =
  lambda_L1    * ||Ihat - I||_1
  + lambda_LPIPS * LPIPS(Ihat, I)
```

Representative pressure:

```
L_v1 = L_pred + beta * Rate(W0)
```

Use the soft `Rate(W0)` term only when minimality is not already enforced
by architecture/export. Acceptable operationalizations:

- KL to a simple prior or posterior family
- MDL / code-length estimate
- hard token, slot, channel, or bandwidth bottleneck
- quantized-rate or entropy-model penalty
- explicit export-size cap

Do not add permanent gradient sources for camera adversaries, JEPA
agreement, token L2 agreement, cross-view consistency, denoising visible
targets, diffusion priors, or geometry teachers. Those are diagnostics
or escape hatches, not the baseline contract.

---

## Diagnostics

Run from day one; no gradients into the model from these metrics.

- Held-out render matrix: quality by observation budget, target time
  offset, camera delta, crop delta, and domain.
- Rate-distortion: export size / estimated rate vs held-out prediction.
- Predictive world agreement: compare `W_s = E(O_s)` and `W_r = E(O_r)`
  by rendering both on a shared supported query set. Agreement is
  predictive equality modulo gauge/permutation, not token equality.
- Camera leakage probe: fit a frozen-`W0` probe for pose/crop/focal.
  Interpret high accuracy as rate waste or weak `D_var` support before
  adding an adversarial loss.
- World-dependence matrix: token drop, token shuffle, zero time basis,
  wrong-world swap.
- Export-purity eval: render with every non-`(W0,c,tau)` tensor nulled.

---

## Escape Hatches

Admit only with a named trigger and measurement.

- EH-1 geometry forcing / VGGT alignment: decaying accelerator only.
  Must retire by end of pretraining.
- EH-2 diffusion-as-loss / SJC: post-training prior on held-out-camera
  renders only. Not a pretraining replacement for ground truth pixels.
- EH-3 residual stochastic world: add `p(W_hi | W0, O_partial)` only
  after demonstrating real multimodality that deterministic `W0` cannot
  summarize without averaging artifacts.
- EH-4 synthetic / multi-view budgets: add as more `D_var` support. No
  architecture or loss change required.

Stacking multiple escape hatches before the baseline tripwires are green
means the baseline or support assumptions are wrong.

---

## D_var Support Assumptions

All guarantees are conditional on `Q_train = supp_D(Q)`.

- Do not claim camera/time behavior outside supported query deltas.
- If monocular-only data cannot disambiguate a region, either state the
  ambiguity, add support, or admit EH-3.
- Cross-time consistency is only supervised where `D_var` samples
  omitted targets across time.
- Cross-view consistency is only supervised where `D_var` samples
  omitted camera changes or crop-as-extrinsic changes.
- Crop robustness requires treating crops as camera/intrinsics changes,
  not pixel augmentations with hidden metadata.
- Deployment query distribution must be a subset of, or explicitly
  labeled outside, `Q_train`.

Unsupported queries are evaluation probes, not contract guarantees.

---

## Deployment / Export Contract

Exported per-scene asset:

```
asset = W0
metadata = {rate/export_size, time_domain, renderer_version,
            generator_version, coordinate/gauge conventions}
```

Deploy render:

```
render(asset, c, tau):
    S_tau = G_theta(asset, tau)
    return R_fixed(S_tau, c)
```

The deployed renderer reads only `asset`, query camera `c`, query time
`tau`, and fixed model/rasterizer weights. It does not read encoder
intermediates, source frames, source poses, target images, teacher
networks, or training-time caches.

If stochastic export is enabled later, the asset must state whether it
exports a sampler, a finite hypothesis set, or a deterministic summary.
v1 default remains deterministic `W0`.

---

## Failure Tripwires

Stop and fix before interpreting quality metrics if any tripwire fires.

- Target/input overlap: any exact target observation is visible to the
  encoder in the same step; same source time is allowed only for a genuinely
  different query whose target pixels were not in the encoder input.
- Camera branch leak: query camera enters a learned module before
  `R_fixed`.
- Hidden state leak: render quality drops when non-`(W0,c,tau)` tensors
  are nulled at eval.
- Token equality objective: training rewards `W_s ~= W_r` directly
  instead of predictive agreement modulo gauge.
- Unsupported guarantee: paper/demo claims behavior outside `Q_train`.
- Rate bypass: export size or estimated rate grows without held-out
  prediction gain.
- Constant-world failure: wrong-world swaps render plausibly or token
  drops barely change output.
- Time-path failure: zeroing time basis barely changes dynamic renders.
- Stochasticity mismatch: deterministic `W0` averages incompatible
  futures, but EH-3 is not admitted or ambiguity is not reported.

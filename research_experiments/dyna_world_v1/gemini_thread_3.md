# Gemini Thread 3 — Synthesis: TokenGS, Rays-as-Pixels, Marigold-Zero-SNR, and the Implicit Token DiT

**Date:** 2026-04-19
**Context:** Third Gemini research thread exploring how to fuse video diffusion priors with decoupled splat tokenizers. Builds on `thread_1.txt` (TokenGS basics), `thread_2.txt` (DL3DV/Lyra 2/poly splats), and `SESSION_Q_AND_A_SYNTHESIS.md` (single-step diffusion + ChopGrad debate).

This note crystallizes the architectural insights from the thread. It is **synthesis**, not transcript — the raw Q&A trail is preserved implicitly in the structure below.

---

## 1. TokenGS Architecture — The Full Picture

TokenGS (NV-TLabs, CVPR 2026) replaces the dominant *encoder-only* splat prediction paradigm (pixelSplat, LRM, Splatter Image) with a **DETR-style encoder-decoder**.

### Why the old way is broken (Grid-Lock)
Encoder-only splat models map each input pixel to a splat by guessing a depth along its camera ray. This couples the number, density, and placement of 3D Gaussians to the 2D pixel grid of the inputs. Three consequences:
- **Redundancy:** a flat wall spends millions of splats duplicating nothing.
- **Spiky geometry:** depth-estimation error smears splats along the camera frustum.
- **No extrapolation:** splats can only live inside the input camera rays.

### TokenGS pipeline
1. **2D Encoder (ViT):** extracts dense multi-view features from an unordered set of posed images. Pure 2D understanding — no temporal backbone, no video model.
2. **Learnable 3DGS tokens:** a fixed set of N queries (e.g. 50k). **N is independent of input resolution and view count.** These are the knobs.
3. **Transformer decoder (cross-attention):** tokens query the 2D feature maps. Each token learns, organically, which region of 3D space it is responsible for.
4. **Output heads (MLPs):** regress **absolute global XYZ** (not depth-along-ray), plus scale, rotation (quaternion), opacity, and SH color.

### Training
- From scratch on multi-view datasets (RealEstate10K, DL3DV).
- ViT encoder may be init'd with DINOv2/CroCo, but the decoder + token heads are trained fresh.
- **Pure photometric rendering loss** — no 3D ground truth. A differentiable rasterizer renders novel views and compares to held-out images.

### What it buys
- Splat count decoupled from pixel count.
- Global coordinates — splats can land outside any input view's frustum (supports extrapolation).
- Emergent static/dynamic decomposition and scene flow from cross-view attention alone.
- Test-time optimization is trivially fast: just fine-tune the N tokens, not the whole network.

---

## 2. Rays-as-Pixels (Raxel) — Same Genre, Opposite Bet

Paper: https://wbjang.github.io/raysaspixels/

Raxels is a **joint video+camera diffusion model**, not a reconstructor. Both papers live in the same intellectual neighborhood ("unbind 3D from rigid conventions") but make opposite bets about the role of rays.

| Axis | TokenGS | Rays-as-Pixels |
|---|---|---|
| **Core output** | Persistent 3D asset (Gaussians, exportable) | 2D video frames + camera trajectories |
| **Role of the ray** | Bypassed — tokens predict global XYZ directly | Embraced — rays encoded as 2D "raxel" images and diffused |
| **Model type** | Deterministic feed-forward reconstructor | Probabilistic generative diffusion (DiT) |
| **Loss** | Photometric rendering (L1 + LPIPS) | Flow-matching / denoising on latents |
| **Attention trick** | Cross-attention: tokens → 2D features | Decoupled self-cross attention: video latents ↔ raxel latents |
| **Three capabilities** | Reconstruct from posed images | Pose prediction, pose-controlled gen, single-image-to-video |

**Shared DNA:** both use specialized attention mechanisms as bridges between incompatible representations (2D grids ↔ 3D tokens for TokenGS; appearance latents ↔ geometry latents for Raxels).

**Why this matters for dyna_world_v1:** Raxels proves that diffusion models *can* learn camera geometry if you format it as a grid. But it also confirms that diffusion models fundamentally *want* 2D grids — which is exactly why we need a token bridge if we're going to use them for splat prediction.

---

## 3. The Marigold Single-Step Trick — It's Zero-SNR, Not LCM

This is the most important technical clarification in the thread.

### Common misconception
Many assume Marigold went single-step via **Latent Consistency Model (LCM) distillation** (student-teacher compression from 50 steps → 1 step). That was the original Marigold-LCM, but it's been obsoleted.

### The actual story
Garcia et al. ("Fine-Tuning Image-Conditional Diffusion Models is Easier than You Think", WACV 2025) found that slow Marigold inference was a **math bug in the DDIM scheduler** for image-conditioned diffusion — a mismatch between the assumed noise level at T=1000 and the actual noise level.

The fix:
- **Enforce Zero Terminal SNR:** schedule so that t=1000 is *exactly* 100% pure noise (not 99.5%).
- **Trailing timesteps** for sampling.
- At inference: feed the input image, condition the network with "mean noise" (a zero tensor), lock timestep to t=1000, run **exactly one forward pass**.
- End-to-end fine-tune with a **pixel-space task loss** (e.g., affine-invariant depth loss through the *frozen* VAE decoder).

### Why this matters
- Single-step diffusion is **deterministic, not distilled** — no teacher/student quality degradation.
- The training signal is task loss in pixel space, not noise-prediction loss in latent space.
- **For dyna_world_v1:** we can apply this exact recipe to video diffusion models, bypassing all the multi-step BPTT horrors.

### The even cleaner "bypass VAE decoder" insight (from prior threads)
PrimeDepth-style: don't even route through the VAE decoder. Extract **preimage activations** from the U-Net after 1 step, feed them straight to the splat head. The differentiable rasterizer *replaces* the VAE decoder as the pathway from latents → pixels. Gradients flow through actual simulated light.

---

## 4. The Two-Stage Latent Diffusion Blueprint (The "Stable Diffusion of 4DGS")

This is the conservative, staged way to marry a video diffusion prior with TokenGS.

### Stage 1 — Train a time-conditioned TokenGS as a 4D VAE
- Extend TokenGS decoder to emit not just canonical splats but also **motion parameters** (polynomial ΔXYZ, ΔRotation over t — per the polynomial splat decision in `KEY_ARCHITECTURE_DECISIONS.md`).
- Train on multi-view video with photometric rendering loss.
- **Critical trap:** regularize the token latent space with **KL divergence** (Gaussian VAE) or **VQ** (discrete codebook). Without regularization, the token distribution is chaotic and undiffusable.
- Freeze the decoder when Stage 1 is done.

### Stage 2 — Fine-tune a video DiT to emit tokens
- Start with a pre-trained Video DiT (SVD, Mochi, Hunyuan, Sora-class).
- Rip off the RGB output head; replace with a projection to N token latents.
- **Condition on video + depth** (run a Marigold-style monocular depth pass to rigidly ground geometric scale — makes the token placement task much easier).
- **Apply the Zero-SNR fine-tuning trick:** lock t=1000, single forward pass, emit clean tokens.
- **Loss:** L2/Huber against ground-truth tokens from Stage 1. **No rasterizer in the backward pass** — all loss is in token space, so no OOM from the rasterizer.

### Why this is attractive
- Stage 2 trains unbelievably fast (no differentiable rendering in the gradient path).
- DiTs natively handle unordered token sequences — zero grid-lock.
- Depth conditioning removes the "where does this token go in world space" ambiguity.

### Why this is not yet transcendent
The VAE bottleneck is **information-destructive**. KL regularization smooths out the high-frequency detail that makes splats look sharp. You are forcing a massive intelligent video model to squeeze its hallucinated geometry through a narrow pipe.

---

## 5. The Transcendent Single-Stage Merge — Implicit Token DiT (IT-DiT)

**The novel insight of this thread.** Kill the VAE, kill the two-stage pipeline, kill the regularization penalty. Fuse the splat decoder directly into the diffusion transformer.

### Architecture

```
[Input video] ──► [Frozen Video DiT, locked at t=1000, 1-step pass]
                         │
                         ▼
               [Hidden DiT feature volume: hallucinated 3D/4D scene]
                         │
                         ▼
   [N learnable "Implicit Splat Tokens" (queries)] ◄─── cross-attend
                         │
                         ▼
              [TokenGS-style MLP heads]
                         │
                         ▼
              [Explicit 4D Gaussians (polynomial splats)]
                         │
                         ▼
              [Differentiable rasterizer]
                         │
                         ▼
        [Rendered frames at novel (t, pose)] ─── photometric loss
```

### Key mechanics
1. **No VAE bottleneck.** Tokens cross-attend directly to the DiT's hidden features. No KL penalty, no VQ codebook, no information loss.
2. **Tokens self-organize.** With no artificial distribution constraint, tokens learn whatever representation minimizes rendering loss. One might converge on "background geometry"; another on "motion edges." Emergent decomposition — no architect required.
3. **One-step DiT = tractable end-to-end training.** Backpropagating a rendering loss through 50 diffusion steps is OOM suicide. Through 1 step, it's a normal forward/backward pass.
4. **LoRA the DiT backbone.** Freeze base DiT weights, train only LoRA adapters on the attention layers + full training on the token queries + MLP heads. Preserves the generative prior, keeps VRAM sane.

### Training signal
- **Pure photometric loss** on rendered frames (L1 + SSIM + optional LPIPS).
- No noise-prediction loss. No latent-space loss. The differentiable rasterizer is the translator from the DiT's hallucinated geometry to the light we supervise on.
- Gradients flow: pixel loss → rasterizer → splat params → MLP heads → implicit tokens → DiT attention layers (via LoRA).

### What it buys
- **Generative hallucination of occluded geometry** (comes for free from the DiT prior — if the video shows the front of a car, the DiT's hidden features already encode the back).
- **No grid-lock** (tokens live in global continuous 3D, not pixel-aligned).
- **Single-step inference** (Zero-SNR trick — no iterative denoising).
- **No VAE compression artifacts** (no information bottleneck between diffusion and splat decoder).
- **True end-to-end joint training** (no stage-coupling drift).

---

## 6. Implications for dyna_world_v1

This thread reshapes the architectural priors in `KEY_ARCHITECTURE_DECISIONS.md` as follows:

### Reinforced
- **Videometric loss is the right supervisor.** Thread 3 independently converges on "photometric rendering loss in pixel space, gradients through the rasterizer" as the training signal. No change to Decision #1.
- **Polynomial 4D splats remain the right primitive.** Drop SH3 → SH0, reinvest 45 params in polynomial coefficients. Memory footprint identical to static 3DGS. No change to Decision #2.
- **Foundation video models as backbones.** Thread 3 agrees — but with a strong recommendation on *which* activation-probing strategy to use.

### Refined / updated
- **Option A (Activation Probing) in `KEY_ARCHITECTURE_DECISIONS.md` is essentially IT-DiT — but sharpen it:**
  - Use the Zero-SNR single-step trick, not arbitrary noise levels or multi-step denoising.
  - Freeze the DiT base; train LoRA adapters + N implicit token queries + MLP heads.
  - Cross-attend tokens directly to DiT hidden features (skip the VAE decoder entirely).
- **Option B (Latent-to-Splat decoder) is the two-stage LDM blueprint.** Still a legitimate fallback if single-stage IT-DiT proves unstable — but it has the VAE bottleneck problem and loses information vs IT-DiT.
- **Option C (SDS) is orthogonal** — useful only for text/image → 4D generation from scratch, not for video → splats reconstruction, which is our MVP.

### New open questions to chase
- **Does the frozen Video DiT actually encode usable 3D geometry in its hidden states at t=1000?** PrimeDepth/Marigold prove this for static images; we need to verify for video DiTs.
- **How many implicit tokens do we need?** TokenGS uses ~50k. For 4D polynomial splats covering a 30s clip, is that enough, or do we need 250k+?
- **Which DiT layer has the richest 3D-aware features?** Analogous to "which ResNet block gives the best depth probe?" — needs a probing study.
- **LoRA rank budget** — how much of the DiT do we actually need to adapt? Too much erases the prior; too little underfits.

### Conflict with prior session decisions
- Thread 3's "Parallel/Global" IT-DiT conflicts with the **ChopGrad Autoregressive path (Path B)** from `SESSION_Q_AND_A_SYNTHESIS.md` §5. Single-step diffusion eliminates the recurrent-loop-through-VAE problem that motivated ChopGrad. **If we go IT-DiT, ChopGrad becomes irrelevant for the render pass.** Keep ChopGrad in reserve only if we later go back to multi-step or autoregressive decoding.

---

## 7. Recommended Next Experiment

A minimal IT-DiT probe to test the core hypothesis:

1. **Backbone:** freeze a small Video DiT (SVD-small or similar).
2. **Data:** DL3DV-10K subset (static, for a first probe) — 50 scenes.
3. **Conditioning:** posed multi-view video + Marigold depth pass.
4. **Tokens:** 10k learnable implicit splat queries.
5. **Training:** LoRA adapters on DiT attention + full training on token + MLP heads. Zero-SNR single-step forward pass. Photometric loss through differentiable rasterizer.
6. **Success metric:** PSNR/SSIM on novel-view reconstruction beats a from-scratch TokenGS baseline trained on the same 50 scenes.

If this works statically, move to dynamic (surfing fixture, polynomial splats) as Phase 2.

---

*This document is the synthesis of Gemini research thread 3 (2026-04-19). It supersedes the implicit architectural choices in `KEY_ARCHITECTURE_DECISIONS.md` Option A by sharpening it into IT-DiT. It does not deprecate thread_1/thread_2 — those provide the historical context and alternate paths (two-stage LDM, autoregressive+ChopGrad) that remain viable fallbacks.*

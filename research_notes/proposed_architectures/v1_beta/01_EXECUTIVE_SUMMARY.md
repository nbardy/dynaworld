# Dyna World v1 Beta: Executive Summary

## The Vision
**Dyna World v1** is a zero-shot Video-to-4D reconstruction and generation architecture. By treating Video Diffusion Models as foundation "World Models" that inherently understand geometry, we extract dense 3D/4D structures directly from their internal activations.

## The Three Core Breakthroughs

This `v1_beta` architecture is the synthesis of three major paradigm shifts discovered in recent literature and engineering deep-dives:

1. **Single-Step Preimage (The Encoder)**
   *   *The Myth:* Diffusion models need 50+ iterative steps to recover geometry.
   *   *The Reality:* A single forward pass through a frozen (or lightly fine-tuned) Video Diffusion Backbone (like SVD or HunyuanVideo) yields a massive spatiotemporal feature map (the "Preimage"). This single step contains all the geometric priors needed for 4D reconstruction, unlocking a 100x speedup.

2. **The Causal Splat Decoder (The "Splat Head")**
   *   *The Myth:* 4D Gaussians must be global polynomials across the entire video.
   *   *The Reality:* Global polynomials entangle the entire video's features. Backpropagating a global coefficient wakes up the entire Video Backbone's computation graph, causing instant OOM.
   *   *The Fix:* We use a **Causal Splat Decoder**. It updates the splat state frame-by-frame ($S_t = \text{Update}(S_{t-1}, F_t)$). This handles infinitely complex motion and topology changes gracefully, as time is a recurrent state rather than a global equation.

3. **ChopGrad + Pixel Space Videometric Loss (The Trainer)**
   *   *The Myth:* We must calculate the loss in the VAE's Latent Space to train the backbone.
   *   *The Reality:* Latent loss produces blurry geometry. We bypass the VAE decoder entirely, using a physics-based **Splat Rasterizer** to render the Gaussians to a 2D image.
   *   *The Exploit:* By applying a Videometric Loss (L1 + SSIM) in true pixel space, we get perfect physics-based gradients. Because our Causal Splat Decoder is recurrent, we apply **ChopGrad (Truncated Backpropagation Through Time)** to the splat state ($S_{t-1}$). This breaks the temporal memory chain, guaranteeing $O(1)$ constant memory scaling for infinite-length video training!

---
*Explore the subsequent files in this directory for deep-dives into the Data Flow Diagram, the Causal Decoder, ChopGrad integration, and Loss mechanics.*
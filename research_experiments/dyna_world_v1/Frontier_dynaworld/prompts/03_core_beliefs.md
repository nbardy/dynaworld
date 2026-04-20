# The Dyna World v1 Bible

**Our Core Beliefs:**
1. **World Models = Video Models:** Video diffusion models inherently learn the geometry of the world to generate realistic light. Static and dynamic scenes are functionally equivalent under this paradigm.
2. **Extreme Data Efficiency:** Mapping video to 3D Gaussian Splats requires only a minimal amount of data (linear probe). Like Flux Edit LoRAs, we can achieve this with as few as 10-50 high-quality samples.
3. **Foundations are Sacred:** We do not train new foundation models. The best way to extract 3D/4D representations is to build a lightweight adapter (a Splat Head) on top of a frozen, self-supervised video diffusion model (e.g., SVD, HunyuanVideo).
4. **Single-Step Preimage:** We do not need iterative diffusion to recover geometry. A single forward pass through the diffusion backbone yields a rich spatiotemporal "Preimage" (attention and feature maps) that completely defines the scene.
5. **Pixel Space Videometric Loss:** Latent diffusion loss produces blurry geometry. We bypass the VAE decoder completely, evaluating our generated splats through a differentiable **Splat Rasterizer** and applying a Photometric (L1 + SSIM) loss directly against the ground-truth video pixels.
6. **The SH0 Parameter Trade-Off:** To prevent VRAM explosions, we restrict splat colors to Spherical Harmonics Degree 0 (base RGB). We reinvest the 45 saved parameters per Gaussian into temporal features (polynomials or velocity/acceleration vectors).

**Our Ultimate Goal:**
To design the definitive neural architecture for Dynamic Scene Synthesis (4D Generation) that maps video directly to Gaussian Splats in a single deterministic pass. It must be highly memory-efficient, geometrically accurate, and scale to arbitrarily long videos.

**Current Architectural Crossroads:**
- **Path A (Parallel/Global):** A massive pool of splats generated at once. 100% parallel, no recurrent states, but scales poorly to long videos due to parameter explosion and $O(N)$ cross-attention.
- **Path B (Causal/Autoregressive):** A compact, fixed pool of splats whose state is updated frame-by-frame ($S_t = S_{t-1} + \Delta S_t$). Highly memory-efficient. We use **ChopGrad** (Truncated Backpropagation Through Time) on the recurrent splat state to prevent the video backbone from OOMing during training, achieving $O(1)$ constant memory scaling for infinite-length videos.
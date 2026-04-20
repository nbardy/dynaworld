# Path A: Parallel / Global Splat Decode

## How TokenGS Handles Splat Counts
In the original TokenGS paper (which reconstructs static 3D scenes from multi-view images), the number of 3D Gaussians is not determined by the number of pixels. Instead, they initialize a **fixed number of learnable "3DGS Tokens"** (e.g., $N = 50,000$ or $100,000$). 
- These tokens act as "queries" in a Transformer Decoder.
- They cross-attend to the dense 2D feature maps from the image encoder.
- Each token directly outputs the $(X, Y, Z, \text{Scale}, \text{Rotation}, \text{Color}, \text{Opacity})$ for a single Gaussian.

## Extending to 4D Video (The Massive Splat Head)
To make this work for a dynamic video in a **Parallel / Global** manner, we do not predict polynomials or causal state updates. Instead, we predict a **Spatiotemporal Splat Cloud**.

We initialize a massive pool of tokens (e.g., $N = 1,000,000$). These tokens cross-attend to the spatiotemporal features extracted from the Video Diffusion Backbone (all frames simultaneously).

For each token, the network outputs the standard 3D parameters, plus two new **Temporal Parameters**:
1. **Temporal Center ($t_c$):** The exact timestamp when this splat is most opaque.
2. **Temporal Width ($\sigma_t$):** How long this splat "lives" before fading out.

### The Render Equation
At any given frame time $t$, the effective opacity of a splat is computed dynamically before rasterization:
$$ \alpha_{effective}(t) = \alpha_{base} \cdot \exp\left( - \frac{(t - t_c)^2}{2 \sigma_t^2} \right) $$

If $\alpha_{effective}(t)$ drops below a certain threshold (e.g., $0.05$), the splat is simply discarded from the sorting and rasterization pipeline for that specific frame.

### The Training Flow
1. Pass the entire video sequence through the Video DiT backbone.
2. The Massive Splat Head attends to all features and outputs the 1,000,000 splats.
3. For a randomly selected batch of frames (e.g., $t=5, 12, 28$):
   - Evaluate $\alpha_{effective}(t)$ for all 1M splats.
   - Cull the invisible ones (leaving, say, 50,000 active splats per frame).
   - Rasterize the active splats to an RGB image.
   - Calculate Videometric Loss against the GT frame.
4. Backpropagate. The gradients flow only into the splats that were active at time $t$, updating their $t_c$, $\sigma_t$, and spatial parameters.
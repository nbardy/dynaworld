# Videometric Loss (Pixel Space Training)

## The Rejection of Latent Loss
A major conclusion from recent single-step diffusion papers (PrimeDepth, Marigold-SSD, etc.) is the rejection of latent-space loss. 

Early conditional diffusion models encoded ground truth depth into a latent representation and trained the U-Net to match that latent using Mean Squared Error (MSE). 
*   **The Problem:** The VAE latent space is optimized for compressing RGB textures, not for preserving sharp geometric boundaries. Latent MSE acts as a surrogate loss that leads to blurry, suboptimal geometry.

## The Splat Rasterizer as the True Decoder
Instead of using the heavy, memory-intensive Video VAE Decoder to map latents back to pixels, **we bypass neural decoders entirely.**

The true decoder of the Dyna World v1 architecture is the physics of light: the **Differentiable Splat Rasterizer**.
- It takes the output of our Causal Splat Decoder ($S_t$).
- It projects the 3D Gaussians onto a 2D camera plane.
- It calculates volume rendering (alpha compositing) to produce a 2D RGB image.

Because the rasterizer is fully differentiable, gradients flow perfectly from the output pixels straight back into the splat parameters.

## The Loss Functions
We train the system entirely in **Pixel Space**. No 3D ground truth (point clouds or meshes) is required.

1. **Photometric (RGB) Loss**
   - We compare the rendered RGB image at time $t$ to the ground-truth video frame at time $t$.
   - **L1 Loss:** $\mathcal{L}_{L1} = | I_{rendered} - I_{GT} |$. Ensures absolute color accuracy.
   - **SSIM Loss:** Structural Similarity Index. Preserves high-frequency details, edges, and textures.
   - $\mathcal{L}_{photo} = \lambda_1 \mathcal{L}_{L1} + \lambda_2 (1 - \text{SSIM})$.

2. **Affine-Invariant Depth Loss (Optional / Auxiliary)**
   - If we choose to supervise the model using pseudo-depth (e.g., from Depth Anything V2) to speed up geometric convergence:
   - We use an **Affine-Invariant Loss** to mathematically align the predicted depth map's scale ($s$) and shift ($t$) to the pseudo-GT depth map before calculating the error. This ignores the global scale ambiguity of monocular depth models.

3. **Temporal Regularization**
   - Because we use a Causal Splat Decoder, we can penalize extreme accelerations or velocities between $S_t$ and $S_{t-1}$ to prevent the splats from "teleporting" or jittering erratically.
   - $\mathcal{L}_{reg} = || (S_t - S_{t-1}) - (S_{t-1} - S_{t-2}) ||^2$ (Minimizing acceleration).
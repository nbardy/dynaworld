# Study: Single-Step Diffusion for Video $\rightarrow$ Splat(t)

## 1. Core Inspiration from Depth Models
Recent breakthroughs in monocular depth estimation (PrimeDepth, DepthMaster, Marigold-SSD) have proven that you do not need slow, iterative diffusion loops to extract geometry. Instead, you can treat a frozen Diffusion Model as a massive feature extractor (a "Preimage"):
- **PrimeDepth:** Runs a single step of Stable Diffusion and extracts multi-scale feature maps and attention maps from every neural block (the "Preimage"). It then uses a lightweight refiner to map these generative features into a discriminative depth map.
- **DepthMaster & Marigold-SSD:** Utilize single-step deterministic paradigms, proving that high-quality scene structure can be extracted in a single forward pass without iterative denoising.

## 2. Our Translation: Video $\rightarrow$ Splat(t)
Instead of predicting a dense depth map (like Marigold or PrimeDepth), our goal is to predict **Time-Conditioned 3D Gaussians** — $Splat(t)$. 

Our architecture completely bypasses a complex neural decoder. **The Splat Decoder is just the Differentiable Splat Renderer.**

### The Pipeline Architecture
1. **The Frozen Backbone (Video Diffusion Prior):** 
   - We pass the input video through a Video Diffusion Model (e.g., SVD, HunyuanVideo) for a **single denoising step**.
   - We extract the rich spatiotemporal activations (the "Video Preimage"): cross-attention maps, self-attention maps, and intermediate DiT/U-Net feature blocks.
2. **The "Decoder" (Linear Probe / Shallow Regressor):**
   - We don't use a heavy generative decoder. Instead, we use a lightweight mapping network (e.g., a shallow MLP or TokenGS-style cross-attention layer).
   - This network projects the rich, single-step diffusion activations directly into the **polynomial coefficients** of the 4D Gaussians ($c_0, c_1, c_2 \dots$ for Position, Rotation, and Opacity) and $SH_0$ for Color.
3. **The Splat Renderer (The True Decoder):**
   - The predicted polynomial coefficients are evaluated at a specific time $t$ to materialize the instantaneous 3D geometry.
   - The geometry is fed into the standard differentiable Gaussian splatting rasterizer to render a 2D image.
4. **Videometric Loss:**
   - The entire shallow regressor is trained end-to-end using a **multi-frame videometric (photometric) loss**. The rendered frame at time $t$ is compared directly to the ground-truth video frame at time $t$. No 3D or Depth ground truth is required.

## 3. Why This Works
- **Efficiency:** We get the extreme temporal consistency and semantic understanding of a billion-parameter Video Diffusion Model, but inference operates in a single forward pass (100x faster than iterative diffusion).
- **Geometric Bias:** Diffusion models implicitly learn geometry to generate realistic light. By tapping into their internal activations, we are simply probing for that existing geometric knowledge.
- **No VRAM Explosion:** Because the splat parameters are modeled as polynomial coefficients and color is restricted to $SH_0$, the memory footprint remains incredibly tight, allowing the gradients to flow through the renderer and back into the probe efficiently.
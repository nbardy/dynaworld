# Key Learnings: Single-Step Diffusion for Geometry Extraction

Based on our review of recent single-step diffusion papers (PrimeDepth, Marigold-SSD, DepthMaster, and Fine-Tuning Image-Conditional Diffusion Models is Easier than You Think), we have extracted several critical insights that will drive the architecture of **Dyna World v1**.

## 1. The "Iterative Denoising" Myth
The prevailing assumption that diffusion models require tens or hundreds of denoising steps to extract high-quality geometry (like depth or normals) is false. The rich geometric and semantic priors are already baked into the weights of the pretrained UNet/DiT. You can extract these priors in a **single forward pass**.

## 2. The Scheduler Bug
The reason early conditional diffusion models (like Marigold) performed terribly in the 1-step or few-step regime was due to a fundamental flaw in the DDIM inference scheduler. The model was receiving a timestep encoding indicating a clean image (e.g., $t=1$), but the actual input was pure noise. 
- **The Fix:** By simply aligning the timestep with the correct noise level (or locking $t=T$ and passing zero noise with the RGB latent), the model can accurately predict geometry in a single deterministic step, resulting in a **200x speedup**.

## 3. The "Preimage" / Activation Probing
You do not need a massive generative decoder to get 3D/4D structure. A frozen diffusion model acts as an incredibly powerful feature extractor. By running a single step and pulling out the intermediate feature maps, self-attention maps, and cross-attention maps (the "Preimage"), you can train a very lightweight, shallow network (a linear probe or TokenGS-style head) to regress complex geometry (like our polynomial splat coefficients).

## 4. Ditching Latent Loss for Pixel Space Loss
A major takeaway from the "End-to-End Fine-Tuning" paper and PrimeDepth is that you must **stop computing your loss in the latent space of the diffusion model**. The latent space is optimized for compressing RGB textures, not for evaluating geometry or structural boundaries.

### The Role of the VAE Decoder
Early methods encoded ground truth depth into a latent representation and trained the U-Net to match that latent using MSE. This led to blurry and suboptimal geometry. The new generation of single-step methods fixed this by moving to pixel space:
- **Approach A (End-to-End Fine-Tuning):** Pass the U-Net's predicted depth latent through the *frozen VAE decoder* to get a full-resolution depth map. Backpropagate the pixel loss backward through the frozen VAE decoder and into the U-Net.
- **Approach B (PrimeDepth):** Bypass the VAE decoder entirely. Use a custom, lightweight "Refiner Network" that takes the U-Net "Preimage" activations and outputs depth directly in pixel space, applying the loss there.

### Affine-Invariant Depth Loss
When operating in pixel space on diverse datasets (LiDAR, synthetic, stereo), global scale and shift vary wildly. An **Affine-Invariant Loss** mathematically aligns the predicted depth map to the ground truth depth map by finding the optimal scale ($s$) and shift ($t$) for that specific image before computing the pixel-wise error (MSE/MAE).

## 5. Application to Dyna World v1
For our **Video $\rightarrow$ Splat(t)** architecture, we are following the **PrimeDepth** philosophy to bypass the heavy VAE decoder:
1. Lock the video diffusion model to a single timestep to extract the spatiotemporal "Preimage".
2. Use a shallow network to predict 4D Gaussian polynomial coefficients directly from the preimage activations.
3. **The Splat Rasterizer is our Decoder:** We bypass any neural decoder entirely. We evaluate the coefficients at time $t$ and rasterize the Gaussians to a 2D image.
4. **Task-Specific Loss in Pixel Space:** We calculate the Videometric/Photometric Loss (L1 + SSIM) between the rendered RGB pixels and the actual ground truth video pixels, backpropagating through the physics of the rasterizer straight into our shallow network.
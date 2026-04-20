# Key Architecture Decisions: Dyna World v1

## 1. Videometric (Photometric) Loss
Instead of relying on explicit 3D/4D ground truth (which is inaccurate at scale, expensive, and often relies on synthetic data), the model is supervised entirely by 2D light—raw video frames. 
- **Mechanism:** We render the predicted 4D Gaussians at a specific timestamp $t$ from a specific camera pose, and compare the rendered 2D image against the ground-truth video frame.
- **Loss Function:** A combination of L1 loss, SSIM, and potentially LPIPS. We can also add a temporal consistency regularization to penalize extreme polynomial curves and prevent jitter.
- **Batching Strategy:** The training loop samples random $(t, \text{pose}, \text{image})$ tuples across different sequences. The gradients flow back through the differentiable rasterizer directly into the global polynomial coefficients of the scene.

## 2. Time-Conditioned Splats (Polynomial 4D Gaussians)
We avoid computationally heavy deformation MLPs (which slow down rendering) and VRAM-heavy 4D volumes by modeling motion through **polynomial coefficients**.
- **The SH0 Trade-off:** We restrict color to Spherical Harmonics Degree 0 (base RGB, view-independent). This frees up 45 parameters per primitive (compared to full SH3).
- **Polynomial Re-investment:** We use those reclaimed 45 parameters to store polynomial coefficients for Position, Rotation (Quaternions), and Opacity (e.g., $x(t) = c_0 + c_1t + c_2t^2 \dots$).
- **Efficiency:** The rasterizer simply evaluates these polynomials at the target time $t$ on the fly before sorting and splatting. The VRAM footprint of the primitive array remains identical to standard static 3DGS, shifting the bottleneck slightly toward compute.

## 3. Foundational Backbones & Bootstrapping
We will utilize foundation video models to provide the rich spatiotemporal understanding needed to reconstruct 4D scenes, acting as the powerful "encoder" for our 3DGS token decoder.

---

### Clarification on TokenGS
The **TokenGS** paper is a **single-step, feed-forward encoder-decoder**. It is **not** a diffusion model. 
1. It uses a standard Vision Transformer (ViT) to extract 2D features from images.
2. It uses a Transformer Decoder where a fixed number of learnable "3DGS Tokens" cross-attend to those ViT features to predict the splat parameters.

### How to Bootstrap on Video Diffusion Models?
Video Diffusion Models (VDMs) like Sora, HunyuanVideo, or SVD build incredibly rich internal representations of depth, geometry, and motion within their DiT or U-Net blocks. To attach a TokenGS-style splat decoder to a VDM, we have three main paths:

#### Option A: Activation Probing (The "One-Step" Extractor)
Just like how depth and normals can be extracted from Stable Diffusion with a linear probe or lightweight adapter, we can extract 4D geometry.
1. Take the input video and add a specific amount of noise.
2. Run **one single forward pass** (one denoising step) through the VDM's frozen DiT/U-Net.
3. Extract the rich spatiotemporal attention maps / intermediate activations.
4. Train our TokenGS-style decoder to cross-attend to these activations and regress the polynomial splat coefficients.

#### Option B: Latent-to-Splat Decoder
1. Pass the video through the VDM's VAE to get compressed spatiotemporal latents.
2. Pass the latents through the first few layers of the VDM to contextualize the sequence.
3. Branch off into a custom Transformer Decoder (our Splat Head) that maps these contextualized latents into the global 4D Gaussian polynomial coefficients.
4. Train *only* this Splat Head using our multi-frame videometric loss.

#### Option C: Score Distillation (SDS/VSD) (For Generation from Scratch)
If we want to generate 4D scenes from a text prompt or a single image (rather than reconstructing a known video):
1. Initialize a random set of 4D splat polynomials.
2. Render frames at various times $t$.
3. Add noise and pass them to the frozen VDM.
4. The VDM predicts the noise, providing a gradient direction to update our splat polynomial coefficients to make the rendered frames look more like realistic, temporally consistent video.
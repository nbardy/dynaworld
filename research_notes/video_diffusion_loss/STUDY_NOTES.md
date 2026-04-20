# ChopGrad: Pixel-Wise Losses for Latent Video Diffusion

**Paper:** *ChopGrad: Pixel-Wise Losses for Latent Video Diffusion via Truncated Backpropagation*
**Link:** [https://arxiv.org/abs/2603.17812](https://arxiv.org/abs/2603.17812)

## 1. The Core Problem: The Memory Explosion of Pixel-Wise Video Loss
Modern latent video diffusion models use **Temporal VAEs** to compress videos into latents. To maintain smooth temporal consistency without flickering, these VAE decoders use a technique called **causal caching**.
- **Causal Caching:** At each layer of the decoder, the features from the previous frame group are concatenated (appended) to the current frame group.
- **The Backprop Nightmare:** Because of this causal link, the decoder becomes a recurrent neural network. If you try to calculate a pixel-wise loss (like MSE, L1, or LPIPS) on frame $N$, the gradients must propagate backwards through frame $N-1$, then $N-2$, all the way to the start of the video. 
- **Result:** Memory usage scales linearly ($O(N)$) with the length of the video, making high-resolution, pixel-space fine-tuning computationally intractable.

## 2. The Solution: ChopGrad (Truncated Backpropagation)
The authors made a critical mathematical observation called **Latent Temporal Locality**: the influence of a past latent embedding on a current pixel output decays exponentially over time. 
- **Truncated Backprop:** Instead of backpropagating all the way to the beginning of the video, ChopGrad simply "chops" the gradient flow after a small, fixed number of previous frame groups ($D_{trunc}$).
- **The Impact:** Memory consumption drops from $O(N)$ to $O(1)$ (constant memory), regardless of how long the video is. The gradient error introduced by truncation is bounded and primarily affects magnitude, not direction, so optimizers (like Adam) easily compensate for it.

## 3. How This Impacts Dyna World v1
Our `Dyna World v1` architecture relies on a **Videometric Loss** (a pixel-wise photometric loss between rendered splats and GT video). 

### Scenario A: If we use the VAE Decoder
If we were building a model that predicts depth/splats in the *latent space* and then decodes those latents to pixels using the Video VAE to compute the loss, ChopGrad would be **absolutely mandatory**. Without it, we would immediately OOM trying to backpropagate our pixel loss through the causally-cached Video VAE decoder.

### Scenario B: The PrimeDepth Philosophy (Bypassing the VAE)
As established in our `single_step` learnings, our current plan is to **bypass the VAE decoder entirely**. 
1. We extract the "Preimage" activations from the frozen Video Diffusion U-Net.
2. We map those to 4D Gaussian Polynomial Coefficients.
3. We render the pixels using the physics-based **Splat Rasterizer**.
4. We compute the pixel loss and backpropagate through the rasterizer.

**The Catch:** Even if we bypass the VAE decoder, if we choose to *unfreeze* and fine-tune the Video Diffusion U-Net (or DiT), we may encounter similar temporal memory explosions. Many modern video backbones (like HunyuanVideo or CogVideoX) use sliding-window or causal attention mechanisms. **ChopGrad's philosophy of Truncated Temporal Backpropagation** gives us the exact mathematical justification to chunk our temporal gradients during the U-Net backward pass, guaranteeing that our pixel-space Videometric loss remains tractable across long video sequences!
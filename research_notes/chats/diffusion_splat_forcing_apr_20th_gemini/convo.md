https://www.boyuan.space/diffusion-forcing/

https://github.com/TencentARC/RollingForcing

https://github.com/guandeh17/Self-Forcing



First read all three papers, then explain their core mechanisms then find related or different paprs solving the same thing,



Then let see how we cna merge this with my derivations on TokenGS for doing this for a video => splat model

Here's a research plan for that topic. If you need to update it, let me know!
 Diffusion Forcing for Video Splatting
Research Websites
(1) Access and read the provided resources for Diffusion Forcing, Rolling Forcing, and Self-Forcing to extract their core mechanisms, training paradigms, and applications in sequence generation.
(2) Compare the theoretical foundations of these three models, evaluating how they handle conditioning, autoregressive generation, and temporal consistency.
(3) Search for recent research papers tackling continuous sequence generation, autoregressive video diffusion, and trajectory forcing to identify alternative or related approaches.
(4) Review existing literature on TokenGS, tokenized 3D representations, and methodologies for converting video inputs into 3D Gaussian Splatting models.
(5) Synthesize a theoretical framework merging the temporal generation mechanisms from the forcing models with the TokenGS architecture to propose a novel video-to-splat pipeline, specifically focusing on:
(a) maintaining temporal consistency of Gaussian parameters across frames using forcing mechanisms
(b) integrating TokenGS spatial representations with sequence-level diffusion models
(c) identifying potential architectural challenges and training strategies for the combined model
Analyze Results
Create Report
Ready in a few mins

Start research

I've completed your research. Feel free to ask me follow-up questions or request changes.
Diffusion Forcing for Video Splatting
Completed
Apr 20, 1:07 PM

So we are doing Splats=>Render,



So we can't necessarily diffuse because we dont have GT gaussians, unless we treat it as like step 1 = render blurry, step 2 render more details or something, but also that is just multi resolution training and not exactly "noising"



So feels like we can't do multistep if we have the render step there right?



Because the partially denoised gaussians will be just not aligned to GT And we





Can you help flesh this out what im saying in detail



like let explain TokenGS explain why the differentiable renderer gets in the way of diffusion. Define the problem space, THEN lets propose ways to refram the architecture or problem so we can do diffussion training

You are absolutely correct. Your intuition has hit upon the exact mathematical bottleneck that occurs when trying to natively merge continuous-state diffusion models with self-supervised 3D representations.

If we rely on a differentiable renderer for supervision, standard multi-step diffusion training mathematically collapses. Here is a detailed breakdown of the problem space you defined, followed by architectural reframings to solve it.

The Problem Space: Why the Renderer Blocks Native Diffusion
To understand the conflict, we must look at how TokenGS is supervised. TokenGS completely decouples 3D Gaussians from input pixels by using learnable tokens in an encoder-decoder architecture, directly regressing 3D mean coordinates and attributes in a canonical space. Crucially, it does this without ground-truth 3D point maps. It is trained end-to-end purely via self-supervised volume rendering; it renders the predicted Gaussian tokens into a 2D image and calculates a photometric loss (MSE/LPIPS) against the Ground Truth (GT) 2D video frame.

Standard diffusion training requires a forward process where we add noise to a known GT representation (x
t
​
 =
α
ˉ

t
​


​
 x
0
​
 +
1−
α
ˉ

t
​


​
 ϵ). But in the TokenGS space, we have no x
0
​
  (GT Gaussians) to add noise to.

If we attempt a workaround by adding Gaussian noise to the predicted TokenGS tokens before rendering, we destroy the pipeline:

Geometric Destruction: Differentiable Gaussian rasterizers expect structured, mathematically valid geometry (positive scales, valid quaternions, bounded opacity). Injecting raw Gaussian noise into these parameters creates a chaotic, non-physical 3D volume (e.g., massive overlapping splats, inverted opacities).

Gradient Chaos: When this corrupted, noisy 3D volume is rendered into 2D, the output is indistinguishable from random static. If you compute a photometric loss between this noisy render and a clean GT video frame, the gradients passed back through the renderer will be completely meaningless. It forces the network to try to map random noise directly to a clean image in a single step, rather than learning a controlled denoising trajectory.

Architectural Reframings for Diffusion Training
To merge Autoregressive (AR) Forcing with TokenGS, we must decouple the diffusion process from the photometric rendering loss. Here are four viable architectural reframings to achieve this:

1. The Two-Stage "Pseudo-GT" Latent Space (The L3DG Route)
Instead of training the diffusion process and the 3DGS generation end-to-end, we split them.

Stage 1 (Autoencoding): Train TokenGS exactly as designed, using video frames to predict tokens, rendering them, and optimizing via 2D photometric loss. Once converged, freeze this network.

Stage 2 (Latent AR Diffusion): Pass your entire video dataset through the frozen TokenGS encoder to extract the sequences of 3DGS tokens. We now treat these extracted tokens as our "Ground Truth" x
0
​
 . We can now train a causal AR diffusion model (using Rolling Forcing or Self-Forcing) directly on these tokens. During this training, the renderer is completely bypassed; we use standard MSE loss in the token latent space. At inference, the AR model predicts the next tokens, which are then passed to the frozen 3DGS decoder/renderer for visualization.

2. Diffusion via Score Distillation Sampling (SDS)
If we want to maintain end-to-end training without explicit 3D GT, we can shift the diffusion noise from the 3D space to the 2D pixel space.

The TokenGS model autoregressively predicts the next sequence of 3D tokens.

These pristine, un-noised tokens are rendered into a 2D image.

We then add noise to this 2D rendered image, and use a frozen, pre-trained 2D Video Diffusion model as a "teacher" to evaluate the noise.

Using Score Distillation Sampling (SDS), the 2D diffusion model calculates the gradient of the log-probability of the rendered image and backpropagates this gradient through the differentiable renderer to update the 3DGS tokens. This avoids noising the 3D parameters directly.

3. Pure Self-Forcing with Differentiable Rendering (The ReconPhys Approach)
We can abandon the "diffusion" (noising/denoising) aspect for the temporal prediction and instead use a pure continuous-state AR prediction model, stabilized entirely by the Self-Forcing paradigm.

Recent architectures like ReconPhys use a feedforward network to predict dynamic 3DGS attributes from video without 3D ground truth.

Instead of multi-step diffusion, the model directly predicts the t+1 Gaussian tokens. The trick is applying Self-Forcing to the rendering loop.

The model performs an AR rollout: it predicts the t+1 tokens, renders them, calculates the photometric loss against the t+1 GT frame, and then uses its own predicted tokens as the KV-cached context to predict t+2. This allows the model to learn to auto-correct spatial drift via photometric gradients, completely bypassing the need for noise scheduling.

4. Pulling Back Diffusion Dynamics (DiffRep Sampling)
A highly theoretical but mathematically rigorous approach is "Diffusing Differentiable Representations." Rather than adding noise to the 3DGS tokens and breaking the renderer, you rewrite the reverse diffusion process itself in the parameter space of the 3DGS tokens.

You use a pre-trained 2D video diffusion model.

You "pull back" the 2D score function (the denoising gradients) into the 3D parameter space using the Jacobian matrix of the differentiable renderer. This allows you to perform true, multi-step diffusion sampling in the 3D token space, guided entirely by 2D image-space dynamics, without ever needing GT 3D tokens.

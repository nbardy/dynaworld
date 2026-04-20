# Renderer / Memory Acceleration Notes for dynaworld

## What is actually blowing up in this repo

The current wall is mostly the rasterizer, not the video encoder.

- [`dense.py`](/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/renderers/dense.py) materializes full `G x H x W` tensors such as `dx`, `power`, `alpha`, and `weights`.
- At the current full run, that means roughly `G=512`, `H=W=384`, and `T=16` training frames. In batched mode, the live render graph effectively scales with `T * G * H * W`.
- The encoder is not free, but this is why `32x32` smoke runs are fine and `384x384` dense runs fall over immediately on MPS.

So this is **not** mainly a ChopGrad-style temporal-backprop problem. It is a renderer working-set problem.

## What already helps here

We are already taking one useful shortcut:

- The model decodes plain RGB via `rgb_head`, not higher-order spherical harmonics; see [`dynamic_video_token_gs_implicit_camera.py`](/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/gs_models/dynamic_video_token_gs_implicit_camera.py).
- The renderers consume `rgbs` directly; see [`dense.py`](/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/renderers/dense.py) and [`tiled.py`](/Users/nicholasbardy/git/gsplats_browser/dynaworld/src/train/renderers/tiled.py).

That absolutely helps decode cost and avoids SH basis evaluation, but it does **not** remove the dominant memory terms, because those are the geometric/transmittance tensors, not the color channels.

## Paper takeaways that actually map to dynaworld

### 1. Original 3DGS: the win came from visibility-aware rasterization, not from a heavier network

Kerbl et al. make the rasterizer a first-class part of the method and explicitly call out a "fast visibility-aware rendering algorithm" as one of the three core contributions. That is the right mental model for us too: if we keep a dense PyTorch rasterizer that touches every Gaussian at every pixel, we are moving in the opposite direction of the original system design.  
Source: Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering* ([arXiv:2308.04079](https://arxiv.org/abs/2308.04079)).

### 2. Faster-GS is relevant, but mostly as a renderer/optimizer engineering baseline

Faster-GS is useful because it explicitly frames the gains as a bundle of broadly applicable implementation and optimization improvements, including numerical stability, Gaussian truncation, and gradient approximation, and reports up to `5x` faster training. The key point for us is that these are mostly **systems** wins around the 3DGS loop, not "make the transformer smaller" wins.  
Source: Hahlbohm et al., *Faster-GS: Analyzing and Improving Gaussian Splatting Optimization* ([arXiv:2602.09999](https://arxiv.org/abs/2602.09999)).

My read: this is directionally relevant, but most of the payoff only shows up once the renderer is already on an optimized CUDA-style path.

### 3. Speedy-Splat is directly relevant to our tiled path

Speedy-Splat targets two inefficiencies: sparse pixels and sparse primitives. That maps well to our current problem, because our dense path still behaves like "all primitives, all pixels." Their paper claims that more precise primitive localization plus pruning gives a large speedup without changing the overall 3DGS framing.  
Source: Hanson et al., *Speedy-Splat: Fast 3D Gaussian Splatting with Sparse Pixels and Sparse Primitives* ([arXiv:2412.00578](https://arxiv.org/abs/2412.00578)).

For dynaworld, the practical lesson is simple: tighter screen-space culling and fewer active gaussians per frame matter more than shaving a layer off the encoder.

### 4. StopThePop says better sorting/culling can reduce both memory and primitive count

StopThePop is motivated by view consistency, but the systems lesson is still useful: hierarchical resorting/culling can reduce effective Gaussian count a lot. They report that enforcing consistency allowed them to cut Gaussians by about half, with nearly doubled rendering performance and roughly `50%` lower memory.  
Source: Radl et al., *StopThePop: Sorted Gaussian Splatting for View-Consistent Real-time Rendering* ([arXiv:2402.00525](https://arxiv.org/abs/2402.00525)).

For us, that argues for investing in the tiled/culling path, not in keeping the dense path alive at high resolution.

### 5. Isotropic kernels are interesting but too invasive for the next baseline

Isotropic Gaussian Splatting argues that anisotropic kernels are a computation burden and reports very large speedups from switching to isotropic kernels. That is real evidence that covariance complexity matters. But this is a representation change, not a small renderer refactor, so it is probably a later ablation, not the immediate fix for dynaworld.  
Source: Gong et al., *Isotropic Gaussian Splatting for Real-Time Radiance Field Rendering* ([arXiv:2403.14244](https://arxiv.org/abs/2403.14244)).

## What likely helps immediately on MPS

These are the things most likely to pay off **without** writing custom CUDA:

1. Make the tiled renderer the default for `384x384`.
   - Dense is fine for `32x32` smoke tests.
   - Dense is the wrong default for `384x384` with `512` gaussians and `16` training frames.

2. Strengthen culling before rasterization.
   - Frustum/depth cull projected gaussians early.
   - Drop tiny screen-space splats with an explicit radius threshold.
   - Use a stronger opacity threshold during training.
   - Optionally keep only top-k gaussians per tile by projected alpha or area.

3. Use temporal render microbatching instead of the current binary choice between `batched` and `framewise`.
   - `2` or `4` frames per render/backward chunk is the likely sweet spot on MPS.
   - Same clip semantics, much better memory/throughput tradeoff.

4. Use coarse-to-fine render resolution.
   - Keep clip/video encoding at high resolution if needed.
   - Render the loss at `96` or `192` early, then ramp toward `384`.
   - This attacks the real term that hurts us: `H * W`.

5. Decode fewer active gaussians per frame, even if total latent capacity stays high.
   - Keep `512` gaussians available in principle.
   - Rasterize only the active subset by opacity / depth / projected area during early training.

6. Treat RGB-only as a fixed simplification and do not add SH right now.
   - SH would add cost before solving the actual bottleneck.

## What probably requires CUDA or custom kernels

These ideas are real, but they are not "small Python changes":

1. A fused tile rasterizer.
   - Project, bin, cull, evaluate alpha, and composite inside one kernel or a very small number of kernels.
   - The whole point is to avoid materializing repo-scale `dx`, `power`, `alpha`, and `weights` tensors in global memory.

2. Kernel-fused transmittance/compositing.
   - This is where the original 3DGS-style implementations beat a naive PyTorch dense rasterizer.

3. Faster-GS-style low-level optimizations.
   - Worth it once there is a real optimized renderer backend.
   - Not likely to pay off much if the underlying renderer is still Python tensor algebra on MPS.

4. More aggressive sort/cull data structures.
   - Very likely useful.
   - Also very likely to want a CUDA kernel if we want the full win.

## My current recommendation for dynaworld

Short term:

- Keep RGB-only.
- Keep the current decoder simple.
- Default to tiled at high resolution.
- Add temporal render microbatching.
- Add stricter culling and a coarse-to-fine render schedule.
- Consider an "active gaussian budget" per frame so we do not rasterize the full set every step.

Medium term:

- If `384x384` full runs are going to be a real baseline, a fused CUDA/gsplat-style renderer path is more important than squeezing another `2x` out of the encoder.

The main conclusion is blunt: for dynaworld's current setup, **renderer-side sparsity and fusion are more valuable than more encoder surgery**.

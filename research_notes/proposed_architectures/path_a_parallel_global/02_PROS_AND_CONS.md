# Path A: Pros & Cons (Parallel / Global)

## The Advantages (Why this is elegant)

1. **100% Parallel Inference & Training:**
   Because there is no recurrent state, you can render Frame 99 without ever computing Frames 1 through 98. During training, you can randomly sample any frame $t$, evaluate the opacity equation, and render it instantly.
   
2. **No BPTT (Backpropagation Through Time):**
   There is no causal dependency chain. Frame 2 does not depend mathematically on Frame 1. Therefore, you do not need ChopGrad, and you do not suffer from temporal gradient accumulation. The loss from Frame $t$ flows directly into the specific splats active at time $t$, and directly into the DiT backbone.

3. **Topological Freedom:**
   Because splats are born ($t_c$) and die ($\sigma_t$), the model handles extreme topological changes (explosions, objects appearing from behind walls) flawlessly by simply orchestrating a choreographed dance of fading splats in and out.

## The Disadvantages (The Brutal Scaling Walls)

1. **The Parameter VRAM Explosion:**
   If a static room requires 50,000 splats, a video of a person walking through that room for 10 seconds might require 500,000 splats (since splats "die" and new ones must spawn to represent the moving person). 
   - A 1-minute video might require 5,000,000 splats.
   - The memory required just to hold the physical tensor of $5,000,000 \times 16$ parameters on the GPU becomes the primary bottleneck, preventing scaling to long-form generation.

2. **The Cross-Attention Bottleneck:**
   The Splat Head must perform cross-attention between $N$ tokens and the Spatiotemporal features ($F$).
   - If $N$ scales linearly with the length of the video (to accommodate new movement), the $O(N \times \text{Sequence\_Length})$ attention matrix becomes computationally paralyzing.
   - You must invent clever sparse-attention mechanisms so that "early" tokens only look at "early" video frames, breaking the simplicity of the global attention model.

3. **Lack of True Physics:**
   The model does not actually learn "velocity" or "momentum". It just learns to memorize that an object was at $X_1$ at $t=1$ and a *different* splat should be at $X_2$ at $t=2$. It creates a flipbook of 3D states rather than a continuous physical simulation.
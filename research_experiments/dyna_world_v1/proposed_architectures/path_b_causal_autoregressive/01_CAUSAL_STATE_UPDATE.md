# Path B: Causal / Autoregressive Splat Decode

## The Inspiration: VAE Causal Caching
As we discovered in the ChopGrad paper, modern video autoencoders (like SVD or Wan 2.1) prevent temporal flickering by using **Causal Caching**. They append the features of the previous frame to the current frame. This forces the network to maintain a continuous, evolving state rather than guessing each frame independently.

We apply this exact philosophy to 3D Gaussian Splatting to solve the scalability issues of Path A.

## The Causal State Update
Instead of millions of short-lived splats, we use a **fixed, compact pool of splats** (e.g., $N = 100,000$). These splats are persistent; they exist for the entire video.

Instead of predicting their absolute position globally, our Splat Head predicts how they **change** from frame to frame.

### The Architecture
1. **Base State ($t=0$):**
   The Splat Head looks at the DiT features for the first frame ($F_0$) and predicts the initial room layout ($S_0$).
2. **Causal Step ($t=1$):**
   The Splat Head looks at the DiT features for the second frame ($F_1$) AND the physical state of the splats from the previous frame ($S_0$). 
   It predicts a **Delta** ($\Delta S_1$).
   $$ S_1 = S_0 + \Delta S_1 $$
3. **The Physics (Velocity & Acceleration):**
   To make the motion smooth and prevent the network from predicting erratic jumps, we can formalize the Delta as Velocity ($V$) and Acceleration ($A$).
   The network outputs $V_t$ and $A_t$ for each splat.
   $$ Position_t = Position_{t-1} + V_{t-1}(\Delta t) + \frac{1}{2}A_{t-1}(\Delta t)^2 $$

### Handling Topology (Appearing / Disappearing)
If an object moves behind a wall, the Splat Head doesn't destroy the splat. It simply predicts a negative Delta for the **Opacity ($\alpha$)** parameter. 
$$ \alpha_t = \alpha_{t-1} - 0.5 $$
The splat turns invisible. It continues to exist in the state vector, and the network can predict a positive Delta Opacity to make it reappear later.

### The Splat Head Design
The TokenGS cross-attention mechanism is now $O(1)$ with respect to video length!
- We have $N=100,000$ tokens.
- At frame $t$, they cross-attend **only to the DiT features of frame $t$** ($F_t$).
- The cross-attention matrix is tiny and constant, no matter how long the video is.
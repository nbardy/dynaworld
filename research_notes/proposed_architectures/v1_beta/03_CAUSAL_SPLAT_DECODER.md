# The Causal Splat Decoder

## The Flaw of Global Polynomials
Initially, we hypothesized that modeling 4D Gaussians as a single global polynomial equation (e.g., $Position(t) = c_0 + c_1t + c_2t^2$) would be the most memory-efficient way to render video. 

However, this has a fatal flaw during training:
1. The polynomial coefficients ($c_0, c_1, c_2$) must be predicted by attending to the features of the *entire* video sequence simultaneously.
2. If you render Frame 15 and calculate the loss, the gradient must update the global coefficients.
3. Because the global coefficients depend on the features of ALL frames, PyTorch must keep the computation graph of the **entire Video Diffusion Backbone** in VRAM simultaneously. **Instant OOM.**

## The Solution: Causal Splat Tracklets
We replace the VAE Video Decoder with a **Causal Splat Decoder**. Instead of predicting a global equation, we treat the splats as a recurrent state machine that updates frame-by-frame.

### Forward Pass (State Update)
At any frame $t$, the decoder receives two inputs:
1. $F_t$: The rich spatiotemporal features from the Video Diffusion Backbone for the current frame.
2. $S_{t-1}$: The physical state of the 3D Gaussians from the previous frame.

The Splat Head (a lightweight TokenGS-style cross-attention layer or MLP) predicts the **Delta ($\Delta$)**:
$$ \Delta S_t = \text{SplatHead}(F_t, S_{t-1}) $$
$$ S_t = S_{t-1} + \Delta S_t $$

### The Parameters of $S_t$
To maintain extreme efficiency, we use the SH0 trade-off. We restrict color to Spherical Harmonics Degree 0 (base RGB, 3 parameters). The recurrent state $S_t$ tracks:
- Position ($X, Y, Z$)
- Rotation (Quaternions $q_r, q_i, q_j, q_k$)
- Scale ($S_x, S_y, S_z$)
- Opacity ($\alpha$)
- Color ($R, G, B$)

### Why this is Superior
1. **Infinite Complexity:** A global polynomial cannot model a person walking around a corner, opening a door, and sitting down. A recurrent state machine handles arbitrarily complex, long-horizon motion natively.
2. **Topology Changes:** If an object is occluded or leaves the frame, the Splat Head simply predicts a negative $\Delta \alpha$ (opacity), fading the splats out of existence.
3. **Memory Unlocking:** Because the state updates causally ($t$ depends only on $t-1$), we can use **ChopGrad** to truncate the backpropagation through time, solving the VRAM explosion.
# ChopGrad Integration: Achieving $O(1)$ Memory

## The BPTT Memory Explosion
Because our Causal Splat Decoder is a recurrent network (where $S_t$ depends mathematically on $S_{t-1}$), training it naively requires **Backpropagation Through Time (BPTT)**.

If we calculate a pixel loss on Frame $N$ and call `.backward()`, the gradient flows:
1. Into the current splat state $S_N$.
2. Backwards in time to $S_{N-1}$, which requires the Video Diffusion Backbone features $F_{N-1}$.
3. Backwards to $S_{N-2}$, requiring $F_{N-2}$.
...all the way back to Frame 0.

To do this, PyTorch must hold the intermediate activations of the massive Video Diffusion Backbone in VRAM for *every single frame in the sequence*. Memory scales linearly ($O(N)$) with video length, making high-res training impossible.

## Enter ChopGrad (Truncated BPTT)
The Princeton researchers behind ChopGrad mathematically proved **Latent Temporal Locality**: the influence of a past latent embedding on a current pixel output decays exponentially over time. 

For our architecture, this means that the error gradient of the pixels at Frame $t=10$ provides almost zero useful learning signal to the diffusion features at Frame $t=2$. The gradient is essentially noise by that point.

### Implementation in Dyna World v1
We implement ChopGrad directly on the recurrent splat state $S$:

1. We define a truncation distance, $K$ (e.g., $K=4$ frames).
2. We unroll the forward pass of the Causal Splat Decoder.
3. Every $K$ frames, we call `.detach()` on the splat state tensor $S_{t-K}$. 

```python
# Pseudo-code for ChopGrad on Causal Splats
S_prev = initialize_splats()
loss = 0

for t in range(sequence_length):
    # 1. Forward pass through frozen/LoRA backbone
    F_t = video_backbone.get_features(video_latents, t)
    
    # 2. Causal Splat Update
    S_curr = splat_head(F_t, S_prev)
    
    # 3. Render and Loss
    img_rendered = splat_rasterizer(S_curr)
    loss += videometric_loss(img_rendered, gt_video[t])
    
    # 4. CHOPGRAD TRUNCATION
    if t % K == 0:
        # Backpropagate the accumulated loss up to this point
        loss.backward()
        
        # Detach the state! The gradient can no longer flow past this frame.
        # This frees the entire backbone computation graph from VRAM.
        S_curr = S_curr.detach()
        loss = 0 # Reset loss accumulator
        
    S_prev = S_curr
```

### The Result
Memory consumption drops from $O(N)$ to $O(1)$ (constant memory). We only ever hold $K$ frames of the Video Diffusion Backbone in VRAM at any given time. **We can now train on videos of infinite length.**
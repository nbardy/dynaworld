# Path B: ChopGrad & Memory (Causal)

## The Memory Trap of Autoregression
By making the Splat Head causal ($S_t$ depends on $S_{t-1}$), we successfully fixed the parameter explosion of Path A. We only ever need 100,000 splats, and our cross-attention matrix is tiny.

However, we introduced a new problem: **Backpropagation Through Time (BPTT).**
- If we render Frame 50 and calculate the Videometric Pixel Loss, the gradient must flow into $S_{50}$.
- Because $S_{50} = S_{49} + \Delta S_{50}$, the gradient flows backward into $S_{49}$.
- This chain continues all the way back to $S_0$.
- To compute this, PyTorch must hold the intermediate activations of the Video Diffusion Backbone for **all 50 frames** in VRAM simultaneously. **OOM Explosion.**

## The Solution: ChopGrad (Truncated BPTT)
This is exactly the problem the ChopGrad paper solved for VAE Decoders. We apply their mathematical proof of **Latent Temporal Locality** to our Splat State.

The physics of a splat at Frame 50 are heavily influenced by its velocity at Frame 49. But the gradient from Frame 50 provides almost zero useful learning signal to the DiT features back at Frame 5. The influence decays exponentially.

### The Implementation
We chop the gradient flow through the splat state every $K$ frames (e.g., $K=4$).

1. Unroll the causal update for 4 frames: $S_1 \rightarrow S_2 \rightarrow S_3 \rightarrow S_4$.
2. Render the frames, accumulate the pixel loss.
3. Call `loss.backward()`. The gradients update the Splat Head and the DiT backbone for frames 1-4.
4. **The Chop:** Call `S_4 = S_4.detach()`.
5. Continue to frame 5: $S_5 = S_4 + \Delta S_5$. 
6. Because $S_4$ is detached, when we calculate the loss for frames 5-8 and call `.backward()`, the gradient hits $S_4$ and **stops**. It does not travel back to frames 1-4.

## The Advantages (Why this is the holy grail)
1. **$O(1)$ Constant VRAM:** We only ever hold $K$ frames of the massive Video Diffusion Backbone in memory at any given time. We can train a 4D reconstruction model on a 10-minute video, and it will use the exact same amount of VRAM as a 10-second video.
2. **True Physics Tracking:** The model is forced to learn persistent object permanence, velocity, and momentum, rather than just memorizing a massive flipbook of static splats.
3. **Infinite Generation:** During inference, we can generate infinitely long 4D scenes by simply feeding new video frames into the DiT and causally updating our fixed pool of 100k splats.
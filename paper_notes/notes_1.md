# Notes 1: Memory Cost in the Current Video-Token Implicit-Camera Trainer

## Current setup

- Input clip: `train_frame_count=16`
- Resolution: `384x384`
- Tubelets: `(4, 16, 16)`, so stage-1 video grid is `4 x 24 x 24 = 2304` tokens
- Bottleneck grid after spatial downsample: `4 x 12 x 12 = 576` tokens
- Decoder queries: `1` global camera token + `1` path token + `8` GS tokens
- Gaussians per GS token: `64`
- Total explicit gaussians per decoded frame: `8 x 64 = 512`

The model encodes the whole clip once, then decodes one camera + one gaussian set per decode time. The big cost is not the token decoder itself; it is rendering those gaussians back to full images and keeping the backward graph alive.

## Where memory is likely going

### 1. Dense renderer is the main bottleneck

In `renderers/dense.py`, one frame builds full `G x H x W` tensors:

- `dx`: `[G, H, W, 2]`
- `power`: `[G, H, W]`
- `alpha`: `[G, H, W]`
- `weights`: `[G, H, W]`

For the current full run:

- `G = 512`
- `H = W = 384`
- `G x H x W = 512 x 384 x 384 = 75,497,472`

Rough fp32 sizes for one frame:

- `dx`: about `151M` floats, about `604 MB`
- `power`: about `75.5M` floats, about `302 MB`
- `alpha`: about `302 MB`
- `weights`: about `302 MB`

That is already well over `1.5 GB` of live activations for a single frame before counting gradients, projection intermediates, and PyTorch bookkeeping. If batched backward keeps many frame render graphs alive at once, memory blows up immediately.

### 2. Clip length multiplies render cost linearly

The encode path sees the whole `16`-frame clip once. The render path then pays roughly once per decoded frame. So render memory and compute scale roughly linearly with:

- number of decoded frames
- number of gaussians
- number of pixels

Moving from the small smoke path to the current full path changes all three:

- pixels: `384^2 / 32^2 = 144x`
- gaussians: `512 / 64 = 8x`
- decoded frames: `16 / 4 = 4x`

That is about `4608x` more render-side pressure than the tiny baseline, before constant factors.

### 3. Retained render graphs are expensive; retained encoder graph is not the main problem

`framewise` backward is exact gradient accumulation. It still uses the full-clip encoder graph. The difference is that each frame's render graph can be freed after that frame's backward, instead of keeping all frame render graphs alive until the end.

So:

- `batched`: exact gradients, worst memory
- `framewise`: exact gradients, much lower render memory

This is not a case where the encoder loses multi-frame context. Every frame loss still backprops through the same full-clip encoder and self-attention path.

### 4. Encoder memory is real, but secondary here

The encoder is not free:

- stage-1 attention runs on `2304` tokens
- attention cost scales roughly as `O(N^2)`
- several residual activations are retained through the forward

But compared with full-resolution dense rendering, it is still likely secondary. The biggest clue is that OOM happens inside dense rendering, not inside the encoder blocks.

### 5. Cross-attention is cheap relative to rendering

Cross-attention is only over about `10` learned query tokens attending to the video tokens. That is small compared with:

- stage-1 self-attention over `2304` tokens
- full-screen dense rendering over `512 x 384 x 384`

### 6. Tiled rendering should help a lot when splats stay spatially compact

Dense rendering always pays for every gaussian at every pixel.

Tiled rendering pays more like:

- per tile
- for gaussians whose support overlaps that tile

If gaussians are compact in screen space, tiled should drop both memory and compute substantially. If gaussians get very large and cover most of the image, tiled helps less. Even then, it usually has a better shape than dense.

### 7. RGB-only outputs help a bit, but are not the reason for the current OOM

The gaussian head only predicts RGB, not large spherical harmonic coefficients. That keeps per-gaussian parameter bandwidth smaller, which is good. But the current OOM is dominated by render activations, not by gaussian color parameter size.

## Does ChopGrad really apply here?

Not directly.

ChopGrad-style ideas are most natural when you have a recurrent or causal temporal decoder and want to truncate backpropagation through long temporal chains. That is not the current setup.

Current model shape:

- encode full clip once
- produce shared refined query tokens
- decode each time from the same shared query state plus a time embedding

So there is no long recurrent decoder state being unrolled across time. The temporal dependency is mostly through the shared clip encoder representation, not through a frame-to-frame hidden state.

That means:

- current `framewise` backward is **not** ChopGrad
- it is still the exact full-clip gradient
- it only changes when render graphs are freed

A ChopGrad-like approximation here would require a deliberate architectural approximation, such as:

- detaching temporal groups from a recurrent decoder state, or
- chunking decode times and detaching shared latent state between chunks

That would be an approximation, not a memory scheduling trick. It may be useful as an ablation, but it is not the obvious first lever for this codebase.

## Ranked practical recommendations

### 1. Highest impact: stop using dense full-image rendering for the full run

Default high-resolution training should use:

- tiled renderer, or
- a fused CUDA rasterizer when available

Dense full-image rendering at `384x384`, `512` gaussians, `16` frames is the wrong regime for MPS fp32 training.

### 2. Very high impact: keep full-clip encoding, but supervise fewer decoded frames per step

Good compromise:

- encode `16` input frames
- decode all times if needed for camera regularization
- render and score only `4` or `8` randomly chosen decode times per step

This preserves temporal context in the encoder while cutting render cost almost linearly.

### 3. Very high impact: progressive splat count

Start with fewer gaussians, then grow:

- early: `128` to `256`
- later: `512`

This keeps the early optimization cheap while the scene and camera are still coarse anyway.

### 4. High impact: resolution curriculum

Warm up at lower render size, then raise it:

- `96` or `128`
- then `192`
- then `384`

Since pixel cost is quadratic in image size, this is one of the cleanest knobs available.

### 5. Medium impact: mixed precision and CUDA-first execution

On CUDA, fp16 or bf16 plus an efficient rasterizer should help materially. On current MPS fp32, memory headroom is much worse.

### 6. Medium impact: temporal render microbatching

Keep the current exact objective, but render/backward in groups of `2` or `4` frames instead of `1` or `16`. This is mostly a throughput knob between:

- safest memory: framewise
- best throughput if it fits: batched

### 7. Medium-to-low impact: checkpoint encoder blocks

Gradient checkpointing the video encoder and maybe query cross-attention can save some activation memory, but it will not fix the main render bottleneck by itself.

### 8. Lower impact than it seems: make the encoder even cheaper

Bigger patch compression, a cheap conv stem, or more aggressive early downsampling are all valid. They are worth doing, but only after acknowledging that the first-order problem is still the renderer. Otherwise the project risks shrinking the transformer while the rasterizer remains the actual bottleneck.

## Bottom line

The current OOM and slowdown are mostly render-side, not encoder-side.

The cleanest path is:

1. use tiled or fused rasterization for full-res runs
2. keep full-clip encoding but render only a subset of frames per step
3. use a curriculum over splat count and image size

ChopGrad is not the natural answer for the present architecture. It only becomes compelling if the model grows a genuinely recurrent temporal decoder whose backward graph through time is itself the main memory problem.

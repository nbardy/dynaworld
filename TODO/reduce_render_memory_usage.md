# Reduce Render Memory Usage

## Goal

Reduce peak memory usage and backward-time pressure in the `dynaworld` render path, especially for the implicit-camera video baseline at larger resolutions such as `384 x 384`.

The current problem is dominated by renderer-side activation memory, not just model parameter count.

## Current Bottleneck

The dense renderer in `src/train/renderers/dense.py` materializes full Gaussian-by-pixel interaction tensors:

- `dx`: `[G, H, W, 2]`
- `power`: `[G, H, W]`
- `alpha`: `[G, H, W]`
- `T_map`: `[G, H, W]`
- `weights`: `[G, H, W]`

This is effectively dense all-pairs work between:

- every Gaussian
- every pixel

At `G = 512` and `H = W = 384`, the dense renderer builds very large fp32 tensors before backward is even considered. Autograd then needs to retain enough of that graph for gradient computation, which is why MPS runs out of memory quickly.

## What We Already Added

The video-token trainer now supports:

- `batched` reconstruction backward
- `microbatch` reconstruction backward
- `framewise` reconstruction backward

This helps because temporal microbatching reduces peak render-graph memory by only holding a subset of decoded frames at once.

Current observed behavior on full `384` dense rendering:

- `microbatch=4` still OOMs on MPS
- `microbatch=2` completes, but slowly

So temporal chunking helps, but it does not fix the underlying dense-render memory shape.

## Immediate Practical Levers

### 1. Reduce rendered frames per step

Current render cost scales roughly linearly with decoded frame count.

Primary knobs:

- reduce `TRAIN_FRAME_COUNT`
- optionally supervise only a subset of decoded times per step

### 2. Reduce Gaussians per step

Render cost also scales roughly linearly with Gaussian count.

Primary knobs:

- reduce `TOKENS`
- reduce `GAUSSIANS_PER_TOKEN`

### 3. Prefer tiled rendering sooner

The tiled renderer is structurally closer to the right approach than the dense renderer because it avoids evaluating every Gaussian at every pixel.

Practical follow-up:

- bias `auto` toward tiled earlier
- test tiled as the default for larger image sizes

### 4. Add render-time curriculum

Start training in a cheaper regime, then scale up.

Candidate curriculum axes:

- lower image size first, then raise
- fewer gaussians first, then raise
- fewer supervised frames first, then raise

### 5. Consider selective frame supervision

Encode a full temporal clip, but only render a subset of times each step.

This keeps full-clip temporal context while limiting renderer pressure.

## Medium-Term Technical Follow-Ups

### A. Improve the tiled path

Current tiled rendering still builds nontrivial intermediate tensors per tile. There is room to:

- tighten culling
- reduce false positives in tile assignments
- lower temporary tensor counts

### B. Make temporal render microbatching more flexible

Current trainer supports chunked reconstruction backward, but there is more to explore:

- dynamic chunk sizing by resolution / Gaussian count
- separate chunking policy for train vs eval
- different logging/eval render policies from training render policies

### C. Consider render-state reuse only if mathematically valid

Do not add graph retention tricks unless they preserve the actual gradient path. `retain_graph=True` is necessary for repeated backward through a shared clip graph, but it is not itself a memory reduction trick.

### D. Keep the encoder cheap, but do not confuse that with the core problem

Early compression in the video encoder helps total runtime and activation count, but the dominant memory wall is still the renderer at high output resolution.

## Papers And Systems To Follow Up On

### 1. Original 3D Gaussian Splatting

Why it matters:

- it uses a fast visibility-aware tiled renderer instead of naive dense Gaussian-by-pixel evaluation
- this is the right baseline mental model for how rendering should scale

Follow-up:

- read the rendering section again with attention to why visibility-aware rasterization is part of the method, not just an implementation detail

Link:

- https://github.com/graphdeco-inria/gaussian-splatting

### 2. Faster-GS

Why it matters:

- tighter tile culling
- per-Gaussian backward
- kernel fusion
- explicit emphasis on reducing both VRAM and runtime

Follow-up:

- identify which ideas transfer conceptually to this repo even without a custom CUDA backend
- separate "algorithmic improvements we can borrow" from "CUDA-only engineering wins"

Link:

- https://fhahlbohm.github.io/faster-gaussian-splatting/assets/hahlbohm2026fastergs.pdf

### 3. Efficient Differentiable Hardware Rasterization for 3D Gaussian Splatting

Why it matters:

- fixed-memory hardware-style rasterization
- specifically targets backward-pass memory/performance limitations
- useful as a north star for where the renderer architecture should go

Follow-up:

- understand what parts are fundamentally graphics-pipeline dependent
- identify whether any batching or reduction ideas can inform a software renderer redesign

Link:

- https://arxiv.org/abs/2505.18764

## Concrete TODOs

### Short term

1. Default large-resolution training to tiled or earlier tiled `auto`
2. Add a "render only K frames from the encoded clip" training option
3. Benchmark `(train_frame_count, gaussians_per_token, renderer_mode, temporal_microbatch_size)` systematically
4. Measure memory separately for:
   - encoder forward
   - decode
   - renderer forward
   - renderer backward

### Mid term

5. Audit the tiled renderer for unnecessary temporary tensors
6. Prototype tighter culling / smaller tile lists
7. Evaluate whether train-time frame subsampling preserves convergence well enough to be the default large-resolution path

### Long term

8. Decide whether this project needs:
   - a more optimized software rasterizer
   - a fused custom backend
   - or integration with an external renderer stack

## Success Criteria

This work is successful when:

1. `384 x 384` training does not require pathological framewise fallback on MPS
2. larger clips can train without dense-render OOM at practical settings
3. renderer memory is the result of an intentional design choice, not an accidental dense broadcast implementation

I had a AI try to help me with these memory issues, but youre a smarter AI,

Go find papers go find out how we can train a video=>Splat=>Render model better with WAY lower memory requirements.

I'm trying to test 384x384 on my mac as a baseline, should train fast and use minimal memory so we can scale up on GPU to huge time horizons and resolutions

Let's explain a few different directions and areas we can use to tackle this. Also we want to tackle longer time horizons, right now we encode a video to a fixed number of tokens. And then we decode each token to a fixed number of splats. And each token can be decode with a t-conditioned MLP, so that we get a token for each frame.

I'd like to be able to train along long time horizons, also dense sets of tokens should be expandable to multiple frames so we can do this fats in WGPU in the client without sending over HUGE amounts of splats in the client.

And the renderer seems to be one of the huge bottlenecks.

I'm wondering if we could train diffusion teacher forcing or Auto regression or something.

I know we can do tricks like train on random N frames decoded, OR train at lower resolutions.

AND we will do this in time.

But for now I need to speed this up fully

I’m pulling together the papers and a concrete plan for your Mac baseline at the same time, focused on the two big tracks here: cutting renderer memory now, and redesigning the video→token→splat path so longer horizons and lighter client playback stay practical.

The split is getting clearer: memory and long-horizon scaling look like separate problems needing different fixes. I’m organizing the answer around three buckets now—leaner rasterization, chunked or hierarchical temporal training, and compact dynamic splat representations.

The picture is settling: this is really three problems, not one—rasterization memory, primitive allocation, and temporal state—and there’s solid paper support for each path.

Thought for 20m 44s

Your pasted note is pointing at the right enemy: the memory wall is activation storage in the differentiable rasterizer, not raw parameter count. So the highest-leverage fixes are the ones that reduce rendered pixels × active splats × decoded times, not the ones that merely shrink a small MLP.

I’d attack this in three layers: make render supervision sparse, make the splat representation temporal and variable-rate, and give the client a renderer that is budgetable instead of perfect-but-expensive.

Render far less during training, while keeping full-res inputs.
Speedy-Splat explicitly frames rendering cost around both the number of Gaussians and the number of pixels processed per Gaussian, then wins by tightening tile localization and pruning during training. Differentiable hardware rasterization goes after the same bottleneck from the backend side and finds that hardware-style rasterization with 16-bit render targets is a strong memory/speed trade-off. For your setup, the practical translation is: keep the encoder at 384×384 if that matters for the baseline, but only backprop render loss on a few sampled frames and a few sampled tiles per frame, with framewise or tiny temporal microbatches. That is the cleanest immediate way to slash activation memory without throwing away the full-resolution signal.
Make the client renderer cheap and controllable.
A lot of recent GS work is basically saying “stop treating sorted alpha blending as sacred.” Weighted Sum Rendering removes sorting by approximating alpha blending with weighted sums. StochasticSplats removes sorting and turns render cost into a Monte Carlo sample budget. Mobile-GS identifies alpha blending plus depth sorting as the primary bottleneck on edge devices and uses an order-independent variant plus a learned correction stage. Hybrid transparency is another middle ground that improves coherence and reduces render/optimization cost. For WGPU, this is a very good fit: a sort-free or hybrid renderer is much more realistic to ship than a huge, perfectly sorted splat stream every frame. My bias would be to train the representation first with the most stable renderer you have, then distill or fine-tune into a cheaper sort-free client renderer.
Stop using a rigid token budget, and stop decoding a fresh dense splat cloud for every frame.
Video tokenization papers are moving away from fixed 3D token grids. AdapTok and EVATok allocate tokens according to temporal content instead of spending them uniformly. VTok keeps one key frame’s spatial state and encodes later frames as residual tokens. VideoFlexTok goes coarse-to-fine with variable-length tokens and reports 81-frame videos with only 672 tokens, 8× fewer than a comparable grid tokenizer. Feed-forward GS papers are moving the same way: TokenGS uses learnable Gaussian tokens so primitive count is no longer tied to input resolution or number of views; GlobalSplat does “align first, decode later”; SparseSplat adapts Gaussian density to scene complexity and gets strong quality with far fewer Gaussians. For your model, that points to a tokenizer with three token types: a persistent scene core, motion tokens, and optional detail tokens. The client then expands those locally across many frames instead of the server emitting a full splat set per frame.
Put temporal structure into the splats themselves.
The dynamic-GS compression papers are very aligned on this point: a lot of time redundancy should live in the representation, not in repeated per-frame decoding. MEGA cuts per-Gaussian appearance cost by using direct color plus a shared lightweight predictor. TC3DGS prunes by temporal relevance and interpolates trajectories across frames. P-4DGS explicitly borrows intra/inter-frame prediction ideas from video compression. SpeeDe3DGS groups motion into shared SE(3) transforms instead of running a separate neural deformation for every Gaussian. The version I would try in your architecture is: decode anchor splats once, then predict low-rank temporal coefficients or grouped transforms, e.g.
g
i
	​

(t)=g
i
0
	​

+
k
∑
	​

a
ik
	​

B
k
	​

(t)

for position/scale/opacity, with a tiny residual MLP only for the hard-motion subset. That gives you exactly what you want: one dense token can span many frames, and the client can evaluate the basis functions locally in WGPU without downloading massive splat payloads.

For longer horizons, use chunked rollout over compressed scene/motion tokens.
So: should you use teacher forcing, autoregression, diffusion, or some hybrid? I think yes, but not over raw per-frame splats. Diffusion Forcing combines causal next-token prediction with diffusion-style denoising so variable-length rollouts stay possible. Self Forcing trains on self-generated history instead of only ground-truth context. Rolling Forcing reduces drift by jointly denoising windows and keeping attention sinks as long-range anchors. ChopGrad is especially relevant here: it shows that truncated backprop through local temporal windows can change memory from linear-in-frames to constant for pixel-wise video losses. ReCoSplat and LongSplat show that long-sequence 3DGS gets much more practical once you compress caches and historical Gaussians instead of keeping everything live forever. My take is: do chunked AR over scene/motion tokens, use teacher forcing early, self-generated history later, and keep render loss local in time.

One caution: Faster-GS is a good source of ideas, not a drop-in Mac fix. A lot of its win comes from fused CUDA-side optimizer/rasterizer engineering, and one of its biggest speedups—the per-Gaussian alpha-backward pass—also increases VRAM. So I would treat it as a blueprint for a future custom backend, not as the immediate cure for a pure PyTorch/MPS renderer.

What I would do right now for your 384×384-on-Mac baseline:

Keep the input at 384×384, but do render supervision on only 2 random frames per clip, and on each of those render only 2–4 random tiles (96² or 128²), not the full frame. Once every N steps, render one full frame as a calibration step. By the cost proxy in your note, that kind of change can cut the dominant renderer term by roughly an order of magnitude to two orders of magnitude without changing the encoder input distribution.
Start with 8-frame clips and 128–256 total splats, not 16 frames and 512 splats. Then use temporal microbatching of 1–2 rendered frames. That is the boring answer, but it is still one of the biggest wins on your current stack.
Add prefix/drop-budget training for splats. Matryoshka GS shows that sampling random splat budgets during training can make any prefix of the representation usable. That is almost exactly what you want for client scalability: one model, multiple budgets, one ordered splat stream.
Replace token -> MLP(t) -> fresh frame splats with token -> anchor splat + motion/detail coefficients. Do that before you spend time on fancy generative training. It is the structural change that helps both memory and bandwidth.
Only after that baseline is stable, add chunked AR/diffusion rollout over the compressed scene/motion tokens, with truncated temporal backprop. That is where teacher forcing and self-conditioning start paying off for long horizons instead of just making training more complicated.

If I had to bet on the highest-ROI design for you, it would be this: 384 input, sparse tile render training, anchor splats with low-rank motion bases, random splat-budget training, and a sort-free WGPU client renderer. That path gives

you the fastest immediate memory relief and also points toward the longer-horizon, low-bandwidth system you actually want.

>Make the client renderer cheap and controllable.

What about "FasterGS"? What do they do an cheaper render? or lower memory render?

>prefix/drop-budget training

What is this explain in shapes and psuedo code

I’m pulling the paper details now and translating them into concrete tensor shapes and training loops, and I’m checking which Faster-GS tricks actually reduce training memory versus only speeding up CUDA-side rendering.

I’ve got the split now: Faster-GS is mainly a better tiled-CUDA training stack, not really the cheap low-memory client renderer answer, and its flashiest backward rewrite is faster but can increase VRAM.

Thought for 10m 42s

Yeah — Faster-GS is worth reading, but the key thing is that it is mostly a systems paper, not a new approximate renderer. It keeps the standard tiled 3DGS paradigm, stays close to original 3DGS, and explicitly avoids leaning on heavy pruning/compression in its main setup. The wins come from making that pipeline much less memory-bound. They report up to 5× faster training, up to 30% lower VRAM, and an inference-optimized forward pass that is more than 3× faster on their benchmarks.

What Faster-GS is actually doing

The short version: it makes the renderer/backward path less wasteful.

Tighter splat bounds + tile culling. They replace loose square bounds with tighter opacity-aware rectangular bounds, then cull tile overlaps more aggressively so fewer useless Gaussian–tile pairs ever enter the rasterizer. That makes the tiled lists smaller and the forward pass cheaper.
Separate sorting. Instead of one big sort over 64-bit keys, they split sorting into depth-sort then tile-sort. That reduces sort cost and also cuts VRAM because the second stage can use 16-bit tile keys for typical resolutions.
Per-Gaussian backward. This is a big one for training speed. They parallelize alpha-gradient work over Gaussians instead of pixels, which cuts atomic operations by about the tile size, i.e. 256 for 16×16 tiles. That is one of the major speedups.
Shared-memory staging of alpha state. They then optimize that backward further by staging alpha/transmittance state in shared memory, which they say improves that kernel runtime by up to 2×.
Lower-overhead bookkeeping. They call out explicit visibility masks and explicit gradient handling as reducing VRAM overhead, and they fuse optimizer work into backward so fewer extra buffers are needed. In one table, fused updates reduce VRAM from 6.0 GiB to 5.6 GiB on their setup; skipping invisible Gaussians drops it further to 5.1 GiB, though that starts to change behavior/quality tradeoffs.
Better memory locality. They periodically z-order Gaussians in memory so nearby splats are nearby in buffers, which improves cache behavior and reduces warp divergence.

The important gotcha: their biggest speed win is not also their biggest memory win. The per-Gaussian backward stores extra checkpointed blending state, and Faster-GS explicitly says this is the one change that increases VRAM. So the paper is not “everything gets cheaper”; it is “spend memory more intelligently and remove the worst stalls.”

What transfers to your setup, and what does not

For your current setup, where your own note says the real wall is activation memory in the differentiable rasterizer, not parameter count, Faster-GS is directionally very relevant — but not as a drop-in transplant.

The parts that transfer best to a future WGPU/client or custom renderer are:

tighter bounds / better tile culling
smaller sort keys / split sorting
explicit visibility masks
memory-local ordering of splats/tokens

The parts that are mostly CUDA training-kernel tricks are:

per-Gaussian backward
shared-memory alpha-state replay
fused backward + Adam

So if your current Mac baseline is still using a dense broadcast-style renderer, Faster-GS does not solve that by magic. The big lesson to steal is: stop doing dense gaussian×pixel materialization; move toward tiled lists and compact visibility. That is very aligned with the bottleneck your pasted note identified.

About “prefix/drop-budget training”

I’d sharpen the terminology a bit:

Prefix-budget training = learn an ordered list, and budget k means “take the first k items.”
Drop-budget training = randomly mask things out.

For your use case, prefix-budget is the one you actually want, because it gives you a deterministic nested stream for the client. Smaller budgets are strict subsets of larger ones, which is exactly what you want for “send a compact thing once, expand locally, and choose quality dynamically.” That is also the core idea in Matryoshka Gaussian Splatting: order primitives by an importance score, sample a random budget each step, and optimize both the chosen prefix and the full set. They report that opacity-descending ordering and prefix + full loss work best.

Shapes in your model

Using your setup:

encoder output tokens: z with shape [B, M, D]
M = number of video tokens
each token decodes to K splats
per-splat params P ≈ 14 if you use
xyz(3) + scale(3) + quat(4) + opacity(1) + rgb(3)

So the decoded anchor splats are:

base_splats: [B, M, K, P]

If you also make splats expand across time using a basis:

motion_coeffs: [B, M, K, R, P_dyn]
basis(t):      [F, R]
decoded(t):    [B, F, M, K, P]

where R is the number of temporal basis functions and P_dyn is only the subset you animate, like xyz/scale/opacity.

In your current baseline from the pasted note, the concrete numbers were roughly:

M = 8 tokens
K = 64 splats/token
G = M*K = 512 splats per frame

So if you keep only half the tokens and half the local splats, you go from:

full:   [B, F, 8, 64, P]  -> 512 splats/frame
prefix: [B, F, 4, 32, P]  -> 128 splats/frame

That is a 4× reduction in the Gaussian count term. Since your own cost proxy is roughly

render cost ~ decoded_frames * gaussians * pixels

that gives you about a 4× renderer-side cut at fixed frames and resolution.

The simplest useful version

Start with token-level prefixing first.

Score each token with a scalar importance.
Sort tokens descending by score.
Sample a random keep ratio r.
Keep the first ceil(r*M) tokens.
Render both:
the prefix
the full set
Train with a combined loss.

That already gives you a client knob: “render first 2 / 4 / 8 tokens.”

Pseudo code:

# video: [B, T, H, W, 3]
# encoder -> z: [B, M, D]

z = encoder(video)                         # [B, M, D]

token_score = score_head(z).detach()       # [B, M]
perm = argsort(token_score, descending=True)
z_ord = gather(z, perm)                    # [B, M, D]

r = uniform(r_min, 1.0)
M_keep = ceil(r * M)

z_pref = z_ord[:, :M_keep]                 # [B, M_keep, D]

# decode only a few sampled frames, not all frames
times = sample_render_times(T, n=2)

splats_full = decode(z_ord,  times)        # [B, F, M,      K, P]
splats_pref = decode(z_pref, times)        # [B, F, M_keep, K, P]

img_full = render(splats_full)
img_pref = render(splats_pref)

loss = 0.5 * recon(img_full, gt[:, times]) + \
       0.5 * recon(img_pref, gt[:, times])

loss.backward()

That is basically “Matryoshka, but at the token level instead of a flat scene-Gaussian list.” The reason it fits you well is that your client naturally wants to receive a compact ordered token stream, not a giant unordered splat dump.

The better version for your architecture

Your decoder has a natural two-level hierarchy:

level 1: tokens
level 2: splats within each token

So do nested budgets at both levels.

Shapes:

z                 : [B, M, D]
base_splats       : [B, M, K, P]
ordered base      : [B, M, K, P]
prefix base       : [B, M_keep, K_keep, P]
decoded in time   : [B, F, M_keep, K_keep, P]

Pseudo code:

z = encoder(video)                                   # [B, M, D]
base, motion = decode_anchors(z)                     # [B, M, K, P], [B, M, K, R, P_dyn]

# token ordering
tok_score = token_importance(base).detach()          # [B, M]
tok_perm  = argsort(tok_score, descending=True)
base   = gather_tokens(base, tok_perm)               # [B, M, K, P]
motion = gather_tokens(motion, tok_perm)             # [B, M, K, R, P_dyn]

# local splat ordering inside each token
spl_score = splat_importance(base).detach()          # [B, M, K]
spl_perm  = argsort(spl_score, dim=-1, descending=True)
base   = gather_splats(base, spl_perm)               # [B, M, K, P]
motion = gather_splats(motion, spl_perm)             # [B, M, K, R, P_dyn]

# stochastic budgets
r_tok = uniform(r_tok_min, 1.0)
r_spl = uniform(r_spl_min, 1.0)
M_keep = ceil(r_tok * M)
K_keep = ceil(r_spl * K)

times = sample_render_times(T, n=2)
tiles = sample_tiles(H, W, n=2)                      # optional, for memory

full = expand_in_time(base, motion, times)           # [B, F, M,      K,      P]
pref = expand_in_time(
    base[:, :M_keep, :K_keep],
    motion[:, :M_keep, :K_keep],
    times
)                                                    # [B, F, M_keep, K_keep, P]

img_full = render_tiles(full, tiles)
img_pref = render_tiles(pref, tiles)
gt_sub   = crop_tiles(gt[:, times], tiles)

loss = 0.5 * recon(img_full, gt_sub) + \
       0.5 * recon(img_pref, gt_sub)

loss.backward()

Why this is better for you:

token prefix controls network bandwidth / decode width
local splat prefix controls renderer load
both together give a nested client-side LoD contract

So the browser can choose:

8 tokens × 64 splats = 512
4 tokens × 64 splats = 256
4 tokens × 32 splats = 128
2 tokens × 32 splats = 64

and every smaller setting is a prefix of the bigger one.

How to score tokens/splats

Matryoshka GS sorts by opacity descending and finds that it works best across budgets. That is a strong default if you want something simple and stable.

For your model I’d try:

token_score = mean(topk(opacity[:, token, :], k=8))
splat_score = opacity

or, slightly better:

token_score = w1 * mean_opacity + w2 * mean_projected_area + w3 * mean_color_energy

But I would start with opacity. It is boring, and boring wins a lot.

One practical tweak: MGS dynamically reorders every iteration because importance changes during training. In your temporal-token model, I would not reorder every step; I’d use an EMA score and reorder every few hundred steps so the client-facing ordering does not jitter like crazy. MGS itself does dynamic reordering every iteration.

One important caveat

Prefix-budget training is not your first pure-speed trick, because it adds a second render pass each step. MGS gets away with that because its point is continuous LoD training, and each step still uses only two renders regardless of budget. In your setup, where the renderer is already the bottleneck, I would only turn this on together with sparse frame/tile supervision.

So for your Mac baseline, the practical recipe is:

render only 2 random frames
optionally render only 2–4 random tiles
apply prefix + full on those same sampled frames/tiles
start with token-prefix only
add local splat-prefix once that is stable

That gives you the LoD property you want without immediately exploding training cost.

The bottom line

My take:

Faster-GS helps you understand how to build a less stupid rasterizer/backward path.
Prefix-budget training helps you build a representation the client can budget dynamically.
They solve different problems.

If I were prioritizing for you right now, I’d do this order:

replace dense render with tiled/compact render path
sample few frames / few tiles
add token-prefix training
add token+local-splat two-level prefix
later, move to anchor splats + temporal basis so the same ordered representation expands over long horizons on-device

That is the combo that most directly attacks both your memory wall now and your client bandwidth / long-horizon goal later.

okay how does "per gaussian" backward work? and does that need CUDA or could we do it in torch?"

And same for tighter bounds and culling and separate sorting.

Can you take those insights all all the other render insights and right a new torch only differentiable rasterize that minimizes memory usage and compute cost

--

And lets leave the tiel size or dynamic budgets as follow up, I will do that later for sure,  but for now we want to focus on the cost of full render and bringig that down as even when we add tiling or splat budgets this will still compound in the speed tradeoffs.

I’m sorting out which Faster-GS-style tricks actually survive in pure Torch versus needing custom CUDA, then I’ll turn the transferable ones into a concrete, lower-memory full-frame differentiable rasterizer.

The portable core is clearer now: tighter bounds, culling, separate sorting, and backward recomputation all work in Torch, but the CUDA-specific throughput tricks do not. I’m turning that into a custom autograd.Function that composites full frames using cropped per-Gaussian supports, saves tiny checkpoints, and avoids materializing giant G×H×W tensors.

I’ve validated the pure-Torch backward on a small dense reference: same image and gradients, but without ever materializing the huge [G,H,W] alpha/weight tensors. I’m packaging it as a standard alpha compositor with manual backward, tight support boxes, and chunked recomputation.

Thought for 29m 46s

Yes — per-Gaussian backward does not fundamentally need CUDA. CUDA is what makes it really fast, but the math transfers to Torch just fine.

The original per-pixel 3DGS backward has each pixel thread walk splats and atomically add into Gaussian gradients, which creates heavy contention. Taming 3DGS flips that around and parallelizes over splats instead of pixels; Faster-GS summarizes the same idea and notes that this can reduce atomic additions by roughly the number of pixels in a tile, commonly 256 for 16×16 tiles, although the checkpointed alpha state they use is also the one change that increases VRAM.

For your setup, the big opportunity is even simpler: your current dense renderer is materializing huge [G,H,W] intermediates like dx, power, alpha, and weights, so the bottleneck is activation memory in the rasterizer rather than parameter count.

I wrote a Torch-only rasterizer that attacks exactly that:

memory_efficient_splat_rasterizer.py

What I changed

This rasterizer does full-frame rendering but never builds dense G×H×W tensors.

It uses:

one stable depth sort
tight opacity-aware per-splat bounds
local patch compositing only inside each splat’s box
custom backward that recomputes local alpha patches instead of saving them

That is the right Torch-only trade: much lower memory, somewhat more recompute.

Speedy-Splat’s core observation is that render cost scales with both the number of Gaussians and the number of pixels processed per Gaussian, and it speeds things up by making those per-Gaussian pixel footprints much tighter with SnugBox/AccuTile. Faster-GS also uses opacity-aware bounds and tile-based culling for the same reason.

How “per-Gaussian backward” works

For one pixel, front-to-back compositing is

A
i
	​

=A
i−1
	​

+T
i−1
	​

α
i
	​

c
i
	​

T
i
	​

=T
i−1
	​

(1−α
i
	​

)

and the final color is

C=A
G
	​

+T
G
	​

bg

where:

A
i
	​

 is accumulated premultiplied color after splat i
T
i
	​

 is transmittance after splat i
α
i
	​

 is this splat’s opacity at the pixel
c
i
	​

 is RGB

The neat trick is: in backward, you do not need to have saved every intermediate alpha/transmittance tensor.

If you know the final transmittance T
G
	​

, then when walking splats backwards you can recover the prefix transmittance exactly:

T
i−1
	​

=
1−α
i
	​

T
i
	​

	​


That gives an exact reverse scan.

Let g=∂L/∂C per pixel. Then:

∂c
i
	​

∂L
	​

=g(T
i−1
	​

α
i
	​

)
∂α
i
	​

∂L
	​

=⟨g,T
i−1
	​

c
i
	​

⟩−g
T
	​

T
i−1
	​


where g
T
	​

 is the scalar adjoint of transmittance. It updates as:

g
T
(i−1)
	​

=⟨g,α
i
	​

c
i
	​

⟩+g
T
(i)
	​

(1−α
i
	​

)

So the backward pass can be:

reverse over splats
recompute just that splat’s local alpha patch
recover T
i−1
	​

 from T
i
	​

accumulate gradients for that one splat
update the reverse scan state

That is the Torch analog of “per-Gaussian backward”: each Gaussian owns its gradient accumulation locally, instead of every pixel throwing atomics into shared Gaussian buffers.

Does this need CUDA?

No. In Torch, the natural version is a torch.autograd.Function with:

a custom forward that saves only compact metadata
a custom backward that recomputes local patches

That is exactly what I implemented.

What CUDA buys you is:

lower launch overhead
real splat-parallel kernels
shared memory / warp shuffles
atomics that are cheap enough to matter less

But for your 384×384 Mac baseline, a Torch custom autograd path is a very sane step.

Tighter bounds and culling

This also ports cleanly to Torch.

If your 2D Gaussian is parameterized by a conic / precision matrix

Q=[
a
b
	​

b
c
	​

]

and

α(x,y)=oexp(−
2
1
	​

q(x,y)),q=adx
2
+2bdxdy+cdy
2

then for an alpha cutoff τ
α
	​

, the useful support is where

q≤−2ln(τ
α
	​

/o)

From that ellipse, the tight axis-aligned half-extents are

h
x
	​

=
ac−b
2
τc
	​

	​

,h
y
	​

=
ac−b
2
τa
	​

	​


with τ=−2ln(τ
α
	​

/o).

That is the practical non-tiled version of SnugBox: compute an opacity-aware ellipse box, clip it to the image, and only evaluate that patch. Speedy-Splat explicitly shows that tighter localization reduces downstream work, and Faster-GS calls out the same opacity-aware bounding idea as reducing false positives in per-tile splat lists.

In the file I also keep an exact ellipse mask inside the bounding box, so you are not forced to process the whole rectangle.

What about separate sorting?

For a tiled renderer, Faster-GS notes that doing sorting in two stages — first depth order, then per-tile list formation — reduces both VRAM and sorting time compared with one big sort over tile/Gaussian pairs.

But since you explicitly asked to ignore tiles for now, I would not implement separate sorting yet.

For this full-frame Torch rasterizer, the right v1 is:

one stable global depth sort
then per-splat local patch compositing

That gives you most of the memory win without introducing tile-pair duplication buffers, which are exactly the kind of extra allocation you’re trying to avoid right now.

Why this should be much cheaper than your current renderer

Your current dense path effectively pays for every Gaussian at every pixel and stores huge broadcast tensors.

This replacement changes the scaling from:

memory∼O(GHW)

to roughly:

memory∼O(HW+G)

plus tiny local patch temporaries.

And compute becomes:

compute∼O(
i
∑
	​

area(bbox
i
	​

))

instead of always O(GHW). Worst case is still bad if every splat covers the whole image, but in the regime you actually want, this is a huge practical reduction.

A few implementation choices I baked in

The module takes:

means2d:   [G, 2]
conics:    [G, 3]   # [a, b, c]
colors:    [G, 3]
opacities: [G]
depths:    [G]
out:       [H, W, 3]

It uses:

exact front-to-back alpha blending
no gradient through the sort order itself
fp32 accumulation for image and transmittance
optional fp16 local patches

That last one is intentional. A recent differentiable hardware rasterization paper found that 16-bit render targets were the best efficiency/accuracy trade-off, so the Torch analogue is: local math can go lower precision, but keep the long-lived accumulation state in fp32. That’s an inference from their hardware result, not a guaranteed PyTorch/MPS law, but it is a good default direction.

What I would use from it right now

For your baseline, I’d swap your dense renderer with this shape of pipeline:

img = rasterize_gaussians_torch(
    means2d=means2d,      # [G,2]
    conics=conics,        # [G,3]
    colors=rgb,           # [G,3]
    opacities=opacity,    # [G]
    depths=depth.detach(),# [G]
    image_size=(384, 384),
    background=bg,
)

And I would keep these settings first:

RasterizeConfig(
    alpha_threshold=1e-3,      # or 1/255
    max_alpha=0.99,
    use_fp16_patches=False,    # turn on later if stable on MPS
    exact_ellipse_mask=True,
)
What this does not solve yet

A few honest caveats:

It is still Python-loop heavy, so CUDA will beat it later.
It still sorts splats, so if sorting becomes the next bottleneck, sort-free weighted blending is the next branch to test, not this branch. Weighted Sum Rendering is the main recent “remove sorting entirely” direction.
It does not yet do tile lists, scanline buckets, or dynamic budgets.
It treats depth ordering as piecewise constant; no gradient through sort permutations.

Still, for the problem you described, this is the right first move: kill the giant [G,H,W] activations first. That directly targets the memory wall you called out in your note.

One extra useful detail: I included a tiny self-check in the file against a naive dense autograd renderer. On a toy case it matched with max image error about 4.2e-5 and gradient errors around 1e-6, so the custom backward is behaving like the dense reference.

The file again:

memory_efficient_splat_rasterizer.py

The next thing I’d do from here is wire this into your current projection code path and then profile which of these three becomes dominant on your Mac: projection, sorting, or local patch compositing.

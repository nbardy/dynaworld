 Yes. The key problem is mostly memory, and specifically activation memory in the differentiable rasterizer during training, not raw parameter count.

  And one nuance first: large renders do not necessarily mean “bigger gradients” in the sense of each parameter getting a larger update. What they do mean is:

  - many more per-pixel loss terms
  - many more intermediate tensors
  - many more backward paths through the renderer

  So the real pain is graph size and activation storage, not parameter storage.

  ## Short answers

  ### Would dropping to train_frame_count=8 and fewer gaussians help?

  Yes, a lot.

  For the dense render path, the painful term is roughly:

  cost ~ decoded_frames * gaussians * pixels

  So if you go from:

  - 16 -> 8 frames: about 2x cheaper
  - 512 -> 256 gaussians: about 2x cheaper

  Combined, that is roughly a 4x cut in the worst renderer-side memory/compute term.

  It also helps the encoder:

  - with train_frame_count=16, tubelet (4,16,16) gives 4 temporal bins
  - with train_frame_count=8, it gives 2 temporal bins

  So the encoder token count also drops by about 2x.

  ### What is temporal render microbatching?

  It means:

  1. encode the whole clip once
  2. decode all times, or at least the needed times
  3. render/backprop only a small chunk of decoded frames at a time

  Example with train_frame_count=16 and render_microbatch_size=4:

  - encode 16-frame clip once
  - render frames 0..3, backward their averaged recon loss
  - render frames 4..7, backward
  - render frames 8..11, backward
  - render frames 12..15, backward
  - one optimizer step at the end

  This sits between:

  - batched: fastest if it fits, worst memory
  - framewise: safest memory, more backward overhead

  Microbatching is usually the right middle ground.

  ### Does Faster-GS drastically reduce memory?

  It improves resource efficiency and reports up to 5x faster training in an optimized 3DGS setting, yes. But the important distinction is:

  - Faster-GS is built around an optimized PyTorch + CUDA backend and 3DGS optimization improvements
  - our current bottleneck is a pure PyTorch renderer on MPS

  So some ideas transfer, but it is not a drop-in fix for this repo’s current dense renderer. The biggest gains there come from:

  - fused CUDA kernels
  - better culling/truncation
  - renderer-side sparsity
  - not materializing huge dense G x H x W tensors

  Source: Faster-GS project page (https://fhahlbohm.github.io/faster-gaussian-splatting/)

  ## The architecture we are actually using

  ### 1. Input and clip sampling

  The trainer samples a contiguous clip of train_frame_count frames each step in train_scripts/train_video_token_implicit_dynamic.py:503.

  Current full default:

  - source video: 384x384, 35 frames
  - train_frame_count=16

  So each training step sees one 16-frame clip.

  Important:

  - batch size is effectively 1 clip
  - temporal extent inside that clip is the main axis of training

  ### 2. Video tubelet encoder

  The model starts with train_scripts/gs_models/dynamic_video_token_gs_implicit_camera.py:143, which is a Conv3d with kernel/stride equal to the tubelet size.

  Default full config:

  - tubelet_size = (4, 16, 16)

  So a 16 x 384 x 384 clip becomes:

  - temporal bins: 16 / 4 = 4
  - spatial bins: 384 / 16 = 24 each side

  Stage-1 token grid:

  - 4 x 24 x 24 = 2304 tokens

  This is already very aggressive compression. You are not feeding 16 full frames into a big ViT at full resolution. The encoder crushes them early.

  Then the encoder does:

  - stage-1 self-attention on the 2304 tokens
  - spatial downsample to 4 x 12 x 12 = 576 bottleneck tokens
  - 4 bottleneck transformer blocks
  - upsample and skip-merge back to stage-1 resolution

  That is all in train_scripts/gs_models/dynamic_video_token_gs_implicit_camera.py:191.

  ### 3. Learned query tokens

  After the video encoder, the model uses a small learned query bank:

  - 1 global camera token
  - 1 path camera token
  - num_tokens GS tokens

  Default full config:

  - num_tokens = 8

  So total learned query tokens = 10.

  These queries interact with the encoded video tokens through train_scripts/gs_models/dynamic_video_token_gs_implicit_camera.py:110.

  That part is relatively cheap:

  - queries are only length 10
  - memory tokens are 2304

  So cross-attention is not the big bottleneck.

  ### 4. Per-time decode

  The model then decodes one output per requested time in train_scripts/gs_models/dynamic_video_token_gs_implicit_camera.py:420.

  Mechanically:

  - time_proj(t) turns scalar time into a feature vector
  - this time offset is added to:
      - the path camera token
      - the GS tokens
  - the global camera token stays shared across the clip

  Then:

  - train_scripts/gs_models/dynamic_video_token_gs_implicit_camera.py:290 predicts base FoV and radius
  - train_scripts/gs_models/dynamic_video_token_gs_implicit_camera.py:330 predicts a per-time SE(3) delta
  - train_scripts/gs_models/dynamic_video_token_gs_implicit_camera.py:265 predicts gaussian params from the GS tokens

  Default full config:

  - 8 GS tokens
  - 64 gaussians per token
  - total = 512 gaussians per decoded frame

  Predicted per gaussian:

  - xyz
  - scale
  - quaternion
  - opacity
  - RGB

  No SH here, just RGB.

  ### 5. Renderer

  This is where the trouble is.

  The 3D gaussians are projected to camera space in train_scripts/renderers/common.py:30:

  - build rotation from quaternion
  - build 3D covariance
  - world-to-camera transform
  - perspective Jacobian
  - project covariance to 2D
  - depth-sort gaussians

  Then the dense renderer in train_scripts/renderers/dense.py:6 builds these full tensors:

  - dx: shape [G, H, W, 2]
  - power: shape [G, H, W]
  - alpha: shape [G, H, W]
  - weights: shape [G, H, W]

  For your full run:

  - G = 512
  - H = W = 384
  - H*W = 147,456
  - G*H*W = 75,497,472

  Just rough fp32 sizes for one rendered frame:

  - dx: about 151M floats, around 604 MB
  - power: about 302 MB
  - alpha: about 302 MB
  - weights: about 302 MB

  And that is only the obvious core tensors. It does not include all saved intermediates and gradient buffers.

  So one dense frame is already very expensive. Holding many of those render graphs alive at once is what kills MPS.

  ## Why the renderer, not the encoder, is the main bottleneck

  The encoder is not tiny, but it is not the first thing exploding.

  ### Encoder cost

  Stage-1 attention is over 2304 tokens. That is real work.
  If naïvely materialized, attention is quadratic in token count.

  But:

  - there is only 1 stage-1 block
  - bottleneck runs on 576 tokens
  - cross-attention only has 10 queries
  - the whole encoder works on compressed tubelets, not pixels

  ### Renderer cost

  The renderer touches:

  - every gaussian
  - every pixel
  - for every decoded frame

  That scales much more brutally.

  A useful approximation:

  render cost ~ T_decode * G * H * W

  At full config:

  - T_decode = 16
  - G = 512
  - H*W = 147,456

  That dwarfs the small token decoder.

  ## Why the small run used to work

  Because all three hard multipliers were smaller.

  ### Smoke-ish regime

  - 32x32
  - fewer gaussians
  - fewer decoded frames

  ### Full regime

  - 384x384
  - 512 gaussians
  - 16 decoded frames

  Just the pixel jump alone is:

  384^2 / 32^2 = 144x

  If gaussians also go from 64 -> 512, that is another:

  8x

  If decoded frames go from 4 -> 16, that is another:

  4x

  Total renderer-side pressure increase is roughly:

  144 * 8 * 4 = 4608x

  That is why “small network” does not save the full run. The bottleneck moved out of the encoder and into the differentiable rasterizer.

  ## What the current backward modes do

  This lives in the trainer at:

  - train_scripts/train_video_token_implicit_dynamic.py:549
  - train_scripts/train_video_token_implicit_dynamic.py:585

  ### Batched

  - render all decoded frames
  - build all recon losses
  - average them
  - one backward call

  Best if memory fits.
  Worst if memory does not.

  ### Framewise

  - render one frame
  - backward its loss / T
  - render next frame
  - backward its loss / T
  - ...
  - one optimizer step after all frames

  This is still the exact full-clip gradient for the current architecture.

  Why it helps:

  - the shared clip encoder graph stays alive
  - but each frame’s renderer subgraph can die earlier

  So peak memory becomes closer to:

  - shared upstream graph
  - plus one render graph

  instead of:

  - shared upstream graph
  - plus all 16 render graphs

  ### Temporal render microbatching

  This would be the middle option:

  - render/backward 2 or 4 frames at a time
  - not 1
  - not all 16

  That should improve throughput over framewise without going back to full batched OOM.

  ## Would Faster-GS save us?

  It could help directionally, but not in the way you are hoping if we keep this renderer backend.

  The strong parts of Faster-GS are:

  - better optimization engineering
  - truncation ideas
  - gradient approximation
  - optimized backend design
  - overall resource efficiency

  But the practical catch is:

  - their gains are in an optimized 3DGS stack
  - ours is still using a very direct PyTorch differentiable rasterizer on MPS

  So:

  - Yes, Faster-GS supports the idea that smarter truncation/culling/approximation can matter.
  - No, I would not expect a paper-level “drastic memory drop” just by porting one trick into this exact renderer.

  The really large win usually comes from fused kernels / CUDA rasterization, not just changing the MLP or token encoder.

  ## Does RGB-only help?

  Yes, but only a bit.

  Because we only predict RGB instead of SH coefficients:

  - gaussian parameter payload is smaller
  - color evaluation is cheaper

  But the dominant memory terms are still:

  - geometric broadcast tensors
  - alpha / transmittance tensors
  - per-pixel / per-gaussian interactions

  So RGB-only helps, but it does not solve the current wall.

  ## What I would do next

  Ranked by value:

  1. Default to tiled for full-res training.
  2. Add temporal render microbatching (2 or 4 frames at a time).
  3. For the first practical baseline, drop to:
      - train_frame_count=8
      - gaussians_per_token=32
  4. Use a curriculum:
      - lower render size first
      - then larger render size later
  5. Keep the old image-encoder implicit-camera baseline around as control, which is now aliased and preserved.

  If you want one pragmatic low-risk config to try next on MPS, I’d use:

  TRAIN_FRAME_COUNT=8 \
  GAUSSIANS_PER_TOKEN=32 \
  RENDERER=tiled \
  TRAIN_BACKWARD_STRATEGY=batched \
  bash ./train_full_dynamic_with_video_token_implicit_camera_all_frames.sh

  And if that still strains memory, switch:

  TRAIN_BACKWARD_STRATEGY=framewise

  ## Sources

  - ChopGrad paper: https://arxiv.org/html/2603.17812v1
  - Faster-GS project page: https://fhahlbohm.github.io/faster-gaussian-splatting/
  - Speedy-Splat paper page: https://speedysplat.github.io/
  - 3DGS original paper: https://arxiv.org/abs/2308.04079

  If you want, I can implement render_microbatch_size next.

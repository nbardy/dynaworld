# Novel View Training Strategies

Verbatim dump from session 2026-04-21. Left unedited on purpose.

---

I want to add some loose notes and plans to to todo in top level readme. For me the key thing right now is that I want to figure out how to train a base model that emits splats that can handle novel views.

1) If we just train video encoder to generate image tokens and a camera token. I'm worried the image tokens will be dependant on that camera token somewhat implicitly. and so if we swap the camera token at render time we will unfortunately have splats that dont generalize
2) I"m worried the splat we learn to reproduce will not look good from novel synthesis angles, because they will learn to overfit to splats that just look good from given camera angle and learn to cheat. I think training on the videos will help with this, but in the long run training we need to make sure its encouraged to train on splats that are robust to angles NOT in the encoded video.
The most way to do this is multi view data, BUT I'd like to not require that as there is not robust multi view data, and I'd like to find a way to elicit this behavior

The simplest solution I can currently think of that doesn't rquire special data is to encode two parts of a video at train time. Then use the video tokens fo the frist video, Then swap the camera, and the time conditioning of the 2nd video, so we make it predict the second video GT frames based on the first video

This is good because it has to generate from camera positions it hasn't seen before, but its BAD because it will force it to learn to hallunicate times it hasn't seen before .
Video generation is a MUCH harder task that video synthesis, but maybe thats fine because I'm bootstrapping off video diffusion
secondly its bad because those will still be continuations of early cameras positions, AND NOT fully novel new camera positions at the same point in time.

Another thoughts I had is to turn each clip into "Multi view" by doing a crop and perspective warp. (Classic videography trick where you take high resolution footage and crop a corner and rescale to make the perspective look like its in the center of each frame, This feels valuable TBH but
downsides again:
1) its still from the same angle
2) too much perpective warp is a bit cheap and doesn't exactlt align to GT camera data

We could also not perspective warp it, and we could just like shif tthe rays so we kinda define the crop as like a camera extrinsic, this is more honest, but then it's sort of learning crop shift only, and might not generalize as wel to non crop shift where the shift is the center of the camera.

I think this is the inklings of a good solution. Both are valuable, and maybe if I do both in pretrain that is enough to get a good prior.

The third solution I'm thinking of is that I could do a second stage post training. I am thinking  can render novel passes and train a GAN on them, so we do a GAN for novel and non-novel views. And it has to learn to make them both the same.

TBH this feels like the best solution. MAybe some sort of reward style training here is as wel
https://arxiv.org/abs/2603.17812

Chopgrad Recently did really high quality correction.

My thought is I need a bit of both:
So
1) Make it have some prior off camera capability in pretrain
2) refine that in post training

I also think a BERT like masking could work here possibly, this might be better than only the chop in half and predict seoncd or third half, like maybe some sort of ranomd maksing dropout scheme is robust in pretraining. But worried that will force the camera data to hide itself in the image tokens,

I am worried too much that the wrong task will force camera implicitly into image tokens if we try to hide camera position too much and in non principled ways,

Also I think if we relally look at this there is some sort of architecture / objective idea that is more trenscednant where we properly stop and reflect on the AR vs Diffusion and rolling vs forcing diffusion and the rolling window stuff. Like we need to more nativelt extend to parital/long context and rolling context. MAybe even AR on tokens per frame. In a way that ends up robust to noise at inference time via training on noisier data.

And get away from the like single encoder => Decode paradgim here to somethig more elegant

---

## 2026-05-02 Update: Paired-Camera Token Swap Contract

The multicam train2/heldout1 setup sharpened the original "swap the camera"
idea. The stronger paired-camera contract is:

```text
encode(video_a) -> world_a, camera_a
encode(video_b) -> world_b, camera_b

render(world_a, camera_a) -> video_a
render(world_b, camera_b) -> video_b

render(world_a, camera_b) -> video_b
render(world_b, camera_a) -> video_a
```

This makes sense if `video_a` and `video_b` are synchronized views of the same
scene. The cross terms are the key: the world/video tokens from one camera must
render correctly from the other camera token. That directly trains the behavior
we need at inference time:

```text
encode(source_video) -> world_source, camera_source
query_camera_token = requested new camera
render(world_source, query_camera_token) -> novel view
```

Important caveat: the target camera token cannot be an arbitrary high-capacity
latent from the target video. If it is, it can leak target appearance and make
the swap fake. The camera token needs to decode to an explicit pose/intrinsics
object, or be constrained as calibrated pose plus a small learned residual.

For train2/heldout1:

```text
train:
  W_A + C_A -> GT_A
  W_A + C_B -> GT_B
  W_B + C_B -> GT_B
  W_B + C_A -> GT_A

heldout eval:
  W_A + C_H -> GT_H
  W_B + C_H -> GT_H
```

`C_H` should come from the heldout calibrated pose, not from heldout RGB, if the
goal is a true novel-camera test.

See `research_notes/training_ideas_for_novel_synthesis.md` for the cleaner
contract and falsification tests.

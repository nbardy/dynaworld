# Ideation On What Is A World Model

Loose research note. Keep the raw thought dump mostly verbatim, then separate
the attempt at formalization below it.

## Raw Dump

Remove this phase framing:

> Phase III - Generative world models. Condition on text or image instead of
> video. Same token / cross-attn decoder, different encoder; the splat side
> stays the same.

There is not generative phase planned as we will leave this phase to world
models for generating the initial world.

Our model is generative in the fact that it can extend the world you know
forward and backward in time as well past your initial input and given it's
ideas of what it's seen before it should know this.

And TBH i'm thiking we dont focus on extending pastin time atm as that sort of
requires generating a model that can extend token priors forward and back and
maybe removing that from scope is atually a realy strong way to keep our model
focused, so it only generates novel angles, That IS halluncating some content,
as we don't always know what is behind things, BUT it's not fully generating in
time, and it's like look behidn object or extend backgorun basically.

I mean as I think about that I relaize that does require real generative
capacity and maybe the pre-training kinda generative rolling window is the
right task to get that capacityin this model, like have it train on the GT of
images it didn't encode it is forced to generate data that is not in the encode
path...

And thts how.

So yea trying to zoom in here on what is the best way to train this model what
objectives and architecure and what phases, and I"M realizing it does need
generative capacity, So the 3D inductive bias in the splat renderer really
helps, and the T based MLP Helps to make it generalize to 4D spacetime
modeling, but if I for example "look behind" an object that it hasn't seen
before in the scene it needs to know whats back there EVEN if the input hasn't
seen it.

## Followup

### Attempt at LLM formalization

Dynaworld should not define a separate "generative world model" phase whose
goal is text-to-world or image-to-world generation. That direction belongs to
models that generate the initial world. Dynaworld's sharper scope is different:
given a real input video, reconstruct a dynamic 3D/4D world representation that
can render views the input camera did not directly observe.

That does not mean the model is non-generative. Novel-view synthesis still
requires generation in the local, grounded sense. If the user asks to look
behind an object, extend a background, or reveal a surface that was occluded in
the input video, the model must infer plausible missing content. The difference
is that this generation is anchored to an observed scene and expressed through a
renderable splat world, rather than through unconstrained text/image sampling.

This suggests a narrower product and research definition:

Dynaworld is a grounded generative world model for observed videos. It does not
generate the initial world from scratch. It generates the unobserved parts of a
world implied by the input video, especially the parts needed for novel camera
motion through space.

The immediate scope should stay focused on novel angles, not long-range temporal
extension. Predicting far forward or backward in time likely needs a stronger
token prior or causal dynamics model that can roll the world state. That may be
valuable later, but it risks distracting from the core MVP: video in, dynamic
splats out, editable camera path, render out.

Training still needs generative pressure. If every reconstruction target is
already directly encoded in the input window, the model can overfit to visible
pixels without learning a useful prior over hidden geometry and appearance. A
better pretraining objective may encode only part of a clip or view set and
supervise renders at frames or views outside the encoded path. This forces the
model to use:

- the 3D inductive bias of the splat renderer
- the time-conditioned MLP / token decoder as a 4D spacetime model
- learned priors about continuity, occlusion, and likely backside/background
  content

The key distinction:

- Out of scope for now: generating a whole world from text or a single prompt.
- Out of scope for now: rolling the scene far forward/backward in time.
- In scope: grounded generation of missing spatial content needed for novel
  camera angles in an observed dynamic scene.

This may be the clean training philosophy:

1. Encode a bounded video window.
2. Decode dynamic splat tokens as a continuous function of time.
3. Render supervised target views/frames, including some that were not directly
   available to the encoder.
4. Use reconstruction loss through the renderer so the learned prior is always
   tied to a physically renderable scene representation.

The model is therefore generative where it matters for Dynaworld: not by
inventing the starting world, but by completing the world behind, around, and
between the observations.

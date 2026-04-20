# Voice

This is a guardrail for README, research notes, and planning docs.

The goal is not to imitate a generic terse startup voice. The goal is to keep
Nicholas's actual n-grams alive.

## Slop Detected In The Last Pass

These patterns are the problem:

- turning the thesis into a neat market map
- adding a fake Phase III for text/image generative world models
- replacing raw phrases with polished equivalents
- writing sentences like "The splat is the 3D handle. The video is how you ship it."
- using external name drops as structure: "Sora, Veo, Wan, SVD"
- making the prose sound like a YC landing page
- over-explaining "exploratory" when the phrase already lands
- converting chat-compressed ideas into slogan chains
- making every bullet parallel
- smoothing away useful typos in loose notes

The biggest correction: there is no planned generative phase for generating the
initial world. Leave that to world models that generate the starting world.
DynaWorld still has generative capacity, but it is grounded in a video and used
for novel angles, hidden content, and dynamic splat rendering.

## Keep These Moves

- Short paragraphs.
- Claims first.
- `Video <=> splats`, not a prettier arrow.
- `Video <=> Video`, because the model trains on video.
- "less about X and more about Y"
- "they are complementary, not opposites"
- "world models are video models"
- "The useful signal is in the preimage."
- "Foundations are sacred."
- "Memory should be spent on dynamic scene state, not luxury parameters."
- "Static and dynamic are the same problem."

## Do Not Normalize These Away

These are high-conviction phrases from chat. Reuse them when they fit.

- "Hollywood's Open Source World Model"
- "my goal is dynamic video => compressed splats"
- "world modeling is a broader term TBH"
- "more for doing camera change and then as followup special effects"
- "they are complimentary not opposites"
- "some are generaitve and some are 'exploratory'"
- "dynaworld focused on the exploratory portion"
- "video diffusion the generateive portion"
- "you can use the dynaworld to create an exploratory version"
- "Dynaworld is less about making new world"
- "and more about modality shifts"
- "It lets you go from Video<=>splats both directions"
- "so you can use splats when they're best and video when its best"
- "world model needs to be Video<=>Video"
- "so it can train slf structured on video"
- "unsupersvised or self supervised"
- "There is not generative phase planned"
- "leave this phase to world models"
- "for generating the initial world"
- "Our model is generative in the fact"
- "we dont focus on extending pastin time atm"
- "keep our model focused"
- "it only generates novel angles"
- "That IS halluncating some content"
- "we don't always know what is behind things"
- "not fully generating in time"
- "look behidn object"
- "extend backgorun basically"
- "pre-training kinda generative rolling window"
- "train on the GT of images it didn't encode"
- "forced to generate data that is not in the encode path"
- "3D inductive bias in the splat renderer"
- "T based MLP Helps"
- "generalize to 4D spacetime modeling"
- "it needs to know whats back there EVEN if the input hasn't seen it"

## Cleaned Variants For Public Copy

Use these when README needs spelling cleaned but the phrase should stay his.

- "DynaWorld is less about making new worlds and more about modality shifts."
- "It lets you go from `Video <=> splats` both directions."
- "Use splats when they're best and video when its best."
- "They are complementary, not opposites."
- "DynaWorld is focused on the exploratory portion."
- "Video diffusion is the generative portion."
- "A world model needs to be `Video <=> Video`."
- "There is not a generative phase planned for generating the initial world."
- "For now, keep the model focused. It only generates novel angles."
- "Train on the GT of images it didn't encode."
- "It needs to know whats back there, even if the input hasn't seen it."

## Anti-Slop

Delete these on sight:

- "proves the whole pitch"
- "unlock"
- "seamless"
- "reimagines"
- "at the intersection of"
- "not just X, but Y"
- "a new paradigm"
- "the future of"
- "rich, immersive"
- "geometry-aware, temporally coherent, real-time"
- "the X is the product"
- "the Y is the architecture"
- "DynaWorld is a..."

The voice opens with what it does.

## Pre-Ship Check

1. Did the copy preserve at least one real n-gram?
2. Did it accidentally add a text-to-world Phase III?
3. Did it explain something that was stronger as a short claim?
4. Did it replace "Video <=> splats" or "Video <=> Video" with prettier language?
5. Did it sound like a landing page?

If yes, cut it back.

# Dyna World v1

**Core Hypothesis:** World models are essentially video models that have already learned the geometry of the world. We can extract 3D representations (Gaussian Splats) from a video world model using a very small amount of data, essentially acting as a linear probe or lightweight adapter.

## Key Beliefs & Postulates

1. **World Models = Video Models:** Video models already learn the geometry of the world.
2. **Efficient 3D Extraction (Linear Probe):** Mapping video $\rightarrow$ Gaussian Splats requires only a minimal amount of data. This is analogous to how depth can be extracted from Stable Diffusion using simple probes.
3. **Static == Dynamic:** This approach applies just as easily to dynamic video as it does to static video.
4. **Self-Supervised Video Models as Foundation:**
   - Any 3D labels are inaccurate at scale.
   - The ground truth for 3D is light from video.
   - The best way to train a world model is just to train a self-supervised video model.
5. **Adapters over New Foundations:** If you need other outputs from a world model, you should finetune an adapter to extract it rather than training a completely new foundation model.

## Rationale & Precedents
- Recent advancements in tokenizer papers show strong performance on as few as 50k samples.
- Techniques like Flux Edit LoRAs work on as few as 5-10 samples.
- Therefore, we should be able to train an efficient adapter on a small dataset of video-to-splat pairs to get 3D representations out of an existing video world model.

## Next Steps / Ablations
*(To be determined based on the initial focus)*
- [ ] Select a baseline video model / foundation model to probe.
- [ ] Define the architecture for the lightweight adapter / probe.
- [ ] Construct a micro-dataset (e.g., 10-50 high-quality video-to-splat pairs) for the initial training run.
- [ ] Establish an evaluation metric for the generated Gaussians.

---

## Navigation — where to look for what

Layered, strategic → tactical:

- **`meta_philosophy/`** — strategic core. Read its `README.md` first. How to design architectures, failure modes F1–F7, mistakes log (regression-prevention), prompt-guidance notes, and the driver-prompt template for external LLMs.
- **`meta_philosophy/world_splat_tokens_vs_observed_modality_tokens.md`** — token philosophy. Defines observation/world/splat/memory/camera tokens, explains why splat tokenizers differ from text/image/video tokenizers, and records when two-stage tokenizer training is legitimate versus degeneracy-freezing.
- **`meta_philosophy/dynaworld_architecture_solution_prompt.md`** — paste-ready external-model prompt for generating 3-4 DynaWorld architecture solutions with compact rationale fields, anti-degeneracy arguments, falsification tests, and synthesis.
- **`framing_the_problem/`** — three framings of the novel-view bottleneck. Framing 1 is information-theoretic (for deriving losses); framing 2 is the self-sufficiency / generative-reconstruction contract (for auditing architectures); **framing 3 is the patched bitter-lesson predictive-quotient baseline and the current default** — start there for proposing anything new. Has its own `README.md` with when-to-use guidance.
- **`training_contract_v1.md`** — operational contract for patched framing 3: `D_var` sampler, model signatures, baseline losses, diagnostics, escape hatches, support assumptions, deployment/export contract, and failure tripwires.
- **`three_architectures_for_novel_view_synthesis.md`** — concrete architecture candidates (A/B/C), diagrams, head-to-head debate, pioneer pick. Cross-references the framings.
- **`potential_directions_index.md`** — routing map for all research threads. Status labels (Now / Probe / Background / Speculative) per direction. Start here when scoping a new experiment.
- **`../agent_notes/key_learnings.md`** — dense bank of surprising technical lessons. Tactical, not strategic.
- **`../agent_notes/loose_notes/`** — raw session chronology. Go here when you need the why behind a decision, not just the outcome.

When to use which:

- *New agent, cold start:* read `meta_philosophy/README.md`, then `potential_directions_index.md`, then this file.
- *Proposing an architecture:* run through the checklist in `meta_philosophy/how_to_think_about_architecture.md` before writing anything.
- *Driving an external LLM (ChatGPT Pro, Gemini, etc.):* paste `meta_philosophy/chatgpt_pro_prompt_for_expert_divergent_web_of_thought_model_architecture_development.md` as the system brief and attach the problem doc.
- *Proposing anything new:* start with `framing_the_problem/framing_3.md`. It was patched, not replaced; do not mint `framing_4.md` for the predictive-quotient correction.
- *Implementing the baseline:* use `training_contract_v1.md` for sampler/signature/loss/export details.
- *Auditing an existing architecture for frame-local state leaks:* use framing 2's constraints C1, C3, C5, C6, C7 as the audit checklist.
- *Deriving a new loss:* use framing 1's information-theoretic view.

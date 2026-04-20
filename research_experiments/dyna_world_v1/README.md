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

# Key Learnings

Dense memory bank for surprising lessons learned by trying things. Keep this
file under 200 lines by recompressing it. Raw chronology belongs in
`agent_notes/loose_notes/`.

- `uv runpython` is just a typo, but `uv run <script.py>` can fail differently from `uv run python <script.py>` when the active directory/project is wrong. The durable fix was making `dynaworld` its own `uv` project and running from that project root.
- `uv` already has a shared global package cache by default. A cache env var is only needed to override the default location, not to make sharing happen at all.
- MPS made the tiny TokenGS experiments usable but not automatically fast. Kernel launch/resource contention from another tab can dominate perceived trainer speed.
- Lowering token count is a speed knob, but it also changes representational capacity. Decoupling encoder input size from render/loss size was the better smoke-test knob because it preserved the model input contract while cutting raster cost.
- The naive tiled renderer with nested Python loops was slower than dense PyTorch for the small Mac test case. Tile-based rasterization only pays off when culling/sorting is implemented cheaply, not when Python loops become the scheduler.
- A one-big-tile path being faster than expected showed that "tiled vs dense" was not the real binary. The overhead and code path mattered as much as the raster math.
- For the current renderer, DUSt3R lens metadata beyond pinhole intrinsics/extrinsics is noise. F-stop, lens type, shutter, and sensor metadata do not help unless we add distortion or photometric modeling.
- DUSt3R appeared to run "two passes," but the second phase was global alignment, not duplicate inference. Future logs should label pair inference vs alignment more clearly.
- Stale extracted frame folders are dangerous UX even when the code loads explicit JSON records. Users look at folders, so extraction should clear or isolate outputs to avoid misleading evidence.
- Rendering without camera extrinsics lets the model cheat with frame-conditioned gaussians. For dynamic novel view synthesis, using `camera_to_world` in the renderer is not optional.
- DUSt3R per-frame focal estimates can jitter on a phone clip. Median focal across the sequence is often a better default than trusting each frame.
- A single shared `BaseTrainer` would hide real differences between known-camera, image-implicit, and video-token implicit training. Shared payload contracts are cleaner than shared trainer inheritance.
- Positional tuple outputs were fine for toy iteration but became risky as soon as camera state, videos, and multiple baselines appeared. Named runtime payloads are the right boundary.
- `SequenceData.__getitem__` was useful as a temporary migration crutch, but it hid weak contracts. After trainer call sites moved to attributes and `ClipBatch`, deleting the bridge made violations easy to grep.
- System `python3` can pass compile checks but fail import smokes because dependencies live in the `uv` environment. Use `uv run python` for meaningful Dynaworld checks.
- The smoke wrapper lacked execute permission even though the full wrappers were executable. Always check script mode before assuming a shell script can be run with `./...`.
- JSONC configs are better than environment-variable fanout for trainer experiments. Shell wrappers should choose config files rather than re-declaring every knob.
- Shared JSONC helpers belong in one tiny utility. Keeping `strip_jsonc_comments(...)`, config loading, and W&B-safe serialization in `src/train/config_utils.py` avoids both parser copy-paste and a mirrored Python default map.
- `render_size` should be treated as a viewport, not a crop. Scale `fx/fy/cx/cy` from encoder input pixels to render pixels at the render boundary, then resize GT for loss; pose and FoV stay unchanged.
- Keep full experiment defaults in JSONC, not in a mirrored Python `DEFAULT_CONFIG`. Python should normalize runtime values, validate required sections, and keep constructor-name mapping in one boundary factory.

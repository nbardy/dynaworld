# FunSearch Paper Note

User goal remains active: read 20 papers on AlphaEvolve-style agentic code
evolution and write detailed notes for each with DynaWorld synthesis.

What changed in this chunk:

- Read the FunSearch Nature paper from the author PDF, Nature page, and the
  official `google-deepmind/funsearch` repository.
- Added `alpha_evolve/papers/notes/002_funsearch.md`.
- Marked paper 002 as done in `alpha_evolve/papers/paper_queue.md`.
- Updated `alpha_evolve/papers/README.md` and `synthesis.md`.

Main lesson:

FunSearch is a narrower implementation template than AlphaEvolve and is
probably the right first local proof shape. Start with a fixed skeleton plus one
evolved callable/helper, not broad repo patches. Store score signatures, not
only scalar scores, and prompt Codex with two measured prior candidates from the
same island.

Specific DynaWorld implications:

- `mixed_same_view_novel_scheduler` can be framed as an evolved scheduler policy
  inside a fixed loader/trainer smoke.
- `gaussian_512_promotion_guard` can be framed as an evolved promotion/guard
  policy inside a fixed multires smoke.
- `star_uvt_feature_rgb_handoff` is probably too broad for a first FunSearch
  shape unless reduced to a small helper around gradient-path selection or
  partitioning.

Next paper:

CodeEvolve should be read next because it bridges from FunSearch's one-function
setting to AlphaEvolve-like open-source patch evolution.

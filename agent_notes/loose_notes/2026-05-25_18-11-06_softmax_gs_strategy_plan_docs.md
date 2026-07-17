# Softmax-GS Strategy Plan Docs

Date:
    2026-05-25

Context:
    After ingesting Softmax-GS and discussing where its softmax compositing
    sits relative to rasterization, STAR UVT, and WorldFoam, the user asked for
    the concrete work left to do and for short-term/long-term plan docs.

What changed:
    Added two curated strategy docs under
    `research_notes/gaussian_splatting_papers/`:

    - `2026-05-25_short_term_softmax_gs_plan.md`
    - `2026-05-25_long_term_splats_vs_worldfoam_plan.md`

    Updated `paper_index.md` so future agents can find the paper artifacts,
    the DynaWorld integration note, and both plans from one entry.

Main decision captured:
    Softmax-GS is a bounded short-term renderer probe for dynamic GS first.
    It should not be treated as a STAR UVT support fix or as a WorldFoam
    replacement. STAR gets only a CPU/reference diagnostic until dynamic-GS
    evidence is positive or STAR support coverage improves. WorldFoam remains
    the serious long-term challenger, but it should not become mainline until
    it wins a same-split, heldout-aware tournament against better splats/STAR.

Immediate queue captured:
    Build the one-pixel Torch reference, audit true projected depth through
    dynamic-GS/fast-mac, prove F3 no-op parity, run a matched dynamic-GS smoke,
    then decide whether STAR merits a Softmax-GS CPU diagnostic.

Verification:
    Read back the new docs and index snippets. No runtime tests were run because
    this was documentation/planning only.

# Softmax-GS Short/Long Plan Docs

Date:
    2026-05-25 20:36:31 Asia/Ho_Chi_Minh

Context:
    After reading Softmax-GS and wiring the first dynamic-GS shader path, we
    needed durable short-term and long-term plan docs answering whether this
    should feed dynamic GS, STAR UVT, WorldFoam, or a broader representation
    switch.

Plan docs:
    - `research_notes/gaussian_splatting_papers/2026-05-25_short_term_softmax_gs_plan.md`
    - `research_notes/gaussian_splatting_papers/2026-05-25_long_term_splats_vs_worldfoam_plan.md`

Current model:
    Softmax-GS is a renderer/compositing probe, not a world-representation
    proof. The dynamic-GS fast-mac fork is the right first integration target
    because it matches the paper's intended failure mode: overlapping fuzzy
    splats, boundary ownership, and order-sensitive alpha-over composition.
    STAR UVT should not get a Metal port yet because its current blocker is
    support/coverage; WorldFoam should not inherit Softmax-GS work because foam
    changes the geometry primitive rather than only the final overlap law.

Evidence reflected in the docs:
    - The paper/PDF/source/converted Markdown are indexed under
      `research_notes/gaussian_splatting_papers/`.
    - `v5_softmax_gs` has no-op MPS parity and fixes the synthetic same-depth
      swapped-order color artifact.
    - Native fast/overflow backward exists through the scalar recompute bridge.
    - Bounded top-K tape exists in the reference and Metal ABI.
    - Backward now consumes the tape for color gradients when
      `softmax_gs_tape_k > 0`.
    - Geometry/opacity/depth still use O(K^2) recompute and are the next real
      blocker.
    - The cleaner seeded 50-step tiny source-view row is neutral/slightly
      negative (`no-op 0.1467` vs `enabled 0.1512`), so there is no basis to
      port Softmax-GS into STAR or WorldFoam yet.

Decision implications:
    Short term:
        Finish selected scalar tape rows for geometry/opacity/depth, then run a
        matched dynamic-GS quality row. Treat source-view smokes as mechanics
        only.

    Medium term:
        Keep STAR UVT support/projective-interval work as the main dynamic
        splat-time representation lane. Use Softmax-GS inside STAR only after
        support is adequate and overlap/order is visibly the bottleneck.

    Long term:
        Keep WorldFoam as the serious challenger, but move to it only after an
        honest representation tournament on the same split, resolution, frame
        count, wall-clock budget, source/heldout metrics, and export contract.

Follow-up consistency:
    Updated `research_notes/README.md` so the Gaussian-splatting paper index is
    discoverable as the route to the Softmax-GS short/long plans. Patched the
    integration and long-term notes so they no longer say the bounded tape is
    unused by backward; the precise status is color-gradient tape consumption
    done, scalar VJP pending.

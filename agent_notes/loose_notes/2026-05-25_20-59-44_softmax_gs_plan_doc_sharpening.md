# Softmax-GS Plan Doc Sharpening

Date:
    2026-05-25 20:59 +07

Context:
    User asked what work remains and requested long-term and short-term plan
    docs for the Softmax-GS paper relative to dynamic GS, STAR UVT, and
    WorldFoam.

What changed:
    Added explicit "do this next" sections to:

    - `research_notes/gaussian_splatting_papers/2026-05-25_short_term_softmax_gs_plan.md`
    - `research_notes/gaussian_splatting_papers/2026-05-25_long_term_splats_vs_worldfoam_plan.md`

Current answer:
    The next real task is a matched dynamic-GS quality row using the selected
    scalar tape path with `softmax_gs_tape_k=16` and `GSP_TAPE_CAP=16`. K=8 is
    too lossy; K=32 did not improve the tiny endpoint. Do not port Softmax-GS
    into STAR UVT or WorldFoam until dynamic GS shows a measured heldout,
    stability, or rate/quality gain.

Decision posture:
    Short term: keep Softmax-GS as a bounded RGB/F3 dynamic-GS renderer probe.
    Long term: keep STAR UVT as the main dynamic world-token bridge and keep
    WorldFoam as the serious challenger until it wins a matched representation
    tournament.

# Softmax-GS / STAR / WorldFoam Plan Docs

Date:
    2026-05-26

## Context

The user asked what work remains now and requested long-term and short-term
plan docs after the Softmax-GS paper read, dynamic-GS probe, STAR support
birth/split ladder, target-grid init, support-target alpha, and 2x2
support-target-area experiments.

## Evidence State

- Softmax-GS is implemented far enough to be judged as an opt-in dynamic-GS
  renderer probe. K=16 is usable, K=8 is too lossy, and K=32 does not improve
  the tiny source-view endpoint. Heldout evidence is mixed rather than a clean
  promotion.
- STAR support birth/split has a cap-safe seed and several cap/targeting
  primitives, but dense support has plateaued. Target-grid feature init,
  pointwise support-target alpha, and 2x2 support-target-area patches all learn
  local objectives but do not materially close the normal/forced/oracle gap.
- The current STAR blocker is therefore not another target scorer or tiny local
  patch loss. The next useful question is where the compositing/prefix behavior
  drops target support.
- WorldFoam remains a serious challenger, not the default replacement. It needs
  a matched heldout-quality tournament row before becoming the mainline.

## Docs Updated

- `research_notes/gaussian_splatting_papers/2026-05-25_short_term_softmax_gs_plan.md`
  now starts with a 2026-05-26 execution snapshot: STAR support remains active,
  the next work is a selected-patch diagnostic followed by visibility-prefix /
  compositing-tape behavior if supported, and Softmax-GS/WorldFoam ports stay
  parked.
- `research_notes/gaussian_splatting_papers/2026-05-25_long_term_splats_vs_worldfoam_plan.md`
  now starts with a 2026-05-26 strategic snapshot: STAR stays mainline for the
  next gate, Softmax-GS stays a dynamic-GS renderer probe, WorldFoam stays the
  challenger, and the next decisive long-term artifact should be a matched
  representation tournament.
- `research_notes/gaussian_splatting_papers/README.md` now routes future agents
  to the two roadmap docs.

## Next Work

1. Run a selected support-target patch diagnostic on current checkpoints:
   normal alpha, forced alpha, target-background/oracle composition, and alpha
   coverage over the selected birth-target patches.
2. If that diagnostic shows prefix/order/local-composition failure, build the
   STAR visibility-prefix/compositing tape gate for selected pixels.
3. Only after those two evidence steps should we start shader-side
   visibility-prefix or composition-loss work.

## Follow-Up: STAR UVT Binfix And Train Gate

Context:
    The selected-patch diagnostic contradicted the initial plateau read. It
    showed selected-only support had zero rendered alpha even though the
    selected tubes analytically had mean/max alpha around `0.20/0.36` at the
    selected target points. That localized the immediate failure to sparse
    binning/support bounds rather than Softmax-GS, WorldFoam, or the local
    objective itself.

Current model:
    STAR UVT moving tubes can have centers shifted far outside a local
    2-frame chunk. The valid temporal footprint comes from the inverse
    precision, not from `1/q_tt` or a local-chunk fallback. The binner rejected
    valid determinants around `1e-9` using `eps=1e-8`, then fell back to a
    temporal half-extent of only `mi.frames`. For a chunk-shifted tube with
    `m.z ~= -13`, that fallback still produced an empty time interval.

Code changes:
    Patched
    `third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_kernels.metal`
    so `inverse_sym3_diag` accepts determinants above `max(eps^2, 1e-20)` and
    so fallback bounds cover the shifted local domain with `abs(m) + domain`.
    Added `tests/test_star_uvt_feature_binning.py`, a focused MPS regression
    where one moving tube should render inside a 2-frame chunk even though its
    shifted center lies outside the chunk.

Evidence:
    `PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 uv run --with pytest python -m pytest tests/test_star_uvt_feature_binning.py -q`
    passed.

    Repaired selected-patch diagnostic:
    `outputs/benchmarks/2026-05-26_star_uvt_support_target_patch_diagnostic_binfix.md`
    shows targetinit/targetalpha/targetarea2 selected-patch normal PSNR
    `4.606/4.686/4.684`, forced-alpha PSNR `14.529/14.694/14.677`, and
    selected-only alpha around `0.30`. Before the fix, selected-only alpha was
    `0.0`.

    First repaired train:
    `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit16_multicenter_k16_r40_o04_footprintresidualcapslack_targetinit_targetarea2_binfix_from1500_lr001_50step_media.jsonc`
    completed 50 steps with zero tile overflow, max tile count `110/128`,
    total loss `0.889263 -> 0.863064`, support-target-area loss
    `0.253626 -> 0.217254`, feature loss `0.612217 -> 0.610967`, and
    RGB-probe PSNR `24.253 -> 24.453`.

    Post-train selected-patch diagnostic:
    `outputs/benchmarks/2026-05-26_star_uvt_support_target_patch_diagnostic_targetarea2_binfix_train.md`
    reports normal/forced/oracle patch PSNR `6.644/19.452/26.994`, patch alpha
    mean `0.481`, and selected-only alpha `0.444`.

Decision implication:
    The next STAR question is no longer whether selected born tubes survive the
    renderer. They do. The next gate is dense/media transfer: if the local
    selected-patch gain does not propagate to dense RGB/support, then build the
    visibility-prefix/compositing tape. Softmax-GS and WorldFoam remain parked
    until the representation tournament has matched heldout evidence.

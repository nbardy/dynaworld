# Gauged UVT Visible Swap Bound

Date: 2026-05-24 05:14:31

## Context

The previous Gauged UVT gate added depth/denominator visibility sidecars and
crossing-depth detection. The next gate was to decide when an ambiguous
crossing can be safely ignored versus when it needs split/fallback.

This pass remains compiler-side. No Metal shader or renderer hot path was
edited.

## What Changed

Extended:

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/projective_trace.py
```

with:

```text
ProjectiveTraceAppearanceSidecar
ProjectiveTraceSwapCost
make_projective_trace_appearance_sidecar(alpha_max, color)
bound_projective_trace_visible_swap_cost(order, appearance_a, appearance_b)
```

The swap-cost helper applies the visible-order bound:

```text
|Delta I_ij| <= alpha_i alpha_j |c_i - c_j|
```

using per-window `alpha_max` and constant/interval color bounds. It returns:

```text
swap_bound
safely_commutable
needs_fallback
```

for each primitive pair in the matched sidecars.

Updated exports in:

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/__init__.py
```

Extended:

```text
tests/test_star_uvt_projective_visibility.py
```

with tests proving:

- low-opacity ambiguous crossings are safely commutable below threshold
- high-opacity/color-contrast ambiguous crossings need fallback
- color interval uncertainty contributes to the swap bound

## Tests

Focused projective suite:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_trace.py \
  tests/test_star_uvt_projective_orbit_windows.py \
  tests/test_star_uvt_projective_visibility.py -q
```

Result:

```text
22 passed in 1.17s
```

## Current Model

The local visibility compiler now has:

```text
depth sidecars -> stable order / crossing / ambiguous
appearance sidecars -> visible swap cost
```

So an accepted tile-time chart can classify ambiguous pairs as:

```text
harmless commutation
visible ambiguity needing split or fallback
```

This still does not build a tile-time index or fallback mask. It is the
per-pair decision primitive that a compiler-side binning prototype should use.

## Next Gate

Wire accepted rational/projective windows into a compiler-side binning
prototype:

```text
projective windows + support bounds + visibility/swap sidecars
    -> tile-time records
```

Keep it CPU/Torch first. Do not touch the renderer hot path until the compiler
prototype can emit stable tile-time records and fallback masks on synthetic
orbits.

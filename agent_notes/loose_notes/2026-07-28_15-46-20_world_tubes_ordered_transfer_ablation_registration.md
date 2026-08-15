# World Tubes Ordered-Transfer Ablation Registration

Date: 2026-07-28 KST

## Trigger

The user asked whether the moving-camera extension should literally use
holonomy and requested a forked World Tubes shader/math variant that can be
ablated in the paper.

## Terminology Decision

The strong but technically precise label is:

```text
World Tubes + Ordered Ray Transfer
```

An open camera ray carries ordered parallel transport, equivalently a product
integral or path-ordered exponential of the optical generator. Holonomy is the
transport around a closed loop. The repository already has a separate
cell-complex loop-holonomy diagnostic, so naming an open-ray backend
`ray_holonomy` would create a real collision. The construction is holonomy-
inspired, but should not be sold as literal holonomy unless a loop is closed.

## Backtrack: The Fork Already Exists

Inspection showed that the requested bounded fork had already landed in the
live dirty checkout:

```text
research_experiments/spd4_world_tubes/retained_fiber.py
research_experiments/spd4_world_tubes/retained_fiber_transfer.metal
research_experiments/spd4_world_tubes/retained_fiber_metal.py
research_experiments/spd4_world_tubes/hybrid_transfer.py
```

Creating another shader would duplicate the physical operator and muddy paper
identity. The existing implementation is the intended fork:

```text
certified tile:
  fast mean-ordered STAR Beer--Lambert

ambiguous tile:
  retain conditional Gaussian depth profiles
  integrate ordered emission--absorption
  use the matching native Metal VJP
```

The World Tubes depth fibers already handle moving camera charts and changing
conditional depths. The new branch fixes the narrower failure mode where
thick, differently colored fibers overlap so that one representative-depth
ordering is not sufficient. It is therefore a general robustness/physics
improvement and an ablation, not a repair required for the central compiled
world-tube or world-foam concept.

## Registered Ablation

Created `TODO/world_tubes_ordered_transfer_ablation.md` and linked it from the
paper experiment plan, TODO index, and experiment registry. It freezes four
separate identities:

```text
WT-OT0  legacy tube + peak splat + fast STAR
WT-OT1  full SPD(4) + Beer--Lambert + fast STAR
WT-OT2  full SPD(4) + Beer--Lambert + all-retained transfer
WT-OT3  full SPD(4) + Beer--Lambert + certified hybrid
```

The existing 16-atom `10/64` fallback smoke is the positive routing check. The
199-atom `64/64` fallback result is preserved as the negative selectivity
control. The next experiment is depth initialization and/or a bounded
color-commutator certificate, not publication-scale reruns.

## Falsifiers

Demote the hybrid if it cannot:

```text
match all-retained image/VJP results on fallback stress cases
reduce ordinary-scene fallback below the declared 20% target
beat all-retained end-to-end wall time
improve a colored-overlap/order-change case over mean-order STAR
meet a declared quadrature error tolerance
```

Failure of this extension does not falsify the central World Tubes interval
atlas or sublinear-across-frames result.

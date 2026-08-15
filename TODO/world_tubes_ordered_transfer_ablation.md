# World Tubes Ordered-Transfer Ablation

Status: **Bounded evidence verified; selective dense-scene gate open**

## Naming And Scope

Use **World Tubes + Ordered Ray Transfer** for the ablation label and
`hybrid_retained_fiber` for the executable backend identity.

The construction uses the same connection/parallel-transport mathematics that
motivates holonomy, but an ordinary camera ray is an open path. In the paper,
reserve **holonomy** for closed-loop transport or an explicitly closed ray
cycle. Do not rename the method or backend to `ray_holonomy`: this repository
already uses loop holonomy as a separate diagnostic.

This is an in-paper World Tubes extension, not a rename of the World Tubes
paper and not a replacement for the WorldFoam umbrella. It borrows
WorldFoam's ordered emission--absorption operator only at cells where a
mean-depth STAR order cannot be certified.

## Existing Fork

Do not create another renderer clone. The bounded shader/math fork is already:

```text
research_experiments/spd4_world_tubes/retained_fiber.py
research_experiments/spd4_world_tubes/retained_fiber_transfer.metal
research_experiments/spd4_world_tubes/retained_fiber_metal.py
research_experiments/spd4_world_tubes/hybrid_transfer.py
```

The fast branch remains the production STAR Beer--Lambert renderer. A detached
tile compiler certifies fixed confidence-band order; uncertified tiles retain
the conditional Gaussian depth fibers and evaluate ordered optical transfer.
The Metal fork has a matching VJP. The current implementation is bounded to
the affine/static retained-depth route, fixed midpoint quadrature, at most 64
depth samples, and at most 256 active atoms.

## Paper Ablation Rows

Run these identities separately; never collapse them under one “World Tubes”
label:

| ID | World source | Alpha law | Renderer | Question |
| --- | --- | --- | --- | --- |
| WT-OT0 | `legacy_tube` | `peak_splat` | `metal_tile` | Historical World Tubes control |
| WT-OT1 | `full_spd4` | `beer_lambert` | `metal_tile` | Does native depth uncertainty plus physical alpha suffice? |
| WT-OT2 | `full_spd4` | `beer_lambert` | `retained_fiber_metal` | All-retained ordered-transfer oracle |
| WT-OT3 | `full_spd4` | `beer_lambert` | `hybrid_retained_fiber` | Does certification recover STAR speed without losing retained quality? |

For WT-OT1--3 use `fiber_integrated` amplitude first. Hold dataset split,
source initialization, atom count, trainable-scalar budget, target pixels,
optimizer steps, seed, quadrature extent, and background fixed. Report both an
equal-atom comparison and the already checked-in matched-scalar comparison
when space permits.

The first executable matched-scalar protocol pair is:

```text
WT-OT0:
  src/train_configs/paper_protocols/
  coffee_martini_protocol_bounded_16f_40step.jsonc

WT-OT1--3:
  src/train_configs/paper_protocols/
  coffee_martini_protocol_bounded_16f_40step_spd4_param_matched_199.jsonc
```

Start with seed 17 on the bounded 16-frame/40-step protocol. Do not add these
rows to the frozen 21-row public matrix. Promote to seeds 17/29/43 and public
scene breadth only on an approved host and only after WT-OT3 is selective.

The existing bounded native/smoke evidence is checked without launching MPS:

```bash
python3 \
  research_experiments/paper_runner_suite/verify_ordered_transfer_ablation.py
```

It writes
`artifacts/foundation_gates/world_tubes_ordered_transfer_ablation_verified.json`.
The verifier binds every consumed JSON by SHA-256, checks native forward and
all-source VJP parity, checks the `10/64` hybrid/oracle result, retains the
`64/64` dense negative control, and records that public quality/speed,
projective retained transfer, adaptive quadrature, and dense selectivity are
not established.

## Required Measurements

```text
quality:
  heldout PSNR, SSIM, LPIPS, L1
systems:
  compile, forward, backward, total step, peak driver bytes
hybrid:
  fallback tiles and fraction, reason bits, active atoms,
  minimum certified separation, depth samples, sigma extent
correctness:
  WT-OT3 image/VJP parity against WT-OT2 on every fallback stress fixture
ordering stress:
  colored-overlap commutator proxy and order-change count across camera time
```

## Promotion And Falsification

Existing evidence is deliberately mixed:

- the 16-atom smoke routes `10/64` tiles to fallback and matches WT-OT2 at
  recorded metric precision;
- the 199-atom dense initialization routes `64/64` tiles to fallback and is
  the required negative selectivity control.

Before a paper-speed row:

1. ablate physical depth initialization;
2. test a bounded color-commutator or error-certified support criterion;
3. require WT-OT3 to match WT-OT2 within the declared image/VJP tolerance;
4. require a material fallback reduction and end-to-end speed win over WT-OT2.

Stop or demote the extension if the ordinary-scene fallback fraction remains
above 20%, the hybrid does not beat all-retained wall time, retained transfer
does not improve colored-overlap quality, or fixed quadrature cannot meet a
declared error tolerance. Those outcomes do not invalidate the central World
Tubes interval-atlas result.

## Deferred, Not Implied

```text
adaptive/error-controlled forward and VJP quadrature
exact nonlinear/projective retained-depth records
certificate and integration-bound derivatives
native CUDA implementation
renaming World Tubes or WorldFoam around “holonomy”
```

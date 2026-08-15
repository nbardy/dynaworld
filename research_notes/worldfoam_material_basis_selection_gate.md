# WorldFoam Material Basis Selection Gate

**Status:** verified bounded synthetic result, 2026-07-27  
**Decision:** no universal M3/M5 winner; native-4D material integration remains gated

## Question

The fixed-segment M0--M5 gate proves that each material law and explicit VJP
matches its reference. It does not show that a richer law carries observable
value, or that convex log-P2 is preferable to a direct positive P2 with the
same storage.

The material-value gate therefore asks:

> When one physical material field is observed through multiple chords, which
> fixed-size law predicts held-out chord transfer most accurately?

This is a representation-capacity question. It is not an image-quality,
renderer-throughput, or camera-program compilation result.

## Identifiability result

For a constant-color segment with extinction \(\sigma(\xi)\),

\[
\tau=L\int_0^1 \sigma(\xi)\,d\xi,\qquad
\beta=e^{-\tau},\qquad
m=(1-\beta)c.
\]

Consequently, one complete segment identifies only total optical depth
\(\tau\). Any two density profiles with the same integral produce the same
\((\beta,m)\). Density shape cannot be ranked from a single complete
constant-color transfer element.

The smallest useful correction is to share one global material field across
multiple partial chords \([a,b]\). Different chords expose different
integrals, making shape observable without introducing a full image renderer.

## Controlled contract

The canonical executable is:

```text
research_experiments/world_foam_lane2/finite_element_material_fit.py
```

It uses:

- all six M0--M5 laws;
- identical physical length, constant target color, observations, and loss;
- twelve training chords and eight disjoint held-out chords;
- an independent target oracle: closed-form integration for positive
  Bernstein P2 and composite Simpson integration for convex log-P2;
- the production material evaluator for every fitted prediction;
- seeds `17`, `29`, and `43`;
- float64 Adam for 200 steps followed by the same 50-step strong-Wolfe L-BFGS
  polish for every law;
- canonical held-out loss over \((\beta,m)\);
- explicit serialized material sizes.

M3 and M5 each store three density coefficients plus one constant RGB:
six float32 scalars, or 24 bytes. Their comparison is therefore
payload-matched.

The artifact records 36 rows and embeds source hashes:

```text
artifacts/foundation_gates/worldfoam_material_value_fit_cpu_20260727.json
```

The independent verifier is:

```text
research_experiments/world_foam_lane2/verify_finite_element_material_fit.py
```

## Result

Median held-out losses across three seeds:

| Target field | M0 | M1 | M2 | M3 | M4 | M5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| positive Bernstein P2 | `6.84e-3` | `6.77e-3` | `6.19e-3` | **`5.26e-17`** | `6.32e-3` | `8.80e-5` |
| convex log-P2 | `8.10e-3` | `8.10e-3` | `8.03e-3` | `1.33e-3` | `8.05e-3` | **`6.19e-15`** |

All predeclared checks pass:

- M3 beats M0, M1, and M5 by more than `100x` on held-out positive-P2;
- M5 beats M0, M1, and M3 by more than `100x` on held-out convex-log-P2;
- M3 and M5 use equal serialized bytes;
- every row is finite;
- the saved medians, row matrix, interval split, source hashes, and
  non-promotion state pass the independent verifier.

The result is deliberately symmetric: each three-coefficient family recovers
its own generating law and approximates the other imperfectly.

## Interpretation

### What changed

The earlier decision question, “does M3 beat M5?”, was underspecified. The
answer depends on the target density family. Fixed storage alone does not
select a universal parameterization.

The key result is **basis complementarity**:

- direct positive P2 is the right local chart for polynomial density;
- convex log-P2 is the right local chart for Gaussian-like log density;
- P0/P1 controls verify that partial chords expose shape;
- neither exact-family synthetic win licenses universal promotion.

The artifact therefore records:

```text
winner = null
eligible_for_native_4d_integration = false
```

### What did not change

The public method remains World Tubes with STAR UVT as its implementation
lane. WorldFoam remains a retained-depth optical-transfer sibling. This
material gate does not justify replacing the World Tubes representation,
renaming the paper, or claiming real-scene superiority.

The four-layer decomposition remains useful:

1. **World:** stores a frame-independent spacetime field.
2. **Compiler:** intersects the camera program with world support and emits
   ordered chords.
3. **Evaluator:** maps each chord and material law to \((\beta,m)\).
4. **Adjoint:** reduces residuals through transfer, chord, compiler, and world
   VJPs.

M3 versus M5 is an evaluator/chart decision. It should not require six copies
of the compiler, topology, scan, or renderer.

### STAR opacity boundary

STAR's legacy peak-splat opacity

\[
\alpha=o\exp(-q/2)
\]

is not the same constitutive law as Beer--Lambert transfer

\[
\alpha=1-\exp\{-\tau_0\exp(-q/2)\}.
\]

The thin-opacity mapping agrees only to first order. A fair World Tubes versus
WorldFoam comparison therefore needs explicitly named opacity semantics. The
selected static/full-SPD/q-UVT RGB direct-atomic STAR route now has a
behaviorally verified CPU Beer--Lambert alpha/VJP mode and exact support
cutoff. Its parameter is still peak **projected** optical thickness, not the
retained-depth line integral of a WorldFoam density field. Projective atlas
paths fail closed. Peak-splat, projected Beer--Lambert, and retained-depth
optical-transfer results must remain separately labelled.

## Directed follow-up work

### P0 — close semantics and model selection

1. Run the opt-in behavior-level Beer--Lambert Metal forward/direct-VJP gate
   only in an approved quiet window. Keep projective trace paths fail-closed
   until their support/reference math has the same tested semantics.
2. Add a held-out **per-cell basis-selection** gate. Candidate policies:
   hard M3/M5 selection with a validation or description-length penalty;
   a stable hysteretic selector across camera-program cells; or a compact
   positive family that strictly contains both targets without silently
   increasing payload.
3. Add crossing translucent slabs and Gaussian sheets so multiple ordered
   materials, not only one isolated field, stress the shared scan.

### P1 — connect to the renderer without multiplying implementations

1. If a law or selector wins, integrate it plus M0 first into the RGB-only
   owner-run direct-atomic loss path.
2. Widen the material parameter/gradient ABI once. Keep owner/event words,
   segment endpoints, scan, and adjoint topology identical.
3. Add absorption-depth first moments before claiming exact expected depth;
   the current material ABI returns only `tau`, `beta`, and `m`.
4. Compare per-frame replay and compiled execution from one frozen learned
   world checkpoint, with identical samples, loss, background, precision, and
   budgets.

### P2 — only after a winner

Build the compact native-4D cell field/compiler and measure:

- serialized representation and optimizer bytes;
- event/tape growth with frames;
- chart, order, support, and fallback events;
- replay-versus-compiled forward and adjoint parity;
- held-out image quality and synchronized throughput.

## Non-directions

- Do not clone the roughly 94-kernel production renderer for each M0--M5 law.
- Do not call fixed-segment CPU/Metal parity a material-quality win.
- Do not choose M5 because it looks more physical, or M3 because it wins its
  own polynomial target.
- Do not advance either law to native-4D from this synthetic gate alone.
- Do not run publication-scale MPS experiments on the 24 GiB incident host.

## Verification record

```text
material-fit focused test:                 18 passed
fixed-segment derivative/branch suite:     34 passed, 3 Metal-only skipped
max central-difference VJP normalized err: 6.86e-10
independent fit-artifact verifier:          verified, 36 rows, 3 seeds
```

The saved 12-record Metal artifact remains the accepted device-parity result
and now includes the tiny-tau branch. It matches the current shader hash and
reports `7.51e-8` forward and `5.96e-8` VJP normalized error. The separate
CPU-only derivative artifact is
`artifacts/foundation_gates/worldfoam_material_m0_m5_cpu_12record_20260727.json`.

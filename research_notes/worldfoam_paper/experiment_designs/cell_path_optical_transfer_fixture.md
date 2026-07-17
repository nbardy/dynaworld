# Cell-Path Optical-Transfer Fixture Implementation Plan

Status: implementation spec, written 2026-07-06.

Purpose: turn the WorldFoam math appendix into the first falsifiable code gate.
This fixture should prove that the optical-transfer algebra, compiled
cell-path replay, and fixed-topology VJP are real before boundary flux,
Magnus compression, witness scores, feature transfer, Metal shaders, or real
scene quality claims get promoted.

## Short Answer

Yes: the next code is known. Implement a pure CPU/Torch fixture that evaluates
constant-density owner-run cell words as optical-transfer elements, compares a
compiled word to same-representation per-frame replay, and checks analytic VJP
formulas against finite differences.

Do not start by editing the hot WorldFoam shader path. The first code should be
small, deterministic, and hard to game.

## Code Targets

### New module

```text
research_experiments/world_foam_lane2/cell_path_optical_transfer_fixture.py
```

Public functions/classes to implement:

```text
TransferElement(beta, m)
compose(front, back)
scan(elements)
decode(element, background)
constant_run_element(sigma, length, color)
render_word(sigmas, lengths, colors, background)
render_word_from_elements(elements, background)
make_two_run_fixture()
make_three_run_fixture()
same_representation_replay_fixture()
analytic_prefix_suffix_vjp(sigmas, lengths, colors, background, target)
finite_difference_vjp(sigmas, lengths, colors, background, target, epsilon)
commutator_swap_probe()
run_all_checks()
write_summary_json(path, result)
```

Keep the implementation in float64 by default. Use tensors or small dataclasses;
avoid coupling the first fixture to MPS, W&B, DeepView data, native extensions,
or the current train loop.

### New pytest file

Preferred location:

```text
research_experiments/world_foam_lane2/test_cell_path_optical_transfer_fixture.py
```

Tests to implement:

```text
test_visibility_monoid_associative
test_constant_run_matches_manual_alpha
test_cell_path_replay_equivalence
test_cell_path_vjp_matches_finite_difference
test_commutator_swap_bound
test_fixture_summary_schema
```

Only promote shared helpers into `src/train/` after these tests are green and
there is a second caller. If promotion becomes useful, the likely destination
is:

```text
src/train/worldfoam_optical_transfer.py
```

## Algebra Contract

Represent one event/run as an affine optical-transfer element:

```text
F_i(g) = m_i + beta_i g
beta_i = exp(-sigma_i length_i)
m_i = (1 - beta_i) color_i
```

For front-to-back rendering over background `b`:

```text
C = F_0(F_1(...F_{n-1}(b)))
```

Composition must obey:

```text
compose(a, b) = (a.beta * b.beta, a.m + a.beta * b.m)
F_comp(g) = F_a(F_b(g))
```

This gives the visibility monoid used by the appendix. It also matches sorted
alpha compositing when each splat atom is lowered to:

```text
beta = 1 - alpha
m = alpha * color
```

## Replay Equivalence Contract

The same-representation replay test should build two words over the same
constant-density cells:

```text
compiled_word = [run_0, run_1, ...]
replay_word   = [run_0, run_1, ...]
```

The fixture is intentionally simple at first: the two paths can share the same
synthetic fixture data, but they must go through distinct code paths:

```text
compiled path:
    prebuild TransferElement records, then scan.

replay path:
    recompute each constant-run element from sigma/length/color, then scan.
```

Pass condition:

```text
max_abs(compiled_color - replay_color) <= 1e-12
max_abs(compiled_element - replay_element) <= 1e-12
```

This is the first paper-critical claim: a compiled cell-path atlas is allowed
to amortize event work, but it is not allowed to change representation
semantics.

## VJP Contract

For loss:

```text
L = 0.5 * ||C - target||^2
```

Let:

```text
T_before_i = product_{j<i} beta_j
C_after_i  = F_{i+1}(...F_{n-1}(background))
grad_C     = C - target
```

Then the fixed-topology derivatives are:

```text
dL/dm_i      = T_before_i * grad_C
dL/dbeta_i   = dot(grad_C, T_before_i * C_after_i)
dL/dtau_i    = dot(grad_C, T_before_i * beta_i * (color_i - C_after_i))
dL/dsigma_i  = length_i * dL/dtau_i
dL/dlength_i = sigma_i * dL/dtau_i
dL/dcolor_i  = T_before_i * (1 - beta_i) * grad_C
```

Check analytic derivatives against central finite differences for:

```text
beta
m
DeltaTau
sigma
length
color
```

Initial tolerances:

```text
render_max_abs_error <= 1e-12
element_max_abs_error <= 1e-12
grad_max_abs_error <= 1e-6
finite_difference_epsilon = 1e-5
```

If the finite-difference gradients are noisy, tune epsilon in the fixture and
record the chosen value in the summary JSON. Do not relax thresholds silently.

## Commutator Probe

The first commutator test should be a two-layer swap:

```text
word_ab = [A, B]
word_ba = [B, A]
```

For elements with colors `c_a`, `c_b` and opacities `alpha_a`, `alpha_b`, the
swap color difference is controlled by:

```text
alpha_a * alpha_b * (c_a - c_b)
```

The test should verify the measured swap direction and magnitude for a few
small fixtures. This keeps the commutator theorem connected to code without
starting with full moving-boundary topology.

## Summary Artifact

The standalone script should write a JSON summary. Suggested command:

```bash
PYTHONPATH=src/train uv run python research_experiments/world_foam_lane2/cell_path_optical_transfer_fixture.py \
  --out outputs/benchmarks/2026-07-06_worldfoam_cell_path_optical_transfer_summary.json
```

Suggested schema:

```json
{
  "status": "ok",
  "dtype": "float64",
  "seed": 0,
  "fixture": "constant_density_owner_run_word",
  "thresholds": {
    "render_max_abs_error": 1e-12,
    "element_max_abs_error": 1e-12,
    "grad_max_abs_error": 1e-6
  },
  "checks": {
    "monoid_associative": "ok",
    "alpha_equivalence": "ok",
    "replay_equivalence": "ok",
    "vjp_finite_difference": "ok",
    "commutator_swap": "ok"
  },
  "max_errors": {
    "render": 0.0,
    "element": 0.0,
    "grad": 0.0
  }
}
```

Use exact numbers from the run; zeros above are placeholders.

## Test Command

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  research_experiments/world_foam_lane2/test_cell_path_optical_transfer_fixture.py -q
```

This is a local correctness gate. It does not require W&B, MPS, CUDA, native
extensions, official fixtures, or real loaded frames.

## Failure Meanings

```text
monoid associativity fails:
    The transfer element definition or composition order is wrong.

alpha equivalence fails:
    The optical-transfer algebra does not reproduce ordinary alpha compositing.

replay equivalence fails:
    The compiled atlas is changing semantics instead of amortizing work.

VJP finite difference fails:
    Prefix/suffix backward formulas are not paper-ready.

commutator swap fails:
    The non-commutation theorem is disconnected from the implemented algebra.

summary schema fails:
    The artifact is not machine-checkable enough to cite later.
```

## Promotion Ladder

1. Land the pure CPU fixture and tests above.
2. Add one denser synthetic raymarch/reference comparison if the constant-run
   fixture is green.
3. Only then add moving-boundary checks for face/sphere endpoint derivatives.
4. Only after endpoint derivatives pass, test interface-flux witness scores.
5. Only after those pass, test Magnus/commutator compression.
6. Only after synthetic exactness is green, touch real-scene replay baselines or
   hot shader code for paper claims.

## Non-Goals For This Fixture

```text
Metal or MPS performance
native extension changes
DeepView/Neural3D quality
training-loop integration
boundary flux derivatives
Hessian or second-order terms
Magnus compression
feature-gauge transfer
universal ray-space transfer
```

Those are not bad ideas. They are simply downstream of this exactness gate.

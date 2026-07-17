# 07 - Adjoint Training

The forward compiler stores trace charts:

```text
K_Gamma = {C_a, chi_a, alpha_{i,a}, c_{i,a}, z_hat_{i,a}, order_a}
```

Training can reuse the same atlas by compiling the residual/adjoint field over
sensor-time.

Let:

```text
L = integral_B ell(I(y), I*(y)) dy
A(y) = partial L / partial I(y)
```

Inside a fixed-order stratum:

```text
I(y) = sum_m T_m(y) alpha_{pi_m}(y) c_{pi_m}(y)
```

For primitive `i`:

```text
partial I / partial c_i = T_i alpha_i
partial I / partial alpha_i = T_i (c_i - I_behind_i)
```

Then:

```text
partial L / partial theta_i
= sum_a integral_{C_a}
    A(y)^T partial I(y)/partial theta_i dy
```

Represent the adjoint locally:

```text
A_a(y) ~= sum_m a_{am} psi_{am}(y)
```

For Gaussian/rational traces, gradients reduce to moments:

```text
integral_C psi(y) Gaussian_or_rational_trace(y) polynomial(y) dy
```

## Practical Training Mode

First implementation should not differentiate through all compiler decisions.
Use block coordinate training:

```text
compile atlas
train short block using trace params
map trace gradients to world params or refit world atoms
recompile affected charts
repeat
```

This avoids brittle symbolic derivatives through chart selection, split
decisions, and visibility strata.

## Test

Use a frozen synthetic scene and compare:

```text
autograd per-frame renderer gradients
compiled trace-adjoint gradients
```

for opacity, color, center, and simple camera-gauge coefficients.

# 02 - Gaussian Fiber Pushforward

This is the bridge from invariant bundle math to the current STAR UVT tensor
contract.

Assume a world spacetime Gaussian:

```text
rho_i(x) = a_i exp[-1/2 (x - m_i)^T Lambda_i (x - m_i)]
x in M
```

In a local gauge:

```text
Gamma_a(y, z) ~= x0 + J_y delta_y + J_z delta_z
```

Let:

```text
eta = [delta_y, delta_z]
J = [J_y J_z]
delta = m_i - x0
```

Then:

```text
rho_i(Gamma_a(y,z))
~= a_i exp[-1/2 (J eta - delta)^T Lambda_i (J eta - delta)]
```

Expand:

```text
(J eta - delta)^T Lambda_i (J eta - delta)
= eta^T H eta - 2 g^T eta + const

H = J^T Lambda_i J
g = J^T Lambda_i delta
```

Partition by base/fiber:

```text
H = [H_yy H_yz
     H_zy H_zz]
g = [g_y
     g_z]
```

Fiber integration over `z` gives a UVT Gaussian with precision:

```text
S = H_yy - H_yz H_zz^{-1} H_zy
```

and conditional fiber mean:

```text
z_hat(y) = z0 + H_zz^{-1}(g_z - H_zy delta_y)
```

For a scalar fiber, `H_zz > 0`, a locally constant fiber-measure factor `J_0`,
and an untruncated local fiber domain, completing the square gives the missing
footprint amplitude too:

```text
q_y(delta_y)
  = delta_y^T S delta_y
    - 2 (g_y - H_yz H_zz^{-1} g_z)^T delta_y
    + delta^T Lambda_i delta - g_z^T H_zz^{-1} g_z

bar_rho_i(y)
  ~= J_0 a_i sqrt(2 pi / H_zz) exp[-1/2 q_y(delta_y)]
```

If the fiber domain is clipped or the measure Jacobian varies materially over
the local depth uncertainty, this closed form is only a local approximation;
the compiler must retain the resulting residual/support certificate or use
quadrature.

This is the Schur complement. In bundle language, it is simply the local
coordinate formula for:

```text
pi_* Gamma^* rho_i
```

## Current STAR Mapping

The existing UVT tensors correspond to:

```text
ma       -> local trace mean in y = (u,v,tau)
q_uvt    -> S, stored as 6 upper-triangular coefficients
depth0   -> z_hat(ma)
depth_beta -> first-order coefficients for z_hat(y)
```

The missing richer fields are:

```text
depth_variance / uncertainty
chart_gauge_id
denominator/rational coefficients
fit error or validity certificate
```

## Test

Construct a synthetic 4D Gaussian, a linear camera map, and compare:

1. dense numerical integration along `z`
2. Schur-complement UVT Gaussian

This should match to numerical quadrature tolerance. If it does not, either the
precision partition or the fiber measure is wrong.

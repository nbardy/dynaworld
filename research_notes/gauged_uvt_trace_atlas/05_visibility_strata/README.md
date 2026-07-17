# 05 - Visibility Strata

The trace atlas stores more than opacity. It also stores conditional fiber
statistics:

```text
alpha_i(y)
z_hat_i(y)
Var_i(z | y)
```

Visibility order is a stratification of sensor-time:

```text
B = union_r S_r
```

where each stratum has stable or safely commutable order.

## Order Boundaries

Two primitives change order where:

```text
z_hat_i(y) = z_hat_j(y)
```

In affine depth charts this is a plane. In projective/rational charts it is a
zero set of rational functions. The compiler should store the resulting order
locally, not sort from scratch per frame.

The newest trace contract permits a pixel-varying local depth section:

```text
z_i(u,v,t) =
  z_{c,i}(t)
  + z_{u,i}(t) * (u - u_{c,i}(t))
  + z_{v,i}(t) * (v - v_{c,i}(t))
```

Then an order boundary between two traces inside one tile is no longer just a
center-depth crossing. It is the zero set:

```text
z_i(u,v,t) - z_j(u,v,t) = 0
```

If the slopes are affine/quadratic in time and the center traces are
projective/rational, the compiler should bound this section over the tile,
split on visible roots when needed, or mark the cell for live-depth fallback.
The current helper evaluates this model in Torch for certificates. The
production interval Metal sorter still needs a follow-up kernel/path before it
can consume the pixel-varying depth plane directly.

## Uncertainty-Aware Order

Use intervals:

```text
I_i(C) =
[inf_C z_hat_i - k sigma_i - eps_i,
 sup_C z_hat_i + k sigma_i + eps_i]
```

If:

```text
max I_i(C) < min I_j(C)
```

then `i` is definitely in front of `j` over chart cell `C`.

## Visible Swap Bound

If order is unresolved, the color effect of swapping two translucent layers is
bounded by:

```text
|Delta I_ij(y)| <= alpha_i(y) alpha_j(y) |c_i(y) - c_j(y)|
```

So unresolved order is acceptable when:

```text
sup_C alpha_i alpha_j |c_i - c_j| < eps_order
```

This is the difference between mathematical visibility and practical visibility:
the compiler resolves order only when it matters to the image.

## Test

Create crossing semi-transparent layers. Measure:

```text
order_flip_cells
commuted_pair_error
fallback_fraction
PSNR/SSIM vs dense per-sample sorted reference
```

If `fallback_fraction` dominates ordinary scenes, the representation is not
compressing visibility well enough.

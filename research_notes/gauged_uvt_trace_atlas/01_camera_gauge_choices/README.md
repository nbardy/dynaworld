# 01 - Camera Gauge Choices

The gauge is the local coordinate choice on the ray fiber:

```text
chi_a: E_Gamma|C_a -> C_a x D_a
```

It should make pulled-back primitives simple. The gauge is not a cosmetic
choice; it decides whether an orbit segment is represented by a low-order trace
or explodes into many frame-local bins.

## Candidate Gauges

Ordinary depth:

```text
z = s
```

Good for small camera motion, far objects, narrow FOV, and current affine UVT.
Bad near the camera and near projective denominator events.

Inverse depth:

```text
z = 1 / s
```

Good for near/far scale stability and wide depth ranges. Bad when geometry
crosses infinity or when linear depth order is needed without conversion.

Log depth:

```text
z = log s
```

Good for multiplicative scale changes. Useful for camera dolly/zoom sweeps.

Projective denominator:

```text
z = h_z
```

Good for orbit/revolve and homogeneous camera-time expressions. A projected
center becomes rational:

```text
u(t) = h_u(t) / h_z(t)
v(t) = h_v(t) / h_z(t)
```

Behind-camera and image-infinity events are `h_z = 0` chart boundaries.

Object-local gauge:

```text
z = distance along ray in an object/instance coordinate frame
```

Good for moving rigid or articulated instances. Lets object motion live in the
instance embedding rather than in screen-space velocity.

Foam-cell gauge:

```text
z = local cell coordinate along ray
```

Good for WorldFoam/PowerFoam cells, because active-set and material support are
cell-local rather than primitive-local.

## Decision Rule

Choose the gauge that minimizes trace complexity over the proposed chart:

```text
complexity = fit_error + denominator_risk + visibility_order_variation
             + memory_cost + derivative_condition_number
```

The compiler should rank gauges before splitting. Splitting is not the first
tool; it is what happens after a better gauge fails the chart tolerance.

## Cheap Diagnostic

For a candidate chart, evaluate the true projected centers at sigma points and
fit:

```text
affine UVT
quadratic UVT
projective rational UVT
```

If projective rational reduces max center residual by >10x on orbit windows,
that confirms the gauge is carrying real camera geometry rather than merely
reshuffling error.

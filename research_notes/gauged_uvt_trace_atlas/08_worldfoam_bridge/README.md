# 08 - WorldFoam Bridge

WorldFoam/PowerFoam can be expressed in the same bundle language.

Foam cells define world support regions:

```text
F_j subset M
```

The camera pulls a cell back to the ray bundle:

```text
Gamma^{-1}(F_j) subset E_Gamma
```

and fiber integration can give a sensor-time trace:

```text
bar_rho_j = pi_* Gamma^* rho_j
```

For WorldFoam, however, this pushed-down trace is not the main visibility
object. The important retained object is:

```text
sigma_l(y,z) = sum_j 1_{Gamma_l(y,z) in F_j}
                    sigma_j(Gamma_l(y,z))
                    J_l(y,z)
```

World Tubes uses the depth fiber as the dimension to marginalize, then keeps
conditional depth/order certificates. WorldFoam keeps the depth fiber alive and
uses cumulative optical depth:

```text
tau_l(y,z) = integral_{z_front}^{z} sigma_l(y,s) ds
T_l(y,z)   = exp(-tau_l(y,z))
I(y)       = integral T_l(y,z) sigma_l(y,z) c_l(y,z) dz
```

The difference from splats is therefore stronger than "cell-local support":
support and material structure are cell-local, and visibility is a
ray-fiber-prefix problem rather than a projected primitive sort problem.

## Why This Helps Revolving Cameras

A revolving camera changes screen footprint aggressively, but the world cell
can remain stable. The compiler should trace:

```text
cell-camera intersections
```

instead of relearning screen tubes from scratch.

## Required Additions

WorldFoam needs:

```text
ray/cell intersection charts
retained depth-fiber intervals or layer statistics
transmittance prefix/suffix summaries
cell-local visibility strata
known-camera path compilation
```

It should not rely on unconstrained learned camera motion for the first orbit
prototype. Current notes already show learned-camera orbit attempts are fragile;
known calibrated/synthetic orbit is the right initial target.

## Bridge Test

Create static foam cells on a synthetic sphere/cube, render an orbit with known
cameras, and compare:

```text
per-frame foam rendering
compiled cell trace atlas
compiled retained-fiber transmittance atlas
```

Metrics:

```text
active cell cache size
chart count per orbit degree
PSNR vs per-frame reference
intersection/rebuild amortization point
```

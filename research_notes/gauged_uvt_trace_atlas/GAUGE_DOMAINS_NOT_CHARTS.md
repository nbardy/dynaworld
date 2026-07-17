# Gauge Domains, Not Weak Charts

Date: 2026-05-24

## Why This Exists

The word "chart" is mathematically correct, but it is easy to hear it as
"piecewise hack" or "we gave up on the richer fiber-bundle idea." That is not
the intended theory.

The invariant object is still:

```text
UVT trace = pi_* Gamma^* world_primitive
```

with:

```text
pi: E_Gamma -> B
B = Omega x T
Gamma: E_Gamma -> M
```

A "chart" in this project means a local trivialization of the camera-ray bundle:

```text
chi_a: E_Gamma|C_a -> C_a x D_a
```

The better implementation phrase is:

```text
gauge domain with validity certificates
```

or, when the domain is defined by geometric events:

```text
event-certified fiber cell
```

## What The Domain Certifies

A gauge domain `C_a subset B` is not just saying "sort order is stable."

It certifies the whole local rendering contract:

```text
projection / denominator is well behaved
trace representation error <= epsilon_trace
support bounds are conservative
tile-time active set is valid
depth model is valid
order is stable, visibly commutable, or marked fallback
interval gates prevent segment leakage
backward support matches forward support
```

That is why the compiler needs local domains even when the math is global.

## If We "Throw Away Charts"

There are three honest meanings.

### 1. Exact Global Pullback

Keep one global expression for:

```text
Gamma^* rho
```

and evaluate:

```text
pi_* Gamma^* rho
```

directly over the whole camera path.

This is beautiful when it works. For simple projective camera orbits, global
rational traces can cover a lot. But the renderer still needs event partitions:

```text
h_z = 0
near/far crossings
image/tile boundary crossings
support birth/death
depth-order swaps
disocclusion boundaries
visibility ambiguity
```

So the "charts" disappear as a name, then return as event cells.

### 2. Fully GPU Per-Sample Evaluation

Evaluate the global pullback and fiber integral per pixel/time sample on the
GPU.

This removes the CPU compiler and local fit logic, but it also gives back much
of the intended amortization:

```text
projection + support + binning + order
```

starts growing with frame count again unless the GPU kernels build equivalent
event/cell metadata on device.

### 3. One Conservative Global Support

Store one huge support and one global ordering proxy for each primitive over the
whole orbit.

This is usually the wrong collapse:

```text
tile lists get fat
overdraw rises
visibility ambiguity rises
fallback rises
memory approaches per-frame binning
```

It looks simpler but usually loses the camera-path compiler win.

## The Better Replacement

Do not throw away local domains. Replace weak charting with event-certified
gauge domains.

The compiler should prefer:

```text
global group/orbit parameterization where valid
projective or rational gauge before affine fitting
event surfaces before arbitrary splitting
transition maps before duplicate state
explicit interval gates before temporal Gaussian masking
fallback only after certificate failure
```

In other words:

```text
group/fiber/pullback math defines the object
gauge domains define where a cheap expression is valid
event certificates define where domains start/stop
Metal kernels evaluate the certified domains
```

## Revolving Camera Interpretation

For a revolving camera, a primitive trace may be rational in an orbit coordinate
such as:

```text
r = tan(theta / 2)
```

That can make the global pullback much cleaner than affine frame time. The
compiler should exploit this before splitting.

But even with the right orbit gauge, true events remain:

```text
behind-camera transitions
denominator zeros
near-plane crossings
support entering/leaving the image
depth-order swaps
occlusion/disocclusion boundaries
```

Those events are not defects in the math. They are the geometry of the camera
bundle. The atlas is the event-cell decomposition of that geometry.

## Implementation Translation

Current code names:

```text
ProjectiveTraceWindow
ProjectiveTraceSupportBounds
ProjectiveTraceVisibilitySidecar
ProjectiveTraceTileTimeCell
ProjectiveTraceUVTBridge.active_start / active_stop
```

should be read as:

```text
gauge-domain certificate
support certificate
visibility/order certificate
event-certified tile-time cell
interval gate for a lowered local trivialization
```

Long-term naming can move from:

```text
chart/window
```

to:

```text
gauge_domain / event_cell
```

but the current code is already enforcing the right mathematical role: validity
domains plus certificates, not blind fitted patches.

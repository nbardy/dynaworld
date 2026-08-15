# Certified correctness

bounded event-certified projective chart segments; no 360/720 multi-chart claim

| Claim | Metric | Value | Acceptance | Source |
|---|---|---:|---:|---|
| Fiber value is gauge invariant | max relative error | 3.50087e-13 | <= 1e-10 | gauge_value |
| Fiber gradient is gauge invariant | max gradient relative error | 2.32523e-12 | <= 1e-9 | gauge_gradient |
| Compiled atlas matches dense/replay image | max absolute image error | 0 | <= 1e-5 | decisive_demo |
| Unstratified interval exposes an order-crossing failure | raw crossing quality error | 0.186742 | > 1e-5 (expected failure) | visibility |
| Visibility crossing is repaired by stratification | stratified crossing quality error | 0 | <= 1e-5 | visibility |
| Finite exposure / rolling shutter forward parity | max Metal absolute error | 5.96046e-08 | <= 1e-5 | exposure |
| Finite exposure / rolling shutter gradient parity | max Metal gradient relative error | 6.37738e-07 | <= 1e-5 | exposure_backward |
| Mixed fallback preserves gradients | max mixed gradient relative error | 7.40632e-07 | <= 1e-5 | mixed_fallback_backward |
| Bounded-orbit chart reuses trace state at F=128 | fixed/per-frame trace-count ratio | 0.03125 | < 0.25 | scaling |

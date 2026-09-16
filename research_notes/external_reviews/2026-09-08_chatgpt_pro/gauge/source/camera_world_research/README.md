# Camera–World Co-Design for Dynamic Differentiable Rendering

Research manuscript prepared for Nicholas Bardy, 8 September 2026.

## Contents

- `manuscript.pdf`: 20-page manuscript (approximately 19 pages of research content, with references starting on page 19).
- `manuscript.tex`: editable LaTeX source; bibliography and primary-source links are embedded.
- `checks.py`, `checks.json`: reproducible algebra and local-adjoint checks and their actual output.
- `ellipse_coverage.py`, `ellipse_checks.json`: single-ellipse/box exact area formula, analytic conic boundary-moment adjoint, and verification output.
- `environment.json`: actual Python, NumPy, and SciPy environment used.
- `requirements.txt`: Python dependencies.

## Reproduce the CPU checks

Python 3.10 or later is required. In a virtual environment:

```sh
python -m pip install -r requirements.txt
python checks.py --output checks.json
python ellipse_coverage.py --output ellipse_checks.json
```

The numerical tests use fixed seeds. Small last-bit differences may occur with another numerical library or platform. These are correctness checks, not performance benchmarks.

The first suite checks coupled rigid-frame invariance, camera/object congruence adjoints, regular ray-root adjoints, shared coefficient adjoints, triangle solve adjoints, a moving-edge filtered derivative, and optical-transfer composition. Its largest local finite-difference discrepancy was approximately 6.90e-10; shared and direct coefficient adjoints differed by approximately 1.72e-16.

The second suite checks 80 ellipse/box cases against independent adaptive one-dimensional area integration and 240 conic directional derivatives against finite differences. Maximum area error was approximately 4.70e-14; maximum normalized conic-VJP discrepancy was approximately 1.87e-8. A separate camera-quadric-to-coverage chain check had maximum normalized discrepancy approximately 3.63e-9.

Normalized error means an error divided by max(1, magnitude of the two compared quantities). It is an absolute test for small derivatives, not an unqualified relative-accuracy claim.

## Use the coverage oracle

```python
import numpy as np
from ellipse_coverage import conic_from_affine, coverage

center = np.array([0.4, 0.6])
shape_map = np.array([[0.7, 0.0], [0.1, 0.4]])
conic = conic_from_affine(center, shape_map)
area, grad_conic = coverage(conic, box=(0.0, 1.0, 0.0, 1.0))
```

The conic interior is `[u, 1].T @ conic @ [u, 1] <= 0`. `grad_conic` uses the full symmetric-matrix Frobenius convention: perturbing both symmetric off-diagonal entries contributes twice the corresponding stored gradient entry. For non-unit pixel area, divide coverage by that area; multiply its adjoint by the incoming color-gradient dot the foreground/background radiance jump.

The core coverage calculation needs only NumPy. SciPy is used for independent verification. The implementation has floating-point predicate tolerances and does not provide certified behavior arbitrarily close to degeneracy.

## Compile the manuscript

A LaTeX distribution with the packages in the preamble is needed:

```sh
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
```

## Scope and limitations

This package contains a mathematical research manuscript and small CPU oracles. It is not a full multi-object renderer, a kinetic-visibility compiler, a Gaussian-splatting implementation, a GPU benchmark, or a scene-reconstruction system. The single-ellipse coverage and its boundary adjoint are implemented. General multi-body filtered visibility, nonconstant-radiance boundary quadrature, robust degeneracy handling, captured-scene expressivity, and matched-quality GPU advantages remain unvalidated.

The manuscript's conditional complexity claims retain explicit image-output and incoming-adjoint costs. No speedup over STG, ordinary quadric instancing, or hardware triangle rendering is claimed as measured.

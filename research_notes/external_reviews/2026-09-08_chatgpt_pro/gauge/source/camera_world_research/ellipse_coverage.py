"""Exact one-ellipse / axis-aligned-box area and an arc-moment conic VJP.

Scope: bounded nondegenerate ellipses and fixed box pixels. This is a CPU
mathematical oracle, not a robust general multi-object renderer. Uncertain
near-degenerate predicates require a production precision/fallback policy.
Only NumPy is required for the core; verification also uses SciPy.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

TAU = 2 * np.pi


def det2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def parameters(f: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    f = (np.asarray(f, dtype=float) + np.asarray(f, dtype=float).T) / 2
    if f.shape != (3, 3):
        raise ValueError('Expected a symmetric 3x3 conic matrix')
    s, v, c = f[:2, :2], f[:2, 2], f[2, 2]
    np.linalg.cholesky(s)  # Require positive definite bounded ellipse.
    m = -np.linalg.solve(s, v)
    r2 = float(v @ np.linalg.solve(s, v) - c)
    if r2 <= 0:
        raise ValueError('Conic has no nondegenerate ellipse interior')
    l = np.linalg.cholesky(r2 * np.linalg.inv(s))
    return m, l, r2


def conic_from_affine(m: np.ndarray, l: np.ndarray) -> np.ndarray:
    s = np.linalg.inv(l @ l.T)
    f = np.empty((3, 3))
    f[:2, :2] = s
    f[:2, 2] = f[2, :2] = -s @ m
    f[2, 2] = m @ s @ m - 1
    return f


def coverage(f: np.ndarray, box=(0., 1., 0., 1.)) -> tuple[float, np.ndarray]:
    """Return intersection area and Frobenius gradient with respect to f.

    Region: [u,1]^T f [u,1] <= 0. Box order: xmin, xmax, ymin, ymax.
    The off-diagonal entries of the gradient use the FULL symmetric-matrix
    Frobenius convention: a paired f_ij/f_ji perturbation has 2*gradient_ij.
    """
    m, l, r2 = parameters(f)
    xmin, xmax, ymin, ymax = map(float, box)
    if not (xmin < xmax and ymin < ymax):
        raise ValueError('Box must have positive width and height')
    tol = 5e-12
    def point(a):
        return m + l @ np.array([np.cos(a), np.sin(a)])
    def in_box(p):
        return xmin - tol <= p[0] <= xmax + tol and ymin - tol <= p[1] <= ymax + tol
    def in_ellipse(p):
        z = np.linalg.solve(l, p - m)
        return float(z @ z) <= 1 + tol
    sides = [(0, xmin), (0, xmax), (1, ymin), (1, ymax)]
    crossings: list[tuple[float, np.ndarray]] = []
    for axis, edge in sides:
        a, b = l[axis]
        rad = np.hypot(a, b)
        h = (edge - m[axis]) / rad
        if abs(h) > 1:
            continue
        base, offset = np.arctan2(b, a), np.arccos(np.clip(h, -1, 1))
        for ang in ((base - offset) % TAU, (base + offset) % TAU):
            p = point(ang)
            if in_box(p):
                crossings.append((float(ang), p))
    angles = sorted(a for a, _ in crossings)
    unique = []
    for a in angles:
        if not unique or abs(a - unique[-1]) > 1e-10:
            unique.append(a)
    if len(unique) > 1 and TAU - unique[-1] + unique[0] < 1e-10:
        unique.pop()
    spans = [(0., TAU)] if not unique else list(zip(unique, unique[1:] + [unique[0] + TAU]))
    area, moment = 0., np.zeros((3, 3))
    det_l = float(np.linalg.det(l))
    vmap = np.eye(3); vmap[:2, :2] = l; vmap[:2, 2] = m
    for a, b in spans:
        if not in_box(point((a + b) / 2)):
            continue
        ea, eb = np.array([np.cos(a), np.sin(a)]), np.array([np.cos(b), np.sin(b)])
        delta = b - a
        area += .5 * (det2(m, l @ (eb - ea)) + det_l * delta)
        z = np.array([np.sin(b) - np.sin(a), np.cos(a) - np.cos(b)])
        dz = np.sin(2*b) - np.sin(2*a)
        off = (np.cos(2*a) - np.cos(2*b)) / 4
        zz = np.array([[delta/2 + dz/4, off], [off, delta/2 - dz/4]])
        j = np.zeros((3, 3)); j[:2, :2] = zz; j[:2, 2] = j[2, :2] = z; j[2, 2] = delta
        moment += vmap @ j @ vmap.T
    corners = [np.array([xmin, ymin]), np.array([xmax, ymin]), np.array([xmax, ymax]), np.array([xmin, ymax])]
    for p, q in zip(corners, corners[1:] + corners[:1]):
        d = q - p; d2 = d @ d
        candidates = [0., 1.]
        for _, c in crossings:
            t = float((c - p) @ d / d2)
            if -tol <= t <= 1 + tol and abs(det2(c-p, d)) <= tol * max(1, np.linalg.norm(d)):
                candidates.append(float(np.clip(t, 0, 1)))
        ts = sorted(set(round(t, 13) for t in candidates))
        for a, b in zip(ts[:-1], ts[1:]):
            if in_ellipse(p + ((a+b)/2) * d):
                area += .5 * det2(p + a*d, p + b*d)
    grad_f = -det_l / (2 * r2) * moment
    # Deliberately do not clamp the computed area: expose rounding/predicate errors.
    return float(area), grad_f


def reference_area(f, box=(0., 1., 0., 1.)):
    from scipy.integrate import quad
    m, l, _ = parameters(f)
    s = (f[:2, :2] + f[:2, :2].T) / 2
    xmin, xmax, ymin, ymax = box
    xrad = np.linalg.norm(l[0])
    lo, hi = max(xmin, m[0]-xrad), min(xmax, m[0]+xrad)
    if hi <= lo:
        return 0.
    def height(x):
        a = s[1, 1]
        b = s[0, 1] * x + f[1, 2]
        c = s[0, 0] * x*x + 2*f[0, 2]*x + f[2, 2]
        disc = b*b-a*c
        if disc <= 0:
            return 0.
        root = np.sqrt(disc)
        return max(0., min(ymax, (-b+root)/a) - max(ymin, (-b-root)/a))
    points = []
    for y in (ymin, ymax):
        a = s[0, 0]; b = s[0, 1] * y + f[0, 2]
        c = s[1, 1] * y*y + 2*f[1, 2]*y + f[2, 2]
        disc = b*b-a*c
        if disc > 0:
            points.extend(x for x in ((-b-np.sqrt(disc))/a, (-b+np.sqrt(disc))/a) if lo < x < hi)
    val, _ = quad(height, lo, hi, epsabs=2e-12, epsrel=2e-12, points=sorted(set(points)), limit=250)
    return float(val)


def run_checks():
    rng = np.random.default_rng(612431)
    area_errors, vjp_errors, gauge_errors = [], [], []
    for _ in range(80):
        m = rng.uniform(-.35, 1.35, 2)
        l = np.array([[rng.uniform(.2, 1.1), 0.], [rng.uniform(-.3, .3), rng.uniform(.2, 1.1)]])
        f = conic_from_affine(m, l)
        area, grad = coverage(f)
        area_errors.append(abs(area-reference_area(f)))
        gauge_errors.append(abs(float(np.sum(grad*f))))
        for _ in range(3):
            z = rng.normal(size=(3, 3)); z = (z+z.T)/2; z /= np.linalg.norm(z)
            h = 2e-6
            num = (coverage(f+h*z)[0]-coverage(f-h*z)[0])/(2*h)
            ana = float(np.sum(grad*z))
            vjp_errors.append(abs(num-ana)/max(1., abs(num), abs(ana)))
    # Camera-space quadric-to-screen-conic adjoint on a nondegenerate sphere.
    b = np.array([[1.,0.,0.,-.4],[0.,1.,0.,-.2],[0.,0.,1.,-3.],[-.4,-.2,-3.,8.2]])
    def eval_b(b):
        m, g, c = b[:3,:3], b[:3,3], b[3,3]
        return coverage(c*m-np.outer(g,g), box=(-.2,.25,-.15,.3))
    area, gf = eval_b(b)
    gb = np.empty((4,4)); gb[:3,:3] = b[3,3]*gf
    gb[:3,3] = gb[3,:3] = -gf @ b[:3,3]
    gb[3,3] = np.sum(gf*b[:3,:3])
    berrors=[]
    for _ in range(20):
        z=rng.normal(size=(4,4)); z=(z+z.T)/2; z/=np.linalg.norm(z)
        h=1e-6
        fd=(eval_b(b+h*z)[0]-eval_b(b-h*z)[0])/(2*h)
        ad=float(np.sum(gb*z)); berrors.append(abs(fd-ad)/max(1.,abs(fd),abs(ad)))
    out={
        'scope':'CPU one bounded ellipse intersected with one fixed box; exact arc-area formula plus analytic arc-moment conic gradient. Not a kinetic or multi-object renderer and not a production robust-predicate implementation.',
        'seed':612431,'ellipse_box_cases':80,'conic_directional_checks':240,
        'max_absolute_area_error_against_independent_adaptive_1D_integration':max(area_errors),
        'max_normalized_conic_VJP_error_against_finite_difference':max(vjp_errors),
        'max_absolute_homogeneous_scale_null_gradient':max(gauge_errors),
        'max_normalized_camera_quadric_coverage_VJP_error':max(berrors),
        'passed':max(area_errors)<1e-9 and max(vjp_errors)<1e-6 and max(berrors)<1e-6,
    }
    if not out['passed']:
        raise AssertionError(json.dumps(out,indent=2))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,default=Path('ellipse_checks.json')); args=p.parse_args()
    result=run_checks(); args.output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()

"""Small CPU algebra/adjoint checks for the accompanying manuscript.

This is NOT a complete renderer, visibility compiler, GPU benchmark, or
validation of silhouette-gradient quadrature. Requires Python 3.10+ and NumPy.
Run: python checks.py --output checks.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

RNG = np.random.default_rng(71293)


def rotation() -> np.ndarray:
    q, _ = np.linalg.qr(RNG.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def rigid() -> np.ndarray:
    g = np.eye(4)
    g[:3, :3] = rotation()
    g[:3, 3] = RNG.normal(size=3)
    return g


def finite_difference(fun, x: np.ndarray, step: float = 1e-6) -> np.ndarray:
    ans = np.empty_like(x, dtype=float)
    for idx in np.ndindex(x.shape):
        xp, xm = x.copy(), x.copy()
        xp[idx] += step
        xm[idx] -= step
        ans[idx] = (fun(xp) - fun(xm)) / (2 * step)
    return ans


def relerr(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(1.0, np.linalg.norm(a), np.linalg.norm(b)))


def run_checks() -> dict:
    q0 = np.diag([1.0, 1.0, 1.0, -1.0])
    gauge_errors, congruence_errors, camera_errors = [], [], []
    for _ in range(24):
        h = np.eye(4)
        h[:3, :3] = rotation() @ np.diag(RNG.uniform(0.6, 1.7, 3))
        h[:3, 3] = RNG.normal(size=3)
        g, gauge = rigid(), rigid()
        w = np.linalg.solve(h, g)
        wprime = np.linalg.solve(np.linalg.solve(gauge, h), np.linalg.solve(gauge, g))
        gauge_errors.append(relerr(w, wprime))
        z = RNG.normal(size=(4, 4)); sb = (z + z.T) / 2
        bw = 2 * q0 @ w @ sb
        gh = -np.linalg.solve(h.T, bw) @ w.T
        gg = np.linalg.solve(h.T, bw)
        def objective_h(top):
            hx = h.copy(); hx[:3, :] = top
            wx = np.linalg.solve(hx, g)
            return float(np.sum(sb * (wx.T @ q0 @ wx)))
        def objective_g(top):
            gx = g.copy(); gx[:3, :] = top
            wx = np.linalg.solve(h, gx)
            return float(np.sum(sb * (wx.T @ q0 @ wx)))
        congruence_errors.append(relerr(gh[:3], finite_difference(objective_h, h[:3])))
        camera_errors.append(relerr(gg[:3], finite_difference(objective_g, g[:3])))

    root_errors = []
    for _ in range(80):
        a = RNG.uniform(0.5, 2.0)
        lo = RNG.uniform(0.2, 2.0); hi = lo + RNG.uniform(0.5, 2.0)
        abc = np.array([a, -a * (lo + hi) / 2, a * lo * hi])
        def near_root(x):
            aa, bb, cc = x
            dd = bb * bb - aa * cc
            if aa <= 0 or dd <= 0:
                raise ValueError('Finite difference left the regular root domain')
            return (-bb - np.sqrt(dd)) / aa
        s = near_root(abc)
        exact = -np.array([s * s, 2 * s, 1.0]) / (2 * (abc[0] * s + abc[1]))
        root_errors.append(relerr(exact, finite_difference(near_root, abc)))

    w0, w1 = np.eye(4), np.zeros((4, 4))
    w0[:3] += RNG.normal(scale=0.2, size=(3, 4))
    w1[:3] = RNG.normal(scale=0.15, size=(3, 4))
    ts = np.linspace(0.0, 1.0, 65)
    s = RNG.normal(size=(len(ts), 4, 4))
    s = (s + s.transpose(0, 2, 1)) / (2 * len(ts))
    moments = [np.einsum('t,tij->ij', ts**k, s) for k in range(3)]
    direct0 = sum(2 * q0 @ (w0 + t * w1) @ st for t, st in zip(ts, s))
    direct1 = sum(t * 2 * q0 @ (w0 + t * w1) @ st for t, st in zip(ts, s))
    shared0 = 2 * q0 @ (w0 @ moments[0] + w1 @ moments[1])
    shared1 = 2 * q0 @ (w0 @ moments[1] + w1 @ moments[2])
    def obj_w(flat):
        aa, bb = flat[:16].reshape(4, 4), flat[16:].reshape(4, 4)
        return sum(float(np.sum(st * ((aa + t * bb).T @ q0 @ (aa + t * bb))))
                   for t, st in zip(ts, s))
    combined = np.concatenate([w0.ravel(), w1.ravel()])
    analytic = np.concatenate([shared0.ravel(), shared1.ravel()])
    shared_fd_error = relerr(analytic, finite_difference(obj_w, combined))

    triangle_errors = []
    for _ in range(30):
        m = np.eye(3) + RNG.normal(scale=0.12, size=(3, 3))
        rhs = RNG.normal(size=3); upstream = RNG.normal(size=3)
        sol = np.linalg.solve(m, rhs)
        rhsbar = np.linalg.solve(m.T, upstream)
        mbar = -np.outer(rhsbar, sol)
        mm = finite_difference(lambda v: float(upstream @ np.linalg.solve(v, rhs)), m)
        bb = finite_difference(lambda v: float(upstream @ np.linalg.solve(m, v)), rhs)
        triangle_errors.extend([relerr(mbar, mm), relerr(rhsbar, bb)])

    # Exact box-filtered image of x <= a(t) in a unit square.
    coeff = np.array([0.2, 0.5]); upstream = RNG.normal(size=len(ts)) / len(ts)
    foreground, background = 0.9, 0.1
    def area_loss(c):
        area = np.clip(c[0] + c[1] * ts, 0, 1)
        return float(upstream @ (background + (foreground - background) * area))
    area_gradient = (foreground - background) * np.array([sum(upstream), upstream @ ts])
    area_error = relerr(area_gradient, finite_difference(area_loss, coeff))

    # Optical transfer composition, with RGB emission and scalar transmission.
    transfer_errors = []
    for _ in range(30):
        a1, a2 = RNG.random(3), RNG.random(3)
        b1, b2 = RNG.uniform(0.05, 0.95, 2)
        ga, gb = RNG.normal(size=3), float(RNG.normal())
        x = np.r_[a1, b1, a2, b2]
        def transfer_obj(v):
            return float(ga @ (v[:3] + v[3] * v[4:7]) + gb * v[3] * v[7])
        grad = np.r_[ga, ga @ a2 + gb * b2, b1 * ga, b1 * gb]
        transfer_errors.append(relerr(grad, finite_difference(transfer_obj, x)))

    opacity = 0.99
    red, blue = np.array([1., 0., 0.]), np.array([0., 0., 1.])
    rb = opacity * red + (1 - opacity) * opacity * blue
    br = opacity * blue + (1 - opacity) * opacity * red
    result = {
        'scope': 'CPU float64 algebra and local VJP checks only; no GPU timings, training, kinetic visibility implementation, or conic-boundary quadrature.',
        'seed': 71293,
        'max_relative_errors': {
            'coupled_SE3_gauge_invariance_24_cases': max(gauge_errors),
            'quadric_object_matrix_VJP_24_cases': max(congruence_errors),
            'quadric_camera_matrix_VJP_24_cases': max(camera_errors),
            'quadric_regular_root_VJP_80_cases': max(root_errors),
            'shared_vs_direct_coefficient_VJP_65_times': max(relerr(direct0, shared0), relerr(direct1, shared1)),
            'shared_coefficient_VJP_vs_finite_difference': shared_fd_error,
            'triangle_linear_solve_VJP_30_cases': max(triangle_errors),
            'box_filtered_moving_edge_VJP': area_error,
            'optical_transfer_composition_VJP_30_cases': max(transfer_errors),
        },
        'occlusion_counterexample': {
            'opacity_each_layer': opacity,
            'red_in_front_RGB': rb.tolist(),
            'blue_in_front_RGB': br.tolist(),
            'L_infinity_image_difference': float(np.max(np.abs(rb - br))),
        },
        'criteria': 'Every reported normalized error must be below 2e-7.',
    }
    result['passed'] = all(v < 2e-7 for v in result['max_relative_errors'].values())
    if not result['passed']:
        raise AssertionError(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('checks.json'))
    args = parser.parse_args()
    result = run_checks()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

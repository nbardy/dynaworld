#!/usr/bin/env python3
"""CPU check for the geometry-induced exponential/temporal-moment operator.

One point ray through translating, constant-density, overlapping slabs.
The event order is fixed over t in [-0.5, 0.5]. No GPU timing is performed.
Run: python temporal_operator.py --out temporal_results.json
Requires Python 3.10+ and NumPy.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np


def make_exponential_scene(n: int = 24) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(44821)
    centers = .16 * np.arange(n) + rng.uniform(-.02, .02, n)
    widths = rng.uniform(.31, .48, n)
    velocity = rng.uniform(-.08, .08, n)
    density = rng.uniform(.15, 1.0, n)
    color = rng.uniform(.05, .95, (n, 3))
    bg = np.array([.1, .2, .3])
    events = sorted([(centers[i] + sign * widths[i], i, -sign)
                     for i in range(n) for sign in [-1, 1]])
    positions = np.array([e[0] for e in events])
    owners = np.array([e[1] for e in events])
    gap = np.diff(positions)
    slope = np.diff(velocity[owners])
    # Keep each interval strictly positive on the entire physical time interval.
    ratio = .8 * gap / np.maximum(.5 * np.abs(slope), 1e-30)
    velocity *= min(1.0, float(np.min(ratio)))
    slope = np.diff(velocity[owners])
    assert np.min(gap - .5 * np.abs(slope)) > 0
    sigma = 0.0
    emitted = np.zeros(3)
    sigmas, colors = [], []
    for _, owner, sign in events[:-1]:
        sigma += sign * density[owner]
        emitted += sign * density[owner] * color[owner]
        sigmas.append(sigma)
        colors.append(emitted / sigma if sigma > 1e-12 else np.zeros(3))
    sigmas = np.asarray(sigmas)
    colors = np.asarray(colors)
    alpha = np.r_[0., np.cumsum(sigmas * gap)]
    beta = np.r_[0., np.cumsum(sigmas * slope)]
    gamma = np.vstack([colors[0], np.diff(colors, axis=0), bg - colors[-1]])
    w = gamma * np.exp(-alpha)[:, None]
    metadata = {'slabs': n, 'ray_intervals': len(gap), 'exponential_terms': len(beta),
                'time_interval': [-.5, .5], 'minimum_event_gap': float(np.min(gap-.5*np.abs(slope))),
                'max_abs_beta_times_halfwidth': float(np.max(np.abs(beta))*.5)}
    return w, beta, metadata


def compile_coefficients(w: np.ndarray, beta: np.ndarray, degree: int):
    """Return coefficients and their basis, without any T-dependent scene state."""
    if degree < 0:
        raise ValueError('degree must be nonnegative')
    basis = np.empty((len(beta), degree+1), dtype=np.float64)
    basis[:, 0] = 1.0
    for j in range(1, degree+1):
        basis[:, j] = basis[:, j-1] * (-beta) / j
    return basis.T @ w, basis


def reverse_coefficients(w: np.ndarray, basis: np.ndarray, adjoint: np.ndarray):
    gw = basis @ adjoint
    gb = np.zeros(len(w))
    for j in range(1, basis.shape[1]):
        gb -= basis[:, j-1] * (w @ adjoint[j])
    return gw, gb


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out', default='temporal_results.json')
    args=parser.parse_args()
    w,beta,metadata=make_exponential_scene()
    times=np.linspace(-.5,.5,2048)
    exact_basis=np.exp(-times[:,None]*beta[None,:])
    exact=exact_basis@w
    g=np.random.default_rng(881).normal(size=exact.shape)
    exact_gw=exact_basis.T@g
    exact_gb=np.sum((-times[:,None]*exact_basis)*(g@w.T),axis=0)
    ref_grad=np.r_[exact_gw.ravel(),exact_gb]
    x=metadata['max_abs_beta_times_halfwidth']
    rows=[]
    for p in [1,2,3,4]:
        coef,basis=compile_coefficients(w,beta,p)
        powers=times[:,None]**np.arange(p+1)[None,:]
        pred=powers@coef
        moment_adjoint=powers.T@g
        gw,gb=reverse_coefficients(w,basis,moment_adjoint)
        grad=np.r_[gw.ravel(),gb]
        approx_basis=powers@basis.T
        # Derivative with respect to beta is -h times the degree-(p-1) series.
        db_approx=-times[:,None]*(powers[:,:p]@basis[:,:p].T)
        db_exact=-times[:,None]*exact_basis
        dw_error=approx_basis-exact_basis
        row_error=np.sqrt((dw_error**2).sum(1)[:,None]+((db_approx-db_exact)**2)@(w*w))
        ep=np.exp(x)*x**(p+1)/math.factorial(p+1)
        epm=np.exp(x)*x**p/math.factorial(p)
        value_bound=float(np.max(np.sum(np.abs(w),axis=0))*ep)
        gradient_bound=float(len(beta)*ep+np.max(np.sum(np.abs(w),axis=0))*.5*epm)
        row={'degree':p,'coefficients_per_color':p+1,
             'max_image_error':float(np.max(np.abs(pred-exact))),
             'max_row_jacobian_l2_error':float(row_error.max()),
             'random_vjp_relative_l2_error':float(np.linalg.norm(grad-ref_grad)/np.linalg.norm(ref_grad)),
             'analytic_truncation_image_bound_excluding_roundoff':value_bound,
             'analytic_truncation_jacobian_bound_excluding_roundoff':gradient_bound}
        assert row['max_image_error'] <= value_bound+1e-12
        assert row['max_row_jacobian_l2_error'] <= gradient_bound+1e-12
        rows.append(row)
    # Check the coefficient reverse directly, separately from its approximation error.
    p=3
    coefficients,basis=compile_coefficients(w,beta,p)
    adjoint=np.random.default_rng(555).normal(size=coefficients.shape)
    gw,gb=reverse_coefficients(w,basis,adjoint)
    packed=np.r_[w.ravel(),beta]
    def objective(z):
        ww=z[:w.size].reshape(w.shape); bb=z[w.size:]
        return float(np.sum(compile_coefficients(ww,bb,p)[0]*adjoint))
    fd=np.zeros_like(packed); step=1e-6
    for j in range(len(packed)):
        delta=np.zeros_like(packed);delta[j]=step
        fd[j]=(objective(packed+delta)-objective(packed-delta))/(2*step)
    ana=np.r_[gw.ravel(),gb]
    fd_relative=float(np.linalg.norm(fd-ana)/np.linalg.norm(fd))
    assert fd_relative<1e-7
    result={'scope':'CPU float64 mathematical test; one ray; no reconstruction training or GPU timing',
            'construction':'Translating constant-width, constant-density overlapping slabs; fixed event topology',
            'scene':metadata,'time_queries':len(times),'approximations':rows,
            'coefficient_reverse_fd_parameters':len(packed),
            'coefficient_reverse_fd_relative_l2_error':fd_relative}
    Path(args.out).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()

# Gauged UVT cell-trace evaluator

Worker A lane: nonlinear/projective atlas-cell Metal evaluator.

What changed:

- Added a packed cell-trace atlas path in
  `third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/projective_trace.py`.
- The new atlas lowers accepted projective trace windows into direct raw-time
  polynomial rows for `(u(t), v(t), depth(t))`, with cell tile records indexing
  those trace rows. This makes the gauge-domain cell itself GPU-evaluable
  instead of only using the original rational primitive coeffs after binning.
- Added a native Metal forward op, `star_uvt_v0.render_projective_trace_cell_tiles`,
  with Python wrapper `render_projective_trace_cell_atlas_metal`.
- Added CPU reference coverage and an MPS/Metal parity test in
  `tests/test_star_uvt_projective_correctness.py`.

Why this matters:

- The previous projective tile renderer was useful but slightly indirect: it
  used the atlas for candidate/support intervals, then evaluated the original
  rational trace in the shader. The new path stores the accepted cell's local
  evaluator as the rendered object.
- This is a small concrete step toward the "event-certified gauge domains"
  formulation: accepted windows now carry both support/time validity and the
  local trace evaluator needed by Metal.

Verification:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_projective_correctness.py -q
# 15 passed in 6.72s

PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_trace.py \
  tests/test_star_uvt_projective_orbit_windows.py \
  tests/test_star_uvt_projective_visibility.py \
  tests/test_star_uvt_projective_binning.py \
  tests/test_star_uvt_projective_correctness.py -q
# 41 passed in 4.20s
```

The STAR UVT extension was rebuilt with:

```bash
( cd third_party/fast-mac-gsplat/variants/star_uvt_v0
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

No trainer configs or trainer tests were touched.

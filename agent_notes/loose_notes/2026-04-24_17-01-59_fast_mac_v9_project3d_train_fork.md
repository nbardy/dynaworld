# Fast-Mac v9 Project3D Train Fork

User asked to make the staged "mega training variant" rather than a literal
single mega-kernel. Forked the project3d path into:

```text
third_party/fast-mac-gsplat/variants/v9_project3d_train/
```

The fork is namespaced as:

```text
torch_gsplat_bridge_v9_project3d_train
gsplat_metal_v9_project3d_train
```

The v9 path keeps the good fusion boundary:

```text
Metal project forward
Metal raster forward
Metal raster backward -> dmeans2d/dconics/dopacity/dcolors
Metal project backward VJP -> dmeans3d/dscales/dquats/dopacities/dcamera
```

The projection backward is one thread per splat. Camera/intrinsic gradients are
atomically accumulated per batch; per-splat means/scales/quats/opacity gradients
are direct writes.

Extended `src/benchmarks/fast_mac_project3d_benchmark.py` so v9 is the default
Metal project3d candidate. v8 can still be included with `--include-v8`; v9 can
be built with `--build-v9`.

Smoke command:

```bash
.venv/bin/python src/benchmarks/fast_mac_project3d_benchmark.py --build-v9 --cases smoke:64:512:1 --warmup 2 --iters 3
```

Smoke passed. Full gradient max diff was `7.2e-05`, mean `4.5e-07`.

Large benchmark:

```bash
.venv/bin/python src/benchmarks/fast_mac_project3d_benchmark.py --cases realistic_128_8192:128:8192:1,large_256_65536:256:65536:1 --warmup 5 --iters 10
```

Results:

```text
realistic_128_8192,128,8192,1,forward_eval,v5_torch_project_plus_metal,6.5925,4.5022,8.6471,0,0,0,0
realistic_128_8192,128,8192,1,forward_backward,v5_torch_project_plus_metal,29.9302,20.6695,41.8315,0,0,0,0
realistic_128_8192,128,8192,1,forward_eval,v9_metal_project_train,7.5353,6.0767,9.7171,1.78814e-07,1.35432e-08,0.03125,1.05488e-06
realistic_128_8192,128,8192,1,forward_backward,v9_metal_project_train,15.6167,12.4170,24.2232,1.78814e-07,1.35432e-08,0.03125,1.05488e-06
large_256_65536,256,65536,1,forward_eval,v5_torch_project_plus_metal,16.6285,15.8189,17.1882,0,0,0,0
large_256_65536,256,65536,1,forward_backward,v5_torch_project_plus_metal,39.2649,37.6327,41.4533,0,0,0,0
large_256_65536,256,65536,1,forward_eval,v9_metal_project_train,5.9230,4.8212,6.8655,2.22921e-05,2.23182e-08,nan,nan
large_256_65536,256,65536,1,forward_backward,v9_metal_project_train,17.3225,16.9479,17.9475,2.22921e-05,2.23182e-08,nan,nan
```

The `0.03125` first-case grad max diff is dominated by accumulated
`camera_to_world` atomics. A gradient breakdown showed per-splat parameter
max diffs were small (`means3d 4.8e-05`, `scales 1.8e-04`, `quats 2.3e-05`,
`opacities 7.6e-06`, `colors 2.9e-06`), while `camera_to_world` max diff was
`0.0234` against `ref_abs_max 20281`, i.e. relative max around `3.2e-05`.

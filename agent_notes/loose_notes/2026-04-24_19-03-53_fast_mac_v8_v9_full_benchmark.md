# fast-mac v8/v9 full project3d benchmark

## Why

The user asked to run the full benchmark and accuracy sweep for the new Metal
project3d shaders against the existing baseline, including forward and backward
timing.

This pass covers:

- `v5_torch_project_plus_metal`: baseline Torch `3D->2D` projection plus Metal raster.
- `v8_metal_project_plus_metal`: v8 fork with Metal pinhole projection forward/backward.
- `v9_metal_project_train`: clean named training fork with Metal projection forward/backward.

## Harness change

`src/benchmarks/fast_mac_project3d_benchmark.py` now has:

```text
--grad-check {first,all,none}
```

The old behavior is preserved: by default it checks only the first case unless
`--skip-grad-check` is set. For this run, `--grad-check all` forced full
gradient parity on every case.

## Build and first full run

Command:

```text
.venv/bin/python src/benchmarks/fast_mac_project3d_benchmark.py --build-v8 --build-v9 --include-v8 --grad-check all --cases smoke:64:512:1,realistic_128_8192:128:8192:1,large_256_65536:256:65536:1,batch4_128_8192:128:8192:4 --warmup 5 --iters 10
```

Both v8 and v9 extensions built successfully and all cases ran.

Key result from the build run:

```text
case,size,gaussians,batch,phase,path,mean_ms,min_ms,max_ms,max_abs_err,mean_abs_err,grad_max_err,grad_mean_err
smoke,64,512,1,forward_backward,v5_torch_project_plus_metal,12.7703,10.5975,16.2204,0,0,0,0
smoke,64,512,1,forward_backward,v8_metal_project_plus_metal,4.9649,4.4146,5.4518,5.96046e-08,2.07535e-09,0.00012207,4.61595e-07
smoke,64,512,1,forward_backward,v9_metal_project_train,5.2366,4.9122,5.4575,5.96046e-08,2.07535e-09,0.00012207,4.64051e-07
realistic_128_8192,128,8192,1,forward_backward,v5_torch_project_plus_metal,16.7534,10.6647,25.4110,0,0,0,0
realistic_128_8192,128,8192,1,forward_backward,v8_metal_project_plus_metal,6.5143,6.0322,7.0867,2.38419e-07,1.31115e-08,0.0341797,1.19141e-06
realistic_128_8192,128,8192,1,forward_backward,v9_metal_project_train,6.4062,6.0262,7.1252,2.38419e-07,1.31115e-08,0.0283203,9.9885e-07
large_256_65536,256,65536,1,forward_backward,v5_torch_project_plus_metal,36.5543,34.0882,42.8272,0,0,0,0
large_256_65536,256,65536,1,forward_backward,v8_metal_project_plus_metal,19.4646,14.9141,23.6357,3.57628e-07,2.20992e-08,0.617188,1.5817e-06
large_256_65536,256,65536,1,forward_backward,v9_metal_project_train,14.5974,14.2898,15.1313,3.57628e-07,2.20992e-08,0.671875,1.72765e-06
batch4_128_8192,128,8192,4,forward_backward,v5_torch_project_plus_metal,27.0350,22.4308,32.6694,0,0,0,0
batch4_128_8192,128,8192,4,forward_backward,v8_metal_project_plus_metal,12.3787,8.7029,15.7022,2.38419e-07,1.34401e-08,0.0410156,1.08292e-06
batch4_128_8192,128,8192,4,forward_backward,v9_metal_project_train,11.0317,8.3119,16.0968,2.38419e-07,1.34401e-08,0.0429688,1.07541e-06
```

## Cleaner timing run

Command:

```text
.venv/bin/python src/benchmarks/fast_mac_project3d_benchmark.py --include-v8 --grad-check all --cases smoke:64:512:1,realistic_128_8192:128:8192:1,large_256_65536:256:65536:1,batch4_128_8192:128:8192:4 --warmup 10 --iters 30
```

Full output:

```text
case,size,gaussians,batch,phase,path,mean_ms,min_ms,max_ms,max_abs_err,mean_abs_err,grad_max_err,grad_mean_err
smoke,64,512,1,forward_eval,v5_torch_project_plus_metal,6.4904,5.7007,8.3846,0,0,0,0
smoke,64,512,1,forward_backward,v5_torch_project_plus_metal,16.8314,12.9865,28.7931,0,0,0,0
smoke,64,512,1,forward_eval,v8_metal_project_plus_metal,4.6548,3.8691,6.3252,5.96046e-08,2.07535e-09,6.10352e-05,4.53678e-07
smoke,64,512,1,forward_backward,v8_metal_project_plus_metal,7.2765,6.1861,9.6145,5.96046e-08,2.07535e-09,6.10352e-05,4.53678e-07
smoke,64,512,1,forward_eval,v9_metal_project_train,4.2804,3.8044,6.3067,5.96046e-08,2.07535e-09,0.000244141,4.81077e-07
smoke,64,512,1,forward_backward,v9_metal_project_train,9.0430,5.6182,29.9120,5.96046e-08,2.07535e-09,0.000244141,4.81077e-07
realistic_128_8192,128,8192,1,forward_eval,v5_torch_project_plus_metal,8.6255,7.6237,10.4486,0,0,0,0
realistic_128_8192,128,8192,1,forward_backward,v5_torch_project_plus_metal,29.8058,22.8154,39.4106,0,0,0,0
realistic_128_8192,128,8192,1,forward_eval,v8_metal_project_plus_metal,6.6860,4.6120,12.7720,2.38419e-07,1.31115e-08,0.0234375,1.04028e-06
realistic_128_8192,128,8192,1,forward_backward,v8_metal_project_plus_metal,8.4614,5.8391,14.5568,2.38419e-07,1.31115e-08,0.0234375,1.04028e-06
realistic_128_8192,128,8192,1,forward_eval,v9_metal_project_train,3.3968,3.1318,3.9781,2.38419e-07,1.31115e-08,0.0136719,7.90755e-07
realistic_128_8192,128,8192,1,forward_backward,v9_metal_project_train,10.3274,6.0456,15.1237,2.38419e-07,1.31115e-08,0.0136719,7.90755e-07
large_256_65536,256,65536,1,forward_eval,v5_torch_project_plus_metal,15.5199,9.5601,24.1841,0,0,0,0
large_256_65536,256,65536,1,forward_backward,v5_torch_project_plus_metal,32.5041,29.9606,36.7035,0,0,0,0
large_256_65536,256,65536,1,forward_eval,v8_metal_project_plus_metal,7.1780,5.1828,8.7025,3.57628e-07,2.20992e-08,0.640625,1.64751e-06
large_256_65536,256,65536,1,forward_backward,v8_metal_project_plus_metal,17.7714,14.4373,33.9121,3.57628e-07,2.20992e-08,0.640625,1.64751e-06
large_256_65536,256,65536,1,forward_eval,v9_metal_project_train,5.1241,4.9157,5.7613,3.57628e-07,2.20992e-08,0.632812,1.60312e-06
large_256_65536,256,65536,1,forward_backward,v9_metal_project_train,18.6746,14.3293,28.3222,3.57628e-07,2.20992e-08,0.632812,1.60312e-06
batch4_128_8192,128,8192,4,forward_eval,v5_torch_project_plus_metal,6.2490,5.9605,6.7688,0,0,0,0
batch4_128_8192,128,8192,4,forward_backward,v5_torch_project_plus_metal,31.7570,26.4180,39.2096,0,0,0,0
batch4_128_8192,128,8192,4,forward_eval,v8_metal_project_plus_metal,5.4113,4.6396,8.9044,2.38419e-07,1.34401e-08,0.0390625,1.08664e-06
batch4_128_8192,128,8192,4,forward_backward,v8_metal_project_plus_metal,8.9052,8.4973,9.6900,2.38419e-07,1.34401e-08,0.0390625,1.08664e-06
batch4_128_8192,128,8192,4,forward_eval,v9_metal_project_train,3.9961,3.6505,4.5929,2.38419e-07,1.34401e-08,0.0351562,9.21861e-07
batch4_128_8192,128,8192,4,forward_backward,v9_metal_project_train,17.1654,8.2010,34.1916,2.38419e-07,1.34401e-08,0.0351562,9.21861e-07
```

The 30-iteration run had visible MPS timing spikes in some train rows, especially
v9 batch-4. Rerunning the original large and batch-4 scenes for 50 iterations
showed the rows remain correct and the relative ordering is noisy rather than a
parity failure:

```text
large_256_65536 seed 1236:
v5 forward_backward mean 34.4080 ms
v8 forward_backward mean 15.7261 ms
v9 forward_backward mean 21.7249 ms

batch4_128_8192 seed 1237:
v5 forward_backward mean 29.8848 ms
v8 forward_backward mean 14.6485 ms
v9 forward_backward mean 14.0216 ms
```

## Gradient-error interpretation

The absolute gradient max on the large case looks high because the camera-pose
gradient is huge and accumulated through atomics. A parameter breakdown on
`large_256_65536` showed the per-splat gradients stay near fp32 noise, while
`camera_to_world` dominates:

```text
variant,v8
param,max_abs,mean_abs,ref_abs_max,rel_to_ref_max
means3d,9.15527e-05,3.11333e-07,384.466,2.38129e-07
scales,0.000549316,1.10946e-06,1288.14,4.2644e-07
quats,7.62939e-05,1.04116e-07,311.937,2.44581e-07
opacities,3.05176e-05,1.16574e-07,206.443,1.47826e-07
colors,1.14441e-05,6.03625e-08,65.9313,1.73576e-07
camera_to_world,0.539062,0.0679961,112498,4.79174e-06
fx,0.00088501,0.00088501,267.443,3.30916e-06
fy,0.000518799,0.000518799,275.179,1.88532e-06
cx,1.37091e-05,1.37091e-05,0.324184,4.22879e-05
cy,1.29938e-05,1.29938e-05,1.90552,6.81904e-06

variant,v9
param,max_abs,mean_abs,ref_abs_max,rel_to_ref_max
means3d,9.15527e-05,3.09704e-07,384.466,2.38129e-07
scales,0.000671387,1.10886e-06,1288.14,5.21205e-07
quats,6.86646e-05,1.0417e-07,311.937,2.20123e-07
opacities,3.05176e-05,1.17474e-07,206.443,1.47826e-07
colors,1.14441e-05,6.04477e-08,65.9313,1.73576e-07
camera_to_world,0.554688,0.0687276,112498,4.93064e-06
fx,0.000793457,0.000793457,267.443,2.96683e-06
fy,0.000762939,0.000762939,275.179,2.77252e-06
cx,3.3766e-05,3.3766e-05,0.324184,0.000104157
cy,2.81334e-05,2.81334e-05,1.90552,1.47642e-05
```

## Takeaway

The new Metal project3d paths are training-ready for this pinhole benchmark:
forward images match the v5 baseline at fp32-scale error and gradients match to
small relative error. The speed win is real versus v5, especially
forward+backward. v8 and v9 are close enough that we should keep benchmarking
them side by side while iterating; v9 is the cleaner named fork, but not
uniformly faster in every noisy MPS mean row.

## Distilled follow-up lessons

- The current dense/fast-mac path is projected-2D splat based, not ray-first.
  Camera math must happen before raster unless we explicitly move projection
  into the renderer.
- Moving pinhole `3D->2D` projection and its VJP into Metal is already a real
  win. It does not require a literal mega-kernel to be useful.
- The one-mega-kernel idea remains an ablation, not a conclusion. It may reduce
  intermediate memory traffic, but it can also increase register pressure and
  make backward harder to debug.
- v9 should remain the clean training fork, but v8/v9 should keep running
  side-by-side until timing statistics include median/p95 or the fused-kernel
  experiment clearly settles the choice.
- For the broader camera-model work, fisheye/common video lenses still need
  explicit projection math. A rays API is right for differentiable rendering in
  general, but this fast-mac raster path needs projected splat parameters unless
  we build a ray-aware renderer.

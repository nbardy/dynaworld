# Fast-Mac v8 Project3D Benchmark Results

Ran the v8 project3d benchmark on an MPS-available shell after the earlier
runtime became available.

Command:

```bash
.venv/bin/python src/benchmarks/fast_mac_project3d_benchmark.py --cases realistic_128_8192:128:8192:1,large_256_65536:256:65536:1 --warmup 5 --iters 10
```

Results:

```text
case,size,gaussians,batch,phase,path,mean_ms,min_ms,max_ms,max_abs_err,mean_abs_err,grad_max_err,grad_mean_err
realistic_128_8192,128,8192,1,forward_eval,v5_torch_project_plus_metal,6.1276,4.8444,9.0623,1.78814e-07,1.35432e-08,0.000160217,5.63046e-07
realistic_128_8192,128,8192,1,forward_eval,v8_metal_project_plus_metal,6.6006,3.7900,11.3646,1.78814e-07,1.35432e-08,0.000160217,5.63046e-07
realistic_128_8192,128,8192,1,forward_backward,v5_torch_project_plus_metal,17.0315,14.3224,25.3354,1.78814e-07,1.35432e-08,0.000160217,5.63046e-07
realistic_128_8192,128,8192,1,forward_backward,v8_metal_project_plus_metal,15.3523,11.9346,17.2872,1.78814e-07,1.35432e-08,0.000160217,5.63046e-07
large_256_65536,256,65536,1,forward_eval,v5_torch_project_plus_metal,16.8110,16.1393,17.5025,2.22921e-05,2.23182e-08,nan,nan
large_256_65536,256,65536,1,forward_eval,v8_metal_project_plus_metal,9.2442,6.5385,10.5221,2.22921e-05,2.23182e-08,nan,nan
large_256_65536,256,65536,1,forward_backward,v5_torch_project_plus_metal,37.2574,36.2763,38.6002,2.22921e-05,2.23182e-08,nan,nan
large_256_65536,256,65536,1,forward_backward,v8_metal_project_plus_metal,33.9022,32.0763,35.4783,2.22921e-05,2.23182e-08,nan,nan
```

Interpretation:

- At 8k/128px, v8 forward eval was slightly slower by mean timing, but
  forward+backward was about 9.9% faster.
- At 65k/256px, v8 forward eval was about 45% faster and forward+backward was
  about 9% faster.
- Image parity was tight: 8k max abs `1.8e-7`, mean `1.4e-8`; 65k max abs
  `2.2e-5`, mean `2.2e-8`.
- Full gradient parity was checked on the first benchmark case: max grad diff
  `1.6e-4`, mean grad diff `5.6e-7`.

Also ran a smoke case:

```bash
.venv/bin/python src/benchmarks/fast_mac_project3d_benchmark.py --build-v8 --cases smoke:64:512:1 --warmup 2 --iters 3
```

Smoke showed v8 forward slightly faster but forward+backward slower at tiny
size, so the Metal projection win only clearly appears once projection work is
large enough.

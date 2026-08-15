# Bounded native SPD(4) World Tubes ablation

single-scene single-seed bounded convergence and physical-renderer ablation; not a publication-scale quality claim

| Row | Atoms | Parameters | PSNR | SSIM | LPIPS | L1 | Train wall (s) | Peak driver (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy_peak | 256 | 3584 | 5.9865 | 0.02154 | 0.89915 | 0.45313 | 4.9020 | 63.357 |
| full_spd4_peak_parameter_matched | 199 | 3582 | 7.0054 | 0.03438 | 0.84708 | 0.37022 | 4.7512 | 46.596 |
| full_spd4_beer_fiber_parameter_matched | 199 | 3582 | 7.1333 | 0.03239 | 0.84321 | 0.36105 | 4.6758 | 46.596 |

## Claim boundary

Report hashes identify the accepted raw artifacts. The execution predated the durable source commit and therefore is not clean-source publication evidence.

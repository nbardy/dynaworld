# WorldFoam Paper-B verified foundation rows

Only independently accepted local/foundation evidence appears below. This table is not native-memory or public-quality evidence.

| Row | Category | Metric 1 | Value | Metric 2 | Value | Verdict | Scope |
| --- | --- | --- | ---: | --- | ---: | --- | --- |
| m0_m5_cpu_segment_parity | local_material_correctness | integral_max_abs_error | 5.95704e-15 | finite_difference_vjp_normalized_error | 6.85935e-10 | accepted_local_cpu | fixed segments only |
| m0_m5_metal_segment_parity | local_material_correctness | forward_normalized_error | 7.50772e-08 | vjp_normalized_error | 5.96046e-08 | accepted_historical_source_hash_checked_metal | fixed segments only; not current trainer runtime |
| partial_chord_positive_p2 | material_capacity | M3_heldout_loss | 5.25789e-17 | M5_heldout_loss | 8.79669e-05 | M3_family_win | synthetic partial chords; no universal winner |
| partial_chord_convex_log_p2 | material_capacity | M3_heldout_loss | 0.0013282 | M5_heldout_loss | 6.1884e-15 | M5_family_win | synthetic partial chords; no universal winner |
| adaptive_m3_m5_mean_loss | material_basis_selection | adaptive_to_best_fixed_ratio | 0.313405 | adaptive_to_oracle_ratio | 1 | accepted_cpu_adaptive_selection | verified float64 CPU synthetic per-cell basis selection on disjoint chords; matched 24-byte M3/M5 payload plus one tag bit; no native, public-image, runtime, or memory claim |
| adaptive_m3_m5_selection_accuracy | material_basis_selection | pure_family_selection_accuracy | 1 | selection_oracle_agreement | 1 | accepted_cpu_adaptive_selection | verified float64 CPU synthetic per-cell basis selection on disjoint chords; matched 24-byte M3/M5 payload plus one tag bit; no native, public-image, runtime, or memory claim |
| constant_density_ordered_transfer | ordered_transfer_algebra | render_max_abs_error | 0 | vjp_max_abs_error | 2.45576e-11 | accepted_cpu_constant_density | constant-density owner word only |
| g0_analytic_constant_sphere | synthetic_correctness | rgb_max_absolute_error | 0.000435542 | transmittance_max_absolute_error | 0.000505269 | accepted_cpu_g0 | verified float64 CPU synthetic S1-S8/C1-C7 only; no native runtime, allocator, trained-image, or public-data claim |
| g0_physical_gauge_jacobian | synthetic_correctness | with_jacobian_rgb_max_error | 3.32998e-07 | without_over_with_error_ratio | 916927 | accepted_cpu_g0 | verified float64 CPU synthetic S1-S8/C1-C7 only; no native runtime, allocator, trained-image, or public-data claim |
| g3_depth_layer_128_accuracy | synthetic_visibility | psnr_db_mean | 61.0018 | psnr_db_p05 | 37.9252 | accepted_cpu_g3 | verified float64 CPU synthetic S1-S8/C1-C7 only; no native runtime, allocator, trained-image, or public-data claim |
| g3_crossing_vs_representative_sort | synthetic_visibility | depth_layer_128_rgb_mse_mean | 1.41497e-05 | representative_sorted_rgb_mse_mean | 0.00116378 | accepted_cpu_g3 | verified float64 CPU synthetic S1-S8/C1-C7 only; no native runtime, allocator, trained-image, or public-data claim |
| g3_crossing_vs_depth_marginal | synthetic_visibility | depth_layer_128_rgb_mse_mean | 1.41497e-05 | depth_marginal_rgb_mse_mean | 0.00748454 | accepted_cpu_g3 | verified float64 CPU synthetic S1-S8/C1-C7 only; no native runtime, allocator, trained-image, or public-data claim |
| g3_crossing_flicker_vs_representative_sort | synthetic_visibility | depth_layer_128_flicker_mean | 0.00248642 | representative_sorted_flicker_mean | 0.0220636 | accepted_cpu_g3 | verified float64 CPU synthetic S1-S8/C1-C7 only; no native runtime, allocator, trained-image, or public-data claim |
| g3_crossing_gradient_variance_vs_representative_sort | synthetic_visibility | depth_layer_128_gradient_variance_mean | 2.77036e-05 | representative_sorted_gradient_variance_mean | 0.00210548 | accepted_cpu_g3 | verified float64 CPU synthetic S1-S8/C1-C7 only; no native runtime, allocator, trained-image, or public-data claim |
| g3_adaptive_fallback | synthetic_visibility | fallback_fraction_mean | 0.514625 | fallback_fraction_p95 | 0.952489 | accepted_cpu_g3_diagnostic | verified float64 CPU synthetic S1-S8/C1-C7 only; no native runtime, allocator, trained-image, or public-data claim |
| g3_crossing_order_flips | synthetic_visibility | representative_sorted_order_flips | 20 | depth_layer_128_order_flips | 20 | accepted_cpu_g3_diagnostic | verified float64 CPU synthetic S1-S8/C1-C7 only; no native runtime, allocator, trained-image, or public-data claim |

G3 visibility stress: **ACCEPTED — CPU SYNTHETIC ONLY** (S1-S8/C1-C7; not native runtime or public-data quality).

Adaptive M3/M5 basis selection: **ACCEPTED — CPU SYNTHETIC ONLY** (not native material promotion).

G4 public heldout quality: **NOT MEASURED**.

G6 native training memory: **NOT MEASURED**.

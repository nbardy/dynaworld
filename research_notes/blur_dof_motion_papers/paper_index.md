# Blur / DoF / Motion Paper Corpus Index

Date: 2026-04-24

This is the local searchable corpus for blur, depth of field, motion blur, rolling shutter, event-assisted deblurring, and adjacent camera-model papers for NeRF / 3D Gaussian Splatting work.

## Corpus Layout

- `pdfs/` stores downloaded PDFs.
- `text/` stores `pdftotext -layout` extraction output.
- `metadata.json` stores source URLs, local paths, tags, and download status.

Current counts:

- Papers indexed: 55
- PDFs present: 55
- Text extracts present: 55

Search examples:

```bash
rg -a -n "circle of confusion|CoC|aperture|focus distance" research_notes/blur_dof_motion_papers/text
rg -a -n "exposure|shutter|trajectory|Bezier|ODE|rolling shutter" research_notes/blur_dof_motion_papers/text
rg -a -n "event|spike|IMU|blur" research_notes/blur_dof_motion_papers/text
```

## Reading Priority

- `A`: read first for renderer/camera-token design.
- `B`: read when choosing implementation variants or ablations.
- `C`: adjacent context; useful if the data modality or pipeline changes.

## High-Signal Design Buckets

- Finite-aperture DoF: `DoF-NeRF`, `DOF-GS`, `DoF-Gaussian`, `Depth-Consistent 3DGS`, and dynamic defocused-video papers.
- Camera motion blur: `BAD-NeRF`, `ExBluRF`, `Deblur-GS`, `BAD-Gaussians`, `DeblurGS`, `CoMoGaussian`, `GeMS`.
- Dynamic object blur: `DyBluRF`, `BARD-GS`, `MoBGS`, `Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos`.
- Side-channel sensors: event/spike/IMU papers are mostly initialization or trajectory-supervision levers, not replacements for a sharp world model.
- Anti-cheat guardrail: methods that inflate covariance or learn per-pixel kernels are useful approximations but must be tested against exact aperture/shutter sampling.

## Foundational NeRF Blur And DoF

| Priority | Year | Paper | Local PDF | Text | Focus |
|---|---:|---|---|---|---|
| A | 2021 | [Deblur-NeRF: Neural Radiance Fields from Blurry Images](https://arxiv.org/abs/2111.14292) | [pdf](pdfs/2111_14292_deblur_nerf_neural_radiance_fields_from_blurry_images.pdf) | [txt](text/2111_14292_deblur_nerf_neural_radiance_fields_from_blurry_images.txt) | learned blur-kernel rays for blurry-image NeRF supervision |
| A | 2022 | [NeRFocus: Neural Radiance Field for 3D Synthetic Defocus](https://arxiv.org/abs/2203.05189) | [pdf](pdfs/2203_05189_nerfocus_neural_radiance_field_for_3d_synthetic_defocu.pdf) | [txt](text/2203_05189_nerfocus_neural_radiance_field_for_3d_synthetic_defocu.txt) | thin-lens NeRF for synthetic defocus and focus/aperture control |
| A | 2022 | [DoF-NeRF: Depth-of-Field Meets Neural Radiance Fields](https://arxiv.org/abs/2208.00945) | [pdf](pdfs/2208_00945_dof_nerf_depth_of_field_meets_neural_radiance_fields.pdf) | [txt](text/2208_00945_dof_nerf_depth_of_field_meets_neural_radiance_fields.txt) | finite-aperture DoF rendering extension for NeRF |
| A | 2022 | [DP-NeRF: Deblurred Neural Radiance Field with Physical Scene Priors](https://arxiv.org/abs/2211.12046) | [pdf](pdfs/2211_12046_dp_nerf_deblurred_neural_radiance_field_with_physical.pdf) | [txt](text/2211_12046_dp_nerf_deblurred_neural_radiance_field_with_physical.txt) | physical scene priors and rigid blur kernels for deblurred NeRF |
| A | 2022 | [BAD-NeRF: Bundle Adjusted Deblur Neural Radiance Fields](https://arxiv.org/abs/2211.12853) | [pdf](pdfs/2211_12853_bad_nerf_bundle_adjusted_deblur_neural_radiance_fields.pdf) | [txt](text/2211_12853_bad_nerf_bundle_adjusted_deblur_neural_radiance_fields.txt) | bundle-adjusted camera trajectory during exposure |
| A | 2023 | [ExBluRF: Efficient Radiance Fields for Extreme Motion Blurred Images](https://arxiv.org/abs/2309.08957) | [pdf](pdfs/2309_08957_exblurf_efficient_radiance_fields_for_extreme_motion_b.pdf) | [txt](text/2309_08957_exblurf_efficient_radiance_fields_for_extreme_motion_b.txt) | efficient extreme camera-motion blur radiance fields |
| B | 2022 | [PDRF: Progressively Deblurring Radiance Field for Fast and Robust Scene Reconstruction from Blurry Images](https://arxiv.org/abs/2208.08049) | [pdf](pdfs/2208_08049_pdrf_progressively_deblurring_radiance_field_for_fast.pdf) | [txt](text/2208_08049_pdrf_progressively_deblurring_radiance_field_for_fast.txt) | progressive deblurring radiance field optimization |
| B | 2023 | [Hybrid Neural Rendering for Large-Scale Scenes with Motion Blur](https://arxiv.org/abs/2304.12652) | [pdf](pdfs/2304_12652_hybrid_neural_rendering_for_large_scale_scenes_with_mo.pdf) | [txt](text/2304_12652_hybrid_neural_rendering_for_large_scale_scenes_with_mo.txt) | large-scale hybrid neural rendering with blur-aware weighting |
| B | 2024 | [Sharp-NeRF: Grid-based Fast Deblurring Neural Radiance Fields Using Sharpness Prior](https://arxiv.org/abs/2401.00825) | [pdf](pdfs/2401_00825_sharp_nerf_grid_based_fast_deblurring_neural_radiance.pdf) | [txt](text/2401_00825_sharp_nerf_grid_based_fast_deblurring_neural_radiance.txt) | sharpness-prior grid deblurring for fast radiance fields |

## Dynamic / Rolling / Sparse NeRF Blur

| Priority | Year | Paper | Local PDF | Text | Focus |
|---|---:|---|---|---|---|
| A | 2023 | [MoBluRF: Motion Deblurring Neural Radiance Fields for Blurry Monocular Video](https://arxiv.org/abs/2312.13528) | [pdf](pdfs/2312_13528_moblurf_motion_deblurring_neural_radiance_fields_for_b.pdf) | [txt](text/2312_13528_moblurf_motion_deblurring_neural_radiance_fields_for_b.txt) | dynamic deblurring NeRF / MoBluRF lineage from blurry monocular video |
| A | 2024 | [DyBluRF: Dynamic Neural Radiance Fields from Blurry Monocular Video](https://arxiv.org/abs/2403.10103) | [pdf](pdfs/2403_10103_dyblurf_dynamic_neural_radiance_fields_from_blurry_mon.pdf) | [txt](text/2403_10103_dyblurf_dynamic_neural_radiance_fields_from_blurry_mon.txt) | dynamic radiance fields with camera and object DCT trajectories |
| A | 2024 | [Dynamic Neural Radiance Field From Defocused Monocular Video](https://arxiv.org/abs/2407.05586) | [pdf](pdfs/2407_05586_dynamic_neural_radiance_field_from_defocused_monocular.pdf) | [txt](text/2407_05586_dynamic_neural_radiance_field_from_defocused_monocular.txt) | dynamic radiance fields from defocused monocular video |
| B | 2024 | [SMURF: Continuous Dynamics for Motion-Deblurring Radiance Fields](https://arxiv.org/abs/2403.07547) | [pdf](pdfs/2403_07547_smurf_continuous_dynamics_for_motion_deblurring_radian.pdf) | [txt](text/2403_07547_smurf_continuous_dynamics_for_motion_deblurring_radian.txt) | continuous Neural-ODE camera dynamics for motion-deblurring radiance fields |
| B | 2024 | [URS-NeRF: Unordered Rolling Shutter Bundle Adjustment for Neural Radiance Fields](https://arxiv.org/abs/2403.10119) | [pdf](pdfs/2403_10119_urs_nerf_unordered_rolling_shutter_bundle_adjustment_f.pdf) | [txt](text/2403_10119_urs_nerf_unordered_rolling_shutter_bundle_adjustment_f.txt) | unordered rolling-shutter bundle adjustment for NeRF |
| B | 2024 | [Sparse-DeRF: Deblurred Neural Radiance Fields from Sparse View](https://arxiv.org/abs/2407.06613) | [pdf](pdfs/2407_06613_sparse_derf_deblurred_neural_radiance_fields_from_spar.pdf) | [txt](text/2407_06613_sparse_derf_deblurred_neural_radiance_fields_from_spar.txt) | sparse-view deblurred NeRF |
| B | 2024 | [RS-NeRF: Neural Radiance Fields from Rolling Shutter Images](https://arxiv.org/abs/2407.10267) | [pdf](pdfs/2407_10267_rs_nerf_neural_radiance_fields_from_rolling_shutter_im.pdf) | [txt](text/2407_10267_rs_nerf_neural_radiance_fields_from_rolling_shutter_im.txt) | NeRF from rolling-shutter images |
| B | 2025 | [Dual-Camera All-in-Focus Neural Radiance Fields](https://arxiv.org/abs/2504.16636) | [pdf](pdfs/2504_16636_dual_camera_all_in_focus_neural_radiance_fields.pdf) | [txt](text/2504_16636_dual_camera_all_in_focus_neural_radiance_fields.txt) | dual-camera all-in-focus radiance fields |

## 3DGS Motion Blur / Deblurring

| Priority | Year | Paper | Local PDF | Text | Focus |
|---|---:|---|---|---|---|
| A | 2024 | [Deblurring 3D Gaussian Splatting](https://arxiv.org/abs/2401.00834) | [pdf](pdfs/2401_00834_deblurring_3d_gaussian_splatting.pdf) | [txt](text/2401_00834_deblurring_3d_gaussian_splatting.txt) | covariance-manipulating MLP for deblurring 3DGS |
| A | 2024 | [BAGS: Blur Agnostic Gaussian Splatting through Multi-Scale Kernel Modeling](https://arxiv.org/abs/2403.04926) | [pdf](pdfs/2403_04926_bags_blur_agnostic_gaussian_splatting_through_multi_sc.pdf) | [txt](text/2403_04926_bags_blur_agnostic_gaussian_splatting_through_multi_sc.txt) | blur-agnostic per-pixel kernels and blur masks for 3DGS |
| A | 2024 | [BAD-Gaussians: Bundle Adjusted Deblur Gaussian Splatting](https://arxiv.org/abs/2403.11831) | [pdf](pdfs/2403_11831_bad_gaussians_bundle_adjusted_deblur_gaussian_splattin.pdf) | [txt](text/2403_11831_bad_gaussians_bundle_adjusted_deblur_gaussian_splattin.txt) | bundle-adjusted motion-deblur Gaussian Splatting |
| A | 2024 | [DeblurGS: Gaussian Splatting for Camera Motion Blur](https://arxiv.org/abs/2404.11358) | [pdf](pdfs/2404_11358_deblurgs_gaussian_splatting_for_camera_motion_blur.pdf) | [txt](text/2404_11358_deblurgs_gaussian_splatting_for_camera_motion_blur.txt) | 6DoF camera motion and densification annealing for DeblurGS |
| A | 2024 | [Deblur-GS: 3D Gaussian Splatting from Camera Motion Blurred Images](https://github.com/Chaphlagical/Deblur-GS) | [pdf](pdfs/deblur_gs_i3d_2024_deblur_gs_3d_gaussian_splatting_from_camera_motion_blu.pdf) | [txt](text/deblur_gs_i3d_2024_deblur_gs_3d_gaussian_splatting_from_camera_motion_blu.txt) | camera trajectory/time-sampling Deblur-GS author version |
| A | 2025 | [CoMoGaussian: Continuous Motion-Aware Gaussian Splatting from Motion-Blurred Images](https://arxiv.org/abs/2503.05332) | [pdf](pdfs/2503_05332_comogaussian_continuous_motion_aware_gaussian_splattin.pdf) | [txt](text/2503_05332_comogaussian_continuous_motion_aware_gaussian_splattin.txt) | continuous ODE camera trajectories for motion-blurred 3DGS |
| A | 2025 | [GeMS: Efficient Gaussian Splatting for Extreme Motion Blur](https://arxiv.org/abs/2508.14682) | [pdf](pdfs/2508_14682_gems_efficient_gaussian_splatting_for_extreme_motion_b.pdf) | [txt](text/2508_14682_gems_efficient_gaussian_splatting_for_extreme_motion_b.txt) | extreme motion blur with VGGSfM init, 3DGS-MCMC, Bezier camera trajectories |
| B | 2025 | [DeblurSplat: SfM-free 3D Gaussian Splatting with Event Camera for Robust Deblurring](https://arxiv.org/abs/2509.18898) | [pdf](pdfs/2509_18898_deblursplat_sfm_free_3d_gaussian_splatting_with_event.pdf) | [txt](text/2509_18898_deblursplat_sfm_free_3d_gaussian_splatting_with_event.txt) | SfM-free event-camera Gaussian Splatting deblurring |
| B | 2025 | [BSGS: Bi-stage 3D Gaussian Splatting for Camera Motion Deblurring](https://arxiv.org/abs/2510.12493) | [pdf](pdfs/2510_12493_bsgs_bi_stage_3d_gaussian_splatting_for_camera_motion.pdf) | [txt](text/2510_12493_bsgs_bi_stage_3d_gaussian_splatting_for_camera_motion.txt) | bi-stage 3DGS for camera motion deblurring |
| B | 2025 | [Breaking the Vicious Cycle: Coherent 3D Gaussian Splatting from Sparse and Motion-Blurred Views](https://arxiv.org/abs/2512.10369) | [pdf](pdfs/2512_10369_breaking_the_vicious_cycle_coherent_3d_gaussian_splatt.pdf) | [txt](text/2512_10369_breaking_the_vicious_cycle_coherent_3d_gaussian_splatt.txt) | coherent 3DGS from sparse and motion-blurred views |

## 3DGS Defocus / Depth Of Field

| Priority | Year | Paper | Local PDF | Text | Focus |
|---|---:|---|---|---|---|
| A | 2024 | [DOF-GS:Adjustable Depth-of-Field 3D Gaussian Splatting for Post-Capture Refocusing, Defocus Rendering and Blur Removal](https://arxiv.org/abs/2405.17351) | [pdf](pdfs/2405_17351_dof_gs_adjustable_depth_of_field_3d_gaussian_splatting.pdf) | [txt](text/2405_17351_dof_gs_adjustable_depth_of_field_3d_gaussian_splatting.txt) | finite-aperture DOF-GS with CoC maps and refocus controls |
| A | 2025 | [DoF-Gaussian: Controllable Depth-of-Field for 3D Gaussian Splatting](https://arxiv.org/abs/2503.00746) | [pdf](pdfs/2503_00746_dof_gaussian_controllable_depth_of_field_for_3d_gaussi.pdf) | [txt](text/2503_00746_dof_gaussian_controllable_depth_of_field_for_3d_gaussi.txt) | DoF-Gaussian lens model, depth priors, defocus-to-focus adaptation |
| B | 2025 | [Depth-Consistent 3D Gaussian Splatting via Physical Defocus Modeling and Multi-View Geometric Supervision](https://arxiv.org/abs/2511.10316) | [pdf](pdfs/2511_10316_depth_consistent_3d_gaussian_splatting_via_physical_de.pdf) | [txt](text/2511_10316_depth_consistent_3d_gaussian_splatting_via_physical_de.txt) | physical defocus modeling as depth-consistency supervision for 3DGS |

## Dynamic 3DGS Blur

| Priority | Year | Paper | Local PDF | Text | Focus |
|---|---:|---|---|---|---|
| A | 2025 | [BARD-GS: Blur-Aware Reconstruction of Dynamic Scenes via Gaussian Splatting](https://arxiv.org/abs/2503.15835) | [pdf](pdfs/2503_15835_bard_gs_blur_aware_reconstruction_of_dynamic_scenes_vi.pdf) | [txt](text/2503_15835_bard_gs_blur_aware_reconstruction_of_dynamic_scenes_vi.txt) | dynamic 3DGS decomposition of camera and object motion blur |
| A | 2025 | [MoBGS: Motion Deblurring Dynamic 3D Gaussian Splatting for Blurry Monocular Video](https://arxiv.org/abs/2504.15122) | [pdf](pdfs/2504_15122_mobgs_motion_deblurring_dynamic_3d_gaussian_splatting.pdf) | [txt](text/2504_15122_mobgs_motion_deblurring_dynamic_3d_gaussian_splatting.txt) | dynamic blurry monocular video deblurring with latent camera ODE |
| A | 2025 | [Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos](https://arxiv.org/abs/2510.10691) | [pdf](pdfs/2510_10691_dynamic_gaussian_splatting_from_defocused_and_motion_b.pdf) | [txt](text/2510_10691_dynamic_gaussian_splatting_from_defocused_and_motion_b.txt) | dynamic Gaussian Splatting from defocused and motion-blurred monocular videos |
| B | 2025 | [UPGS: Unified Pose-aware Gaussian Splatting for Dynamic Scene Deblurring](https://arxiv.org/abs/2509.00831) | [pdf](pdfs/2509_00831_upgs_unified_pose_aware_gaussian_splatting_for_dynamic.pdf) | [txt](text/2509_00831_upgs_unified_pose_aware_gaussian_splatting_for_dynamic.txt) | pose-aware dynamic scene deblurring via Gaussian Splatting |
| B | 2025 | [PEGS: Physics-Event Enhanced Large Spatiotemporal Motion Reconstruction via 3D Gaussian Splatting](https://arxiv.org/abs/2511.17116) | [pdf](pdfs/2511_17116_pegs_physics_event_enhanced_large_spatiotemporal_motio.pdf) | [txt](text/2511_17116_pegs_physics_event_enhanced_large_spatiotemporal_motio.txt) | physics-event enhanced large spatiotemporal motion reconstruction |

## Event / Spike Assisted Blur

| Priority | Year | Paper | Local PDF | Text | Focus |
|---|---:|---|---|---|---|
| A | 2024 | [EvaGaussians: Event Stream Assisted Gaussian Splatting from Blurry Images](https://arxiv.org/abs/2405.20224) | [pdf](pdfs/2405_20224_evagaussians_event_stream_assisted_gaussian_splatting.pdf) | [txt](text/2405_20224_evagaussians_event_stream_assisted_gaussian_splatting.txt) | event stream assisted 3DGS from blurry images with exposure-time motion |
| A | 2024 | [EaDeblur-GS: Event assisted 3D Deblur Reconstruction with Gaussian Splatting](https://arxiv.org/abs/2407.13520) | [pdf](pdfs/2407_13520_eadeblur_gs_event_assisted_3d_deblur_reconstruction_wi.pdf) | [txt](text/2407_13520_eadeblur_gs_event_assisted_3d_deblur_reconstruction_wi.txt) | event-assisted 3D deblur reconstruction with Gaussian Splatting |
| A | 2024 | [E-3DGS: Gaussian Splatting with Exposure and Motion Events](https://arxiv.org/abs/2410.16995) | [pdf](pdfs/2410_16995_e_3dgs_gaussian_splatting_with_exposure_and_motion_eve.pdf) | [txt](text/2410_16995_e_3dgs_gaussian_splatting_with_exposure_and_motion_eve.txt) | 3DGS with exposure and motion events |
| B | 2024 | [Mitigating Motion Blur in Neural Radiance Fields with Events and Frames](https://arxiv.org/abs/2403.19780) | [pdf](pdfs/2403_19780_mitigating_motion_blur_in_neural_radiance_fields_with.pdf) | [txt](text/2403_19780_mitigating_motion_blur_in_neural_radiance_fields_with.txt) | events and frames to mitigate motion blur in NeRF |
| B | 2024 | [SpikeNVS: Enhancing Novel View Synthesis from Blurry Images via Spike Camera](https://arxiv.org/abs/2404.06710) | [pdf](pdfs/2404_06710_spikenvs_enhancing_novel_view_synthesis_from_blurry_im.pdf) | [txt](text/2404_06710_spikenvs_enhancing_novel_view_synthesis_from_blurry_im.txt) | spike camera for NVS from blurry images |
| B | 2024 | [Deblurring Neural Radiance Fields with Event-driven Bundle Adjustment](https://arxiv.org/abs/2406.14360) | [pdf](pdfs/2406_14360_deblurring_neural_radiance_fields_with_event_driven_bu.pdf) | [txt](text/2406_14360_deblurring_neural_radiance_fields_with_event_driven_bu.txt) | event-driven bundle adjustment for deblurring NeRF |
| B | 2024 | [E2GS: Event Enhanced Gaussian Splatting](https://arxiv.org/abs/2406.14978) | [pdf](pdfs/2406_14978_e2gs_event_enhanced_gaussian_splatting.pdf) | [txt](text/2406_14978_e2gs_event_enhanced_gaussian_splatting.txt) | event-enhanced Gaussian Splatting |
| B | 2024 | [E$^{3}$NeRF: Efficient Event-Enhanced Neural Radiance Fields from Blurry Images](https://arxiv.org/abs/2408.01840) | [pdf](pdfs/2408_01840_e_3_nerf_efficient_event_enhanced_neural_radiance_fiel.pdf) | [txt](text/2408_01840_e_3_nerf_efficient_event_enhanced_neural_radiance_fiel.txt) | efficient event-enhanced NeRF from blurry images |
| B | 2024 | [LSE-NeRF: Learning Sensor Modeling Errors for Deblured Neural Radiance Fields with RGB-Event Stereo](https://arxiv.org/abs/2409.06104) | [pdf](pdfs/2409_06104_lse_nerf_learning_sensor_modeling_errors_for_deblured.pdf) | [txt](text/2409_06104_lse_nerf_learning_sensor_modeling_errors_for_deblured.txt) | RGB-event stereo sensor modeling errors for deblurred NeRF |
| B | 2024 | [Deblur e-NeRF: NeRF from Motion-Blurred Events under High-speed or Low-light Conditions](https://arxiv.org/abs/2409.17988) | [pdf](pdfs/2409_17988_deblur_e_nerf_nerf_from_motion_blurred_events_under_hi.pdf) | [txt](text/2409_17988_deblur_e_nerf_nerf_from_motion_blurred_events_under_hi.txt) | NeRF from motion-blurred events in high-speed/low-light conditions |
| B | 2024 | [BeSplat: Gaussian Splatting from a Single Blurry Image and Event Stream](https://arxiv.org/abs/2412.19370) | [pdf](pdfs/2412_19370_besplat_gaussian_splatting_from_a_single_blurry_image.pdf) | [txt](text/2412_19370_besplat_gaussian_splatting_from_a_single_blurry_image.txt) | single blurry image plus event stream to Gaussian Splatting |
| B | 2026 | [Seeing through Light and Darkness: Sensor-Physics Grounded Deblurring HDR NeRF from Single-Exposure Images and Events](https://arxiv.org/abs/2601.15475) | [pdf](pdfs/2601_15475_seeing_through_light_and_darkness_sensor_physics_groun.pdf) | [txt](text/2601_15475_seeing_through_light_and_darkness_sensor_physics_groun.txt) | sensor-physics grounded HDR/deblur NeRF with events |
| B | 2026 | [Event-Aided Sharp Radiance Field Reconstruction for Fast-Flying Drones](https://arxiv.org/abs/2602.21101) | [pdf](pdfs/2602_21101_event_aided_sharp_radiance_field_reconstruction_for_fa.pdf) | [txt](text/2602_21101_event_aided_sharp_radiance_field_reconstruction_for_fa.txt) | event-aided sharp radiance fields for fast-flying drones |
| C | 2025 | [AE-NeRF: Augmenting Event-Based Neural Radiance Fields for Non-ideal Conditions and Larger Scene](https://arxiv.org/abs/2501.02807) | [pdf](pdfs/2501_02807_ae_nerf_augmenting_event_based_neural_radiance_fields.pdf) | [txt](text/2501_02807_ae_nerf_augmenting_event_based_neural_radiance_fields.txt) | event-based NeRF augmentation for non-ideal conditions |

## Camera Sampling / Anti-Aliasing Adjacent

| Priority | Year | Paper | Local PDF | Text | Focus |
|---|---:|---|---|---|---|
| B | 2024 | [Rip-NeRF: Anti-aliasing Radiance Fields with Ripmap-Encoded Platonic Solids](https://arxiv.org/abs/2405.02386) | [pdf](pdfs/2405_02386_rip_nerf_anti_aliasing_radiance_fields_with_ripmap_enc.pdf) | [txt](text/2405_02386_rip_nerf_anti_aliasing_radiance_fields_with_ripmap_enc.txt) | anti-aliasing radiance fields with ripmaps; camera footprint discipline |
| B | 2024 | [fNeRF: High Quality Radiance Fields from Practical Cameras](https://arxiv.org/abs/2406.10633) | [pdf](pdfs/2406_10633_fnerf_high_quality_radiance_fields_from_practical_came.pdf) | [txt](text/2406_10633_fnerf_high_quality_radiance_fields_from_practical_came.txt) | high-quality radiance fields from practical cameras |

## General Degradation-Robust Radiance Fields

| Priority | Year | Paper | Local PDF | Text | Focus |
|---|---:|---|---|---|---|
| B | 2025 | [Casual3DHDR: Deblurring High Dynamic Range 3D Gaussian Splatting from Casually Captured Videos](https://arxiv.org/abs/2504.17728) | [pdf](pdfs/2504_17728_casual3dhdr_deblurring_high_dynamic_range_3d_gaussian.pdf) | [txt](text/2504_17728_casual3dhdr_deblurring_high_dynamic_range_3d_gaussian.txt) | casual HDR and deblurring 3DGS from videos |
| C | 2024 | [Towards Degradation-Robust Reconstruction in Generalizable NeRF](https://arxiv.org/abs/2411.11691) | [pdf](pdfs/2411_11691_towards_degradation_robust_reconstruction_in_generaliz.pdf) | [txt](text/2411_11691_towards_degradation_robust_reconstruction_in_generaliz.txt) | generalizable degradation-robust NeRF reconstruction |
| C | 2025 | [Exploiting Deblurring Networks for Radiance Fields](https://arxiv.org/abs/2502.14454) | [pdf](pdfs/2502_14454_exploiting_deblurring_networks_for_radiance_fields.pdf) | [txt](text/2502_14454_exploiting_deblurring_networks_for_radiance_fields.txt) | using 2D deblurring networks for radiance fields |

## SLAM / Camera Pipeline Blur

| Priority | Year | Paper | Local PDF | Text | Focus |
|---|---:|---|---|---|---|
| C | 2026 | [TRGS-SLAM: IMU-Aided Gaussian Splatting SLAM for Blurry, Rolling Shutter, and Noisy Thermal Images](https://arxiv.org/abs/2603.20443) | [pdf](pdfs/2603_20443_trgs_slam_imu_aided_gaussian_splatting_slam_for_blurry.pdf) | [txt](text/2603_20443_trgs_slam_imu_aided_gaussian_splatting_slam_for_blurry.txt) | IMU-aided 3DGS SLAM for blurry, rolling-shutter, noisy thermal images |

## Implementation Extraction TODO

- Build a formula table: CoC equations, shutter integration equations, trajectory parameterizations, and loss terms.
- Build a dataset table: synthetic blur generation, real blur capture assumptions, event/spike/IMU availability, dynamic-scene support.
- Build an anti-cheat checklist per method: which parameters can explain away geometry, and which held-out queries expose the cheat.
- Compare exact aperture/shutter sampling against covariance/kernel shortcuts in one synthetic splat scene.

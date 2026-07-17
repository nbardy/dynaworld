# STAR UVT Checkpoint Ladder Regeneration

Context:
    The selected STAR support continuation (`K=8/r64/o0.4`, 50 steps from the
    sparse-forward 1500-step checkpoint) could not start because the local
    workspace no longer had the old `outputs/checkpoints/` artifacts. The
    preflight verifier made the initial missing inputs explicit.

Artifact search:
    Searched the dynaworld repo, parent `gsplats_browser` workspace, and W&B
    offline file cache for the required 1500-step sparse-forward checkpoint and
    RGB-probe/colorizer checkpoint. No matching `.pt` artifacts were present.

Regenerated RGB-probe/colorizer:
    Ran:
    `PYTHONPATH=src/train PYTHONUNBUFFERED=1 WANDB_MODE=offline .venv/bin/python src/train/train_star_uvt_feature_rgb_probe.py src/train_configs/star_uvt_feature_targetgrid_vjepa_rgb_probe_hidden64_lr01_1000step.jsonc`

    The run hit the local torchhub V-JEPA cache, baked the feature target cache,
    and completed successfully. Outputs:
    - `outputs/feature_cache/star_uvt_feature_targets/vjepa2_1_vitb_256crop_64f/a524619cf73c9cc18bdbe53d.pt`
    - `outputs/checkpoints/2026-05-19_star_uvt_feature_targetgrid_vjepa_rgb_probe_hidden64_lr01_1000step.pt`
    - `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_vjepa_rgb_probe_hidden64_lr01_1000step.json`
    - offline W&B `wandb/offline-run-20260525_221940-onsehts5`

    Result: grid loss `0.044358 -> 0.004494`, final full PSNR `20.089`,
    pass `true`.

Regenerated first STAR ladder checkpoint:
    Ran:
    `PYTHONPATH=src/train PYTHONUNBUFFERED=1 WANDB_MODE=offline .venv/bin/python src/train/train.py src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe10_lr005_300step_checkpoint.jsonc`

    This historical autograd target-grid segment is slow but viable. It took
    about 20 minutes locally. A process sample during the run showed it was
    active inside STAR UVT Metal/autograd backward, not stuck in W&B.

    Outputs:
    - `outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_300step.pt`
    - `outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_300step_checkpoint.json`
    - offline W&B `wandb/offline-run-20260525_222053-alkbeo34`

    Result: pass `true`, loss `1.458365 -> 1.057802`, feature loss
    `0.999935 -> 0.812539`, RGB-probe PSNR `13.387 -> 16.104`, zero tile
    overflow, mean step/backward/render `3975.9/1983.0/1438.0ms`, last
    step/backward/render `3357.4/1762.3/1182.8ms`.

Regenerated 300->600 ladder checkpoint:
    Ran:
    `PYTHONPATH=src/train PYTHONUNBUFFERED=1 WANDB_MODE=offline .venv/bin/python src/train/train.py src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe10_lr005_resume300_from300_checkpoint_media.jsonc`

    Outputs:
    - `outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_600step_after_resume.pt`
    - `outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_resume300_from300_checkpoint_media.json`
    - offline W&B `wandb/offline-run-20260525_224438-tkw6nq0o`

    Result: pass `true`, loss `1.056025 -> 0.752422`, feature loss
    `0.811725 -> 0.654100`, RGB-probe PSNR `16.121 -> 20.074`, zero tile
    overflow.

Regenerated 600->800 ladder checkpoint:
    Ran:
    `PYTHONPATH=src/train PYTHONUNBUFFERED=1 WANDB_MODE=offline .venv/bin/python src/train/train.py src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature025_lr005_resume200_from600_checkpoint_media.jsonc`

    Outputs:
    - `outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature025_lr005_800step_after_resume.pt`
    - `outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature025_lr005_resume200_from600_checkpoint_media.json`
    - offline W&B `wandb/offline-run-20260525_230058-nknk3w21`

    Result: pass `true`, total loss `0.556334 -> 0.403675`, RGB-probe PSNR
    `20.078 -> 22.458`, zero tile overflow. Feature loss increases
    `0.653852 -> 0.706235` under feature weight `0.25`; this segment should
    be cited as a total/RGB-probe win, not a feature-loss win.

Regenerated 800->1000 ladder checkpoint:
    Ran:
    `PYTHONPATH=src/train PYTHONUNBUFFERED=1 WANDB_MODE=offline .venv/bin/python src/train/train.py src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe_schedule_balance_resume200_from800_checkpoint_media.jsonc`

    Outputs:
    - `outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe_schedule_balance_1000step_after_resume.pt`
    - `outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe_schedule_balance_resume200_from800_checkpoint_media.json`
    - `outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe_schedule_balance_resume200_from800_contact.jpg`
    - `outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe_schedule_balance_resume200_from800_sbs.mp4`
    - offline W&B `wandb/offline-run-20260525_231916-bubca3vm`

    Result: pass `true`, loss `0.762971 -> 0.428924`, feature loss
    `0.706284 -> 0.637935`, RGB-probe PSNR `22.465 -> 22.598`, zero tile
    overflow, mean step/backward/render `3368.5/1836.8/1137.3ms`, last
    step/backward/render `4076.9/2044.2/1486.2ms`.

Regenerated 1000->1100 ladder checkpoint:
    Ran:
    `PYTHONPATH=src/train PYTHONUNBUFFERED=1 WANDB_MODE=offline .venv/bin/python src/train/train.py src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature05_lr005_resume100_from1000_checkpoint_media.jsonc`

    Outputs:
    - `outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature05_lr005_1100step_after_resume.pt`
    - `outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature05_lr005_resume100_from1000_checkpoint_media.json`
    - `outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature05_lr005_resume100_from1000_contact.jpg`
    - `outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature05_lr005_resume100_from1000_sbs.mp4`
    - offline W&B `wandb/offline-run-20260525_233303-pvv0mbwo`

    Result: pass `true`, total loss `0.538652 -> 0.503427`, RGB-probe PSNR
    `22.602 -> 23.537`, zero tile overflow, mean step/backward/render
    `3433.4/1927.0/1125.4ms`, last step/backward/render
    `2704.5/1555.8/909.4ms`. Feature loss increases
    `0.637887 -> 0.652565` under feature weight `0.5`; this segment should be
    cited as a total/RGB-probe win, not a feature-loss win.

Regenerated 1100->1200 ladder checkpoint:
    Ran:
    `PYTHONPATH=src/train PYTHONUNBUFFERED=1 WANDB_MODE=offline .venv/bin/python src/train/train.py src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe_schedule_recover_resume100_from1100_checkpoint_media.jsonc`

    Outputs:
    - `outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe_schedule_recover_1200step_after_resume.pt`
    - `outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe_schedule_recover_resume100_from1100_checkpoint_media.json`
    - `outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe_schedule_recover_resume100_from1100_contact.jpg`
    - `outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe_schedule_recover_resume100_from1100_sbs.mp4`
    - offline W&B `wandb/offline-run-20260525_234118-08458lgu`

    Result: pass `true`, loss `0.740994 -> 0.600747`, feature loss
    `0.652525 -> 0.624458`, RGB-probe PSNR `23.542 -> 23.552`, zero tile
    overflow, mean step/backward/render `3443.2/1934.2/1128.2ms`, last
    step/backward/render `2690.4/1591.8/896.2ms`.

Regenerated 1200->1250 ladder checkpoint:
    Ran:
    `PYTHONPATH=src/train PYTHONUNBUFFERED=1 WANDB_MODE=offline .venv/bin/python src/train/train.py src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature075_lr005_resume50_from1200_checkpoint_media.jsonc`

    Outputs:
    - `outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature075_lr005_1250step_after_resume.pt`
    - `outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature075_lr005_resume50_from1200_checkpoint_media.json`
    - `outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature075_lr005_resume50_from1200_contact.jpg`
    - `outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature075_lr005_resume50_from1200_sbs.mp4`
    - offline W&B `wandb/offline-run-20260525_235206-y0ml2jc9`

    Result: pass `true`, total loss `0.644657 -> 0.636518`, RGB-probe PSNR
    `23.557 -> 23.817`, zero tile overflow, mean step/backward/render
    `4591.2/2448.5/1450.6ms`, last step/backward/render
    `7399.6/3804.8/2174.7ms`. Feature loss increases
    `0.624403 -> 0.627228` under feature weight `0.75`; this segment should be
    cited as a total/RGB-probe win, not a feature-loss win. The timing tail is
    slower than the previous 100-step rows.

Regenerated 1250->1300 ladder checkpoint:
    Ran:
    `PYTHONPATH=src/train PYTHONUNBUFFERED=1 WANDB_MODE=offline .venv/bin/python src/train/train.py src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume50_from1250_checkpoint_media.jsonc`

    Outputs:
    - `outputs/checkpoints/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_1300step_after_resume.pt`
    - `outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume50_from1250_checkpoint_media.json`
    - `outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume50_from1250_contact.jpg`
    - `outputs/media/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume50_from1250_sbs.mp4`
    - offline W&B `wandb/offline-run-20260525_235827-fkjzpli1`

    Result: pass `true`, loss `0.793051 -> 0.775637`, feature loss
    `0.627185 -> 0.618493`, RGB-probe PSNR `23.823 -> 24.058`, zero tile
    overflow, mean step/backward/render `4806.3/2630.3/1474.5ms`, last
    step/backward/render `5770.3/2679.0/1284.4ms`.

Intermediate preflight after 1300:
    Reran
    `research_experiments/star_uvt_feature_tubes/preflight_support_birthsplit_gate.py`
    for the 50-step `K=8/r64/o0.4` support config. It is now blocked only on:
    `outputs/checkpoints/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_lr001_resume50_from1450_lr005sparse_1500step.pt`.

Remaining ladder at that point:
    Continue in order:
    1. `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_sparseforward_batchedvjp_checkpoint_media.jsonc`
       -> `...sparseforward_batchedvjp_64f512_1400step.pt`
    2. `...feature1_lr001_resume50_from1400_lr005sparse_checkpointselect.jsonc`
       -> `...lr001_resume50_from1400_lr005sparse_1450step.pt`
    3. `...feature1_lr001_resume50_from1450_lr005sparse_media.jsonc`
        -> `...lr001_resume50_from1450_lr005sparse_1500step.pt`

Decision at that point:
    Do not launch the 50-step support continuation until the remaining ladder
    is rebuilt and the preflight passes without `--allow-missing`. The first
    regenerated segment proves the path is viable but slow; expect the full
    ladder to take meaningful local wall time.

Regenerated 1300->1400 sparse-forward/batched-VJP checkpoint:
    Ran:
    `PYTHONPATH=src/train PYTHONUNBUFFERED=1 WANDB_MODE=offline .venv/bin/python src/train/train.py src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_sparseforward_batchedvjp_checkpoint_media.jsonc`

    Outputs:
    - `outputs/checkpoints/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_64f512_1400step.pt`
    - `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_64f512_from1300_100step_media.json`
    - `outputs/media/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_64f512_from1300_100step_contact.jpg`
    - `outputs/media/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_64f512_from1300_100step_sbs.mp4`
    - offline W&B `wandb/offline-run-20260526_000735-inu9e86f`

    Result: pass `true`, loss `0.775389 -> 0.757040`, feature loss
    `0.618394 -> 0.609855`, RGB-probe loss `0.003925 -> 0.003680`,
    RGB-probe PSNR `24.342`, zero overflow, mean step/backward/render
    `722.5/328.6/213.8ms`, last step/backward/render `1007.1/288.9/433.9ms`.

Regenerated 1400->1450 checkpoint:
    Ran:
    `PYTHONPATH=src/train PYTHONUNBUFFERED=1 WANDB_MODE=offline .venv/bin/python src/train/train.py src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr001_resume50_from1400_lr005sparse_checkpointselect.jsonc`

    Outputs:
    - `outputs/checkpoints/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_lr001_resume50_from1400_lr005sparse_1450step.pt`
    - `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_lr001_resume50_from1400_lr005sparse_checkpointselect.json`
    - offline W&B `wandb/offline-run-20260526_000928-bx97h173`

    Result: pass `true`, total loss `0.756800 -> 0.756539`, RGB-probe loss
    `0.003676 -> 0.003660`, RGB-probe PSNR `24.366`, zero overflow, mean
    step/backward/render `859.4/363.5/272.5ms`. Feature loss slightly worsens
    `0.609756 -> 0.610156`, so cite this as a total/RGB-probe maintenance row,
    not a feature-loss win.

Regenerated 1450->1500 checkpoint:
    Ran:
    `PYTHONPATH=src/train PYTHONUNBUFFERED=1 WANDB_MODE=offline .venv/bin/python src/train/train.py src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr001_resume50_from1450_lr005sparse_media.jsonc`

    Outputs:
    - `outputs/checkpoints/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_lr001_resume50_from1450_lr005sparse_1500step.pt`
    - `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_lr001_resume50_from1450_lr005sparse_media.json`
    - `outputs/media/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_lr001_resume50_from1450_lr005sparse_contact.jpg`
    - `outputs/media/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_lr001_resume50_from1450_lr005sparse_sbs.mp4`
    - offline W&B `wandb/offline-run-20260526_001039-hlo6xs7x`

    Result: pass `true`, loss `0.756490 -> 0.752234`, feature loss
    `0.610136 -> 0.608145`, RGB-probe loss `0.003659 -> 0.003602`,
    RGB-probe PSNR `24.434`, zero overflow, mean step/backward/render
    `983.7/381.2/311.8ms`, last step/backward/render `276.1/122.0/105.7ms`.

Clean preflight:
    Reran
    `research_experiments/star_uvt_feature_tubes/preflight_support_birthsplit_gate.py`
    without `--allow-missing`; it returned `status=ready` and wrote:
    - `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_r64_o04_50step_preflight/summary.md`
    - `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_r64_o04_50step_preflight/summary.json`

Ran 50-step `K=8/r64/o0.4` support continuation:
    Ran:
    `PYTHONPATH=src/train PYTHONUNBUFFERED=1 WANDB_MODE=offline .venv/bin/python src/train/train.py src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit32_multicenter_k8_r64_o04_from1500_lr001_50step_media.jsonc`

    Outputs were still written despite the final assertion:
    - `outputs/checkpoints/2026-05-25_star_uvt_feature_targetgrid_birthsplit32_multicenter_k8_r64_o04_from1500_lr001_50step.pt`
    - `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_multicenter_k8_r64_o04_50step_media.json`
    - `outputs/media/2026-05-25_star_uvt_birthsplit_multicenter_k8_r64_o04_50step_contact.png`
    - `outputs/media/2026-05-25_star_uvt_birthsplit_multicenter_k8_r64_o04_50step_rgb_probe_contact.png`
    - `outputs/media/2026-05-25_star_uvt_birthsplit_multicenter_k8_r64_o04_50step_rgb_probe_side_by_side.mp4`
    - `outputs/media/2026-05-25_star_uvt_birthsplit_multicenter_k8_r64_o04_50step_side_by_side.mp4`
    - offline W&B `wandb/offline-run-20260526_001216-5m7vbxzb`

    Result: `pass=false` because `train.require_no_tile_overflow` correctly
    trips. `tile_overflow_sum=277`, max tile count `146/128`, overflow excess
    refs `1233`, and fixed-bin eligibility is false. The support selection did
    reallocate all 32 tubes across 8 centers with center tube counts
    `[3, 1, 10, 1, 6, 8, 2, 1]`; selected opacity moved
    `0.3409 -> 0.4000`. Loss movement is positive despite the overflow:
    weighted loss `0.773832 -> 0.760400`, feature loss
    `0.612675 -> 0.611403`, RGB-probe loss `0.004029 -> 0.003725`,
    RGB-probe PSNR `23.948 -> 24.289`.

Updated decision:
    The selected `K=8/r64/o0.4` 50-step row is a useful failed gate, not a
    promotion. It shows the support mechanism can still improve the objective
    from the regenerated 1500 checkpoint, but 32 births at this support pressure
    violate the cap-128 fixed-bin contract. Next work should run a narrower
    cap-128-safe support-pressure follow-up, for example preserving K=8 while
    reducing born tubes/radius (`n16/r48/o0.4`), before spending STAR effort on
    Softmax-GS or switching the mainline to WorldFoam.

## 2026-05-26 Cap-128 Support-Pressure Follow-Up

Question:
    Could the `K=8/r64/o0.4` support mechanism keep its objective/probe gains
    while obeying the hard cap-128 fixed-bin renderer budget?

Ran `K=8/n16/r48/o0.4`:
    Config:
    `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit16_multicenter_k8_r48_o04_from1500_lr001_50step_media.jsonc`

    Outputs:
    - `outputs/checkpoints/2026-05-25_star_uvt_feature_targetgrid_birthsplit16_multicenter_k8_r48_o04_from1500_lr001_50step.pt`
    - `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_multicenter_k8_n16_r48_o04_50step_media.json`
    - offline W&B `wandb/offline-run-20260526_002046-62pd4ybf`

    Result: `pass=false`, but only barely on capacity: `tile_overflow_sum=2`,
    max tile `131/128`, overflow excess refs `4`. Loss/probe moved positively:
    loss `0.757862 -> 0.750863`, feature loss `0.609050 -> 0.608136`,
    RGB-probe loss `0.003720 -> 0.003568`, RGB-probe PSNR
    `24.294 -> 24.476`. Support selected 16 tubes over 8 centers with counts
    `[1, 1, 5, 1, 2, 4, 1, 1]`.

Ran `K=8/n16/r40/o0.4`:
    Config:
    `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit16_multicenter_k8_r40_o04_from1500_lr001_50step_media.jsonc`

    Outputs:
    - `outputs/checkpoints/2026-05-25_star_uvt_feature_targetgrid_birthsplit16_multicenter_k8_r40_o04_from1500_lr001_50step.pt`
    - `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_multicenter_k8_n16_r40_o04_50step_media.json`
    - offline W&B `wandb/offline-run-20260526_002303-gkf2jhde`

    Result: `pass=false` with the same tiny capacity failure:
    `tile_overflow_sum=2`, max tile `131/128`, overflow excess refs `4`.
    Loss/probe again moved positively: loss `0.756313 -> 0.750070`, feature
    loss `0.608773 -> 0.607858`, RGB-probe loss `0.003689 -> 0.003555`,
    RGB-probe PSNR `24.331 -> 24.491`. Radius `48 -> 40` did not clear the
    last overflow tiles, so born-count/placement is the active limiter here,
    not just projected support radius.

Ran `K=8/n8/r40/o0.4`:
    Config:
    `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit8_multicenter_k8_r40_o04_from1500_lr001_50step_media.jsonc`

    Outputs:
    - `outputs/checkpoints/2026-05-25_star_uvt_feature_targetgrid_birthsplit8_multicenter_k8_r40_o04_from1500_lr001_50step.pt`
    - `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_multicenter_k8_n8_r40_o04_50step_media.json`
    - `outputs/media/2026-05-25_star_uvt_birthsplit_multicenter_k8_n8_r40_o04_50step_contact.png`
    - `outputs/media/2026-05-25_star_uvt_birthsplit_multicenter_k8_n8_r40_o04_50step_rgb_probe_contact.png`
    - `outputs/media/2026-05-25_star_uvt_birthsplit_multicenter_k8_n8_r40_o04_50step_rgb_probe_side_by_side.mp4`
    - `outputs/media/2026-05-25_star_uvt_birthsplit_multicenter_k8_n8_r40_o04_50step_side_by_side.mp4`
    - offline W&B `wandb/offline-run-20260526_002520-xc3rv44y`

    Result: `pass=true`, `tile_overflow_sum=0`, max tile `123/128`,
    fixed-bin eligible. It improves loss `0.754568 -> 0.749460`, feature loss
    `0.608402 -> 0.607554`, RGB-probe loss `0.003654 -> 0.003548`, and
    RGB-probe PSNR `24.372 -> 24.501`. Support selected one born tube per
    center (`[1, 1, 1, 1, 1, 1, 1, 1]`) and opacity moved
    `0.328434 -> 0.400000`.

Dense support diagnostic:
    Ran
    `research_experiments/star_uvt_feature_tubes/dense_alpha_failure_diagnostic.py`
    across `start1500`, `r64_o04_n32`, `r48_o04_n16`, `r40_o04_n16`, and
    `r40_o04_n8`, writing:
    - `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_pressure_50step_dense_support.json`
    - `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_pressure_50step_dense_support.md`

    The cap-safe `n8/r40` row improves the dense diagnostic over `start1500`:
    normal PSNR `6.035 -> 6.472`, forced-alpha PSNR `10.702 -> 14.018`,
    oracle PSNR `16.787 -> 21.602`, alpha `>0.1` `0.6468 -> 0.6542`, and
    alpha `>0.5` `0.2301 -> 0.2365`. The invalid higher-pressure rows remain
    slightly stronger on normal/forced support, so the safe row is not a visual
    closeout. It is the current valid seed for a longer promotion gate or for a
    smarter visibility/support birth rule.

Updated decision:
    Keep Softmax-GS parked. Keep WorldFoam as challenger. For STAR, promote
    `K=8/n8/r40/o0.4` only as the cap-128-safe support seed. The next useful
    work is either a longer safe-row promotion gate with dense diagnostics or a
    smarter support-selection bridge. More radius/opacity sweeping is unlikely
    to clear the remaining quality gap by itself.

## 2026-05-26 Longer Cap-Safe Support Gate

Purpose:
    Test whether the cap-safe `K=8/n8/r40/o0.4` support seed has useful runway
    beyond the initial 50-step row.

Ran 100-step `K=8/n8/r40/o0.4` from the same 1500 checkpoint:
    Config:
    `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit8_multicenter_k8_r40_o04_from1500_lr001_100step_media.jsonc`

    Outputs:
    - `outputs/checkpoints/2026-05-25_star_uvt_feature_targetgrid_birthsplit8_multicenter_k8_r40_o04_from1500_lr001_100step.pt`
    - `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_multicenter_k8_n8_r40_o04_100step_media.json`
    - `outputs/media/2026-05-25_star_uvt_birthsplit_multicenter_k8_n8_r40_o04_100step_contact.png`
    - `outputs/media/2026-05-25_star_uvt_birthsplit_multicenter_k8_n8_r40_o04_100step_rgb_probe_contact.png`
    - offline W&B `wandb/offline-run-20260526_003920-iy24tfn5`

    Result: `pass=false` because `require_loss_decrease` catches a late
    objective regression. It remains fixed-bin safe (`tile_overflow_sum=0`, max
    tile `122/128`), but ends at loss `0.754568 -> 0.755682`, feature loss
    `0.608402 -> 0.610522`, RGB-probe PSNR `24.372 -> 24.402`, and dense RGB
    PSNR `6.450`. The per-step trace shows the real issue: it reaches a strong
    endpoint at global step `1589` (`0.747008` loss, `0.606764` feature loss,
    `0.003506` probe loss), then jumps at `1590` and again at `1594`.

Ran 90-step checkpoint-selection gate:
    Config:
    `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit8_multicenter_k8_r40_o04_from1500_lr001_90step_checkpointselect_media.jsonc`

    Outputs:
    - `outputs/checkpoints/2026-05-25_star_uvt_feature_targetgrid_birthsplit8_multicenter_k8_r40_o04_from1500_lr001_90step_checkpointselect.pt`
    - `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_multicenter_k8_n8_r40_o04_90step_checkpointselect_media.json`
    - `outputs/media/2026-05-25_star_uvt_birthsplit_multicenter_k8_n8_r40_o04_90step_checkpointselect_contact.png`
    - `outputs/media/2026-05-25_star_uvt_birthsplit_multicenter_k8_n8_r40_o04_90step_checkpointselect_rgb_probe_contact.png`
    - offline W&B `wandb/offline-run-20260526_004507-3821f8dh`

    Result: `pass=true`, zero overflow, max tile `122/128`, loss
    `0.754568 -> 0.747006`, feature loss `0.608402 -> 0.606764`, RGB-probe loss
    `0.003654 -> 0.003506`, RGB-probe PSNR `24.372 -> 24.552`, dense RGB PSNR
    `6.462`, mean step/backward/render `837.9/373.8/250.9ms`, and last
    step/backward/render `361.8/223.7/94.9ms`.

Dense support follow-up:
    Wrote:
    - `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_n8_r40_90_100step_dense_support.json`
    - `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_n8_r40_90_100step_dense_support.md`

    Dense support is nearly flat across the safe-row continuations:
    - 50-step: normal `6.472`, forced-alpha `14.018`, oracle `21.602`, alpha
      `>0.1` `0.6542`
    - 90-step: normal `6.462`, forced-alpha `14.012`, oracle `21.579`, alpha
      `>0.1` `0.6523`
    - 100-step: normal `6.450`, forced-alpha `14.054`, oracle `21.681`, alpha
      `>0.1` `0.6506`

Updated decision:
    The selected cap-safe checkpoint is now the 90-step `K=8/n8/r40/o0.4`
    row. It is a better objective/probe checkpoint than the 50-step row and
    stays inside cap-128, but it does not materially improve dense support.
    The 100-step row proves the safe support seed can overrun into late
    objective jumps. Next STAR work should change support selection/visibility
    or add checkpoint-aware scheduling; do not spend another broad
    radius/opacity sweep, and do not port Softmax-GS into STAR yet.

Ran checkpoint-aware 100-step LR-tail schedule:
    Config:
    `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit8_multicenter_k8_r40_o04_from1500_lr001_tail00025_100step_media.jsonc`

    Schedule:
    `lr=0.001` until global step `1588`, then `lr=0.00025` through the 100-step
    endpoint.

    Outputs:
    - `outputs/checkpoints/2026-05-25_star_uvt_feature_targetgrid_birthsplit8_multicenter_k8_r40_o04_from1500_lr001_tail00025_100step.pt`
    - `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_multicenter_k8_n8_r40_o04_tail00025_100step_media.json`
    - `outputs/media/2026-05-25_star_uvt_birthsplit_multicenter_k8_n8_r40_o04_tail00025_100step_contact.png`
    - `outputs/media/2026-05-25_star_uvt_birthsplit_multicenter_k8_n8_r40_o04_tail00025_100step_rgb_probe_contact.png`
    - offline W&B `wandb/offline-run-20260526_005456-omnvnem7`

    Result: `pass=true`, zero overflow, max tile `122/128`, loss
    `0.754568 -> 0.749454`, feature loss `0.608402 -> 0.608167`, RGB-probe
    loss `0.003654 -> 0.003532`, RGB-probe PSNR `24.372 -> 24.520`, dense RGB
    PSNR `6.462`, mean step/backward/render `935.2/410.5/289.5ms`, and last
    step/backward/render `307.7/161.0/95.8ms`.

Dense support follow-up with schedule row:
    Wrote:
    - `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_n8_r40_schedule_dense_support.json`
    - `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_n8_r40_schedule_dense_support.md`

    The schedule fixes the catastrophic 100-step constant-LR endpoint but does
    not produce a better checkpoint than the 90-step row. Its dense support is
    essentially identical to the selected 90-step checkpoint:
    - 90-step: normal `6.462`, forced-alpha `14.012`, oracle `21.579`, alpha
      `>0.1` `0.6523`
    - tail100: normal `6.462`, forced-alpha `14.012`, oracle `21.578`, alpha
      `>0.1` `0.6523`

Updated decision after schedule:
    Keep the 90-step `K=8/n8/r40/o0.4` checkpoint as the current cap-safe
    support checkpoint. Treat the LR-tail schedule as useful stability evidence,
    not a promotion. The next STAR task should change support selection,
    visibility conditioning, or the model handoff; schedule-only cleanup and
    broad radius/opacity sweeps are now low-value.

## 2026-05-26 Allocation Follow-Up: Uniform And K=12/N=12

Purpose:
    Test whether the previous cap wall was caused by proportional tube packing
    into a few centers, or whether the actual limiter is saturated tile support
    that survives more even birth placement.

Code/config change:
    Added opt-in `support_birth_split.tube_allocation` with allowed values
    `proportional` and `uniform`. The default remains `proportional`; `uniform`
    spreads reallocated tubes evenly across active centers, assigning any
    remainder to the largest point groups. Focused tests passed:
    `tests/test_star_uvt_feature_target_adapter.py` selected nodes plus
    `tests/test_star_uvt_support_birthsplit_preflight.py` reported
    `7 passed`.

Ran uniform `K=8/n16/r40/o0.4`:
    Config:
    `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit16_multicenter_k8_r40_o04_uniform_from1500_lr001_50step_media.jsonc`

    Result:
    `pass=false`, max tile `131/128`, two overflow tiles. Uniform allocation
    worked (`[2, 2, 2, 2, 2, 2, 2, 2]`) but did not clear the cap wall. Loss
    still moved positively: `0.758282 -> 0.751447`, feature
    `0.608735 -> 0.607839`, RGB-probe PSNR `24.273 -> 24.449`.

Ran one-tube-per-center `K=16/n16/r40/o0.4`:
    Config:
    `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit16_multicenter_k16_r40_o04_from1500_lr001_50step_media.jsonc`

    Result:
    `pass=false`, max tile `131/128`, two overflow tiles. Counts were
    `[1] * 16`, so the same cap failure survives full center spreading. Loss
    `0.755419 -> 0.749744`, feature `0.608930 -> 0.608018`, RGB-probe PSNR
    `24.363 -> 24.506`.

Ran `K=16/n16/r32/o0.4`:
    Config:
    `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit16_multicenter_k16_r32_o04_from1500_lr001_50step_media.jsonc`

    Result:
    `pass=false`, max tile `131/128`, two overflow tiles. Shrinking radius
    from `40` to `32` did not clear the same saturated tiles. Loss
    `0.754127 -> 0.749176`, feature `0.608651 -> 0.607738`, RGB-probe PSNR
    `24.393 -> 24.515`.

Ran cap-safe `K=12/n12/r40/o0.4`:
    50-step config:
    `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit12_multicenter_k12_r40_o04_from1500_lr001_50step_media.jsonc`

    50-step result:
    `pass=true`, max tile `127/128`, center counts `[1] * 12`, loss
    `0.753998 -> 0.749098`, feature `0.608633 -> 0.607729`, RGB-probe PSNR
    `24.396 -> 24.517`, dense RGB PSNR `6.483`.

    90-step config:
    `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit12_multicenter_k12_r40_o04_from1500_lr001_90step_checkpointselect_media.jsonc`

    90-step result:
    `pass=true`, max tile `126/128`, loss `0.753998 -> 0.749217`, feature
    `0.608633 -> 0.608311`, RGB-probe PSNR `24.396 -> 24.531`, dense RGB PSNR
    `6.474`.

Dense diagnostic:
    Wrote:
    - `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_k12_n12_dense_support.json`
    - `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_k12_n12_dense_support.md`

    Dense table:
    - `start1500`: normal `6.035`, forced-alpha `10.702`, oracle `16.787`,
      alpha `>0.1` `0.647`
    - `n8_r40_50`: normal `6.472`, forced-alpha `14.018`, oracle `21.602`,
      alpha `>0.1` `0.654`
    - `n8_r40_90`: normal `6.462`, forced-alpha `14.012`, oracle `21.579`,
      alpha `>0.1` `0.652`
    - `k12_n12_r40_50`: normal `6.483`, forced-alpha `14.019`, oracle
      `21.577`, alpha `>0.1` `0.655`
    - `k12_n12_r40_90`: normal `6.474`, forced-alpha `14.013`, oracle
      `21.551`, alpha `>0.1` `0.654`

Updated decision:
    The selected support checkpoint remains `K=8/n8/r40/o0.4` 90-step:
    it has the best objective/feature endpoint among the cap-safe 90-step rows.
    `K=12/n12` is a useful cap-safe pressure datapoint and a slightly better
    50-step normal-PSNR row, but it does not change the dense support diagnosis.
    The next STAR work should be tile-cap-aware and visibility/residual-aware
    support birth, or a model handoff that changes what the new support learns.

## 2026-05-26 Cap-Aware Support Birth And Guarded Tile Repair

Purpose:
    Test whether the last two overflowing tiles can be avoided by making target
    selection aware of tile slack, then by repairing only newly born tubes after
    support placement. This is meant to keep the `K=16/n16/r40/o0.4` bridge
    inside the cap-128 fixed-bin contract without backing off to `K=12`.

Code/config change:
    Added cap-slack target sources:
    `cap_slack_uncovered_brightness` and `cap_slack_low_alpha`.
    The trainer now samples sparse alpha with tile load projected onto the same
    target grid and passes that load into target-point selection. Added guarded
    post-placement tile repair:
    `tile_overflow_repair_enabled`, `tile_overflow_repair_max_drops`,
    `tile_overflow_repair_guard_refs`, and `tile_overflow_repair_opacity`.
    Repair considers only selected born tubes, drops the minimum set it can find
    from overloaded tiles, and can aim below hard capacity by a guard margin.

Focused gates:
    - `tests/test_star_uvt_feature_target_adapter.py::test_support_birth_split_uncovered_target_points_prefer_low_alpha_bright_pixels`
    - `tests/test_star_uvt_feature_target_adapter.py::test_support_birth_split_sampled_tile_load_maps_grid_to_tile_bins`
    - `tests/test_star_uvt_feature_target_adapter.py::test_support_birth_split_cap_slack_target_points_avoid_loaded_tiles`
    - `tests/test_star_uvt_feature_target_adapter.py::test_support_birth_split_tile_overflow_repair_selects_new_tube_to_drop`
    - `tests/test_star_uvt_feature_target_adapter.py::test_support_birth_split_set_tube_opacity_hides_repaired_tubes`
    - `tests/test_star_uvt_support_birthsplit_preflight.py`
    Focused runs passed before the train.

Ran cap-slack without repair:
    Config:
    `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit16_multicenter_k16_r40_o04_capslack_from1500_lr001_50step_media.jsonc`

    Initial version wrote:
    `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_multicenter_k16_n16_r40_o04_capslack_50step_media.json`

    Result:
    `pass=false`, `tile_overflow_sum=2`, max `131/128`, loss
    `0.755188 -> 0.749640`, feature `0.608854 -> 0.607851`, RGB-probe PSNR
    `24.367 -> 24.504`, dense RGB PSNR `6.500`. The target sampler did what it
    was asked to do: selected tile load was low (`mean=17.706`, max `36`) and
    slack was high (`mean=0.862`). But broad tube footprints still landed on the
    same saturated tiles. Target-pixel tile slack alone is not enough.

Ran exact-fit repair:
    Output:
    `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_multicenter_k16_n16_r40_o04_capslack_repair4_50step_media.json`

    Result:
    `pass=false`, final `tile_overflow_sum=1`, max `129/128`, loss
    `0.754551 -> 0.749341`. Repair dropped two born tubes (`[37,85]`) and
    cleared the initial placement overflow (`4` tiles / `7` excess refs) to
    post-repair max `128`, but training drifted one tile back over capacity.
    Exact-fit repair is too tight.

Ran guarded repair:
    Current config keeps the same filename but writes `repair4g2` artifacts:
    `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit16_multicenter_k16_r40_o04_capslack_from1500_lr001_50step_media.jsonc`

    Outputs:
    - `outputs/checkpoints/2026-05-25_star_uvt_feature_targetgrid_birthsplit16_multicenter_k16_r40_o04_capslack_repair4g2_from1500_lr001_50step.pt`
    - `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_multicenter_k16_n16_r40_o04_capslack_repair4g2_50step_media.json`
    - offline W&B `wandb/offline-run-20260526_014959-1l1i9nn2`

    Result:
    `pass=true`, final `tile_overflow_sum=0`, max `127/128`, loss
    `0.753847 -> 0.749102`, feature `0.608604 -> 0.607608`, RGB-probe PSNR
    `24.400 -> 24.513`, dense RGB PSNR `6.486`. Repair dropped four born tubes
    (`[37,194,732,1192]`), cleared the initial `4` overflowing tiles / `7`
    excess refs, targeted capacity `126`, and measured post-repair max `126`.
    The guard held through 50 training steps.

Dense diagnostic:
    Wrote:
    - `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_capslack_repair4g2_dense_support.json`
    - `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_capslack_repair4g2_dense_support.md`

    Dense table:
    - `start1500`: normal `6.035`, forced-alpha `10.702`, oracle `16.787`,
      alpha `>0.1` `0.647`
    - `n8_r40_90`: normal `6.462`, forced-alpha `14.012`, oracle `21.579`,
      alpha `>0.1` `0.652`
    - `k12_n12_r40_50`: normal `6.483`, forced-alpha `14.019`, oracle
      `21.577`, alpha `>0.1` `0.655`
    - `capslack_repair4g2_50`: normal `6.486`, forced-alpha `14.021`, oracle
      `21.571`, alpha `>0.1` `0.655`

Updated decision:
    Guarded tile repair is now a useful cap-safety primitive for larger support
    birth rows, but it does not change the selected checkpoint. The selected
    support checkpoint remains `K=8/n8/r40/o0.4` 90-step because it still has the
    best objective/feature endpoint. The cap-aware `K=16/n16` row is a tiny
    dense-support nudge over `K=12/n12`, not a solution to the visibility and
    composition gap. Next useful work should add residual/visibility-aware
    scoring, uncovered-pixel selection that accounts for footprint spill, or a
    feature/RGB handoff that changes what the newly born support learns.

## 2026-05-26 Residual-Cap-Slack Support Birth

Purpose:
    Move beyond raw brightness/low-alpha target points by scoring sampled
    support candidates with current black-background RGB residual, uncovered
    alpha, and tile slack. This tests whether the cap-safe support bridge can
    prefer pixels where the current rendered feature/colorizer output is visibly
    wrong, not only bright and uncovered.

Code/config change:
    Added `residual_uncovered_brightness` and
    `cap_slack_residual_uncovered_brightness` support target sources. Residual
    scoring reuses the sampled support grid, renders sparse feature values,
    colorizes them with the frozen probe, compares black-background RGB against
    target RGB at the sampled pixels, and passes a sampled residual tensor into
    target-point scoring. Added focused test coverage for residual-cap-slack
    preference.

Focused gates:
    `PYTHONPATH=src/train uv run --with pytest python -m pytest
    tests/test_star_uvt_feature_target_adapter.py::test_rgb_probe_config_requires_target_grid_materialization
    tests/test_star_uvt_feature_target_adapter.py::test_support_birth_split_cap_slack_target_points_avoid_loaded_tiles
    tests/test_star_uvt_feature_target_adapter.py::test_support_birth_split_residual_cap_slack_target_points_prefer_errorful_uncovered_pixels
    tests/test_star_uvt_feature_target_adapter.py::test_support_birth_split_sampled_tile_load_maps_grid_to_tile_bins
    tests/test_star_uvt_feature_target_adapter.py::test_support_birth_split_tile_overflow_repair_selects_new_tube_to_drop
    tests/test_star_uvt_feature_target_adapter.py::test_support_birth_split_set_tube_opacity_hides_repaired_tubes
    tests/test_star_uvt_support_birthsplit_preflight.py -q`

    Result: `10 passed`.

Ran residual-cap-slack guarded repair:
    Config:
    `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit16_multicenter_k16_r40_o04_residualcapslack_from1500_lr001_50step_media.jsonc`

    Outputs:
    - `outputs/checkpoints/2026-05-25_star_uvt_feature_targetgrid_birthsplit16_multicenter_k16_r40_o04_residualcapslack_repair4g2_from1500_lr001_50step.pt`
    - `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_multicenter_k16_n16_r40_o04_residualcapslack_repair4g2_50step_media.json`
    - offline W&B `wandb/offline-run-20260526_163811-u8cbcvd9`

    Result:
    `pass=true`, final `tile_overflow_sum=0`, max `127/128`, loss
    `0.753586 -> 0.748839`, feature `0.608503 -> 0.607558`, RGB-probe PSNR
    `24.404 -> 24.520`, dense RGB PSNR `6.486`. Target selection did shift:
    `selected_residual_mean=0.803`, selected alpha `0.00618`, selected tile load
    mean/max `18.826/58`, tile slack mean/min `0.853/0.547`. Repair dropped
    four born tubes (`[37,85,194,732]`) and cleared the initial `4` overflowing
    tiles / `7` excess refs to post-repair max `126`.

Dense diagnostic:
    Wrote:
    - `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_residualcapslack_repair4g2_dense_support.json`
    - `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_residualcapslack_repair4g2_dense_support.md`

    Dense table:
    - `start1500`: normal `6.035`, forced-alpha `10.702`, oracle `16.787`,
      alpha `>0.1` `0.647`
    - `n8_r40_90`: normal `6.462`, forced-alpha `14.012`, oracle `21.579`,
      alpha `>0.1` `0.652`
    - `capslack_repair4g2_50`: normal `6.486`, forced-alpha `14.021`, oracle
      `21.571`, alpha `>0.1` `0.655`
    - `residualcapslack_repair4g2_50`: normal `6.486`, forced-alpha `14.019`,
      oracle `21.579`, alpha `>0.1` `0.655`

Updated decision:
    Residual-cap-slack scoring is a small scalar objective/probe improvement but
    not a dense support or media-quality solution. It preserves the cap-safe K16
    bridge and changes which points/tubes are selected, but normal/forced/oracle
    support is essentially flat versus cap-slack repair. The next useful STAR
    support move should account for support footprint spill or change the
    feature/RGB/model handoff; pointwise residual scoring alone is too weak.
    Do not cite the residual row's wall-clock timing as a speed result: the
    machine had unrelated high-CPU jobs during/after the run.

Added footprint-aware residual target scoring:
    Added `footprint_residual_uncovered_brightness` and
    `cap_slack_footprint_residual_uncovered_brightness`. These multiply
    residual, brightness, and uncoveredness, then mean-pool that score over the
    projected support radius before optional tile-slack weighting. The intent
    was to avoid choosing a single high-error pixel whose tube footprint spills
    into already-saturated tiles. Added focused coverage for choosing a nearby
    lower-load neighbor under tile slack.

Focused gates:
    `PYTHONPATH=src/train uv run --with pytest python -m pytest
    tests/test_star_uvt_feature_target_adapter.py::test_support_birth_split_footprint_residual_can_choose_neighbor_with_tile_slack
    tests/test_star_uvt_feature_target_adapter.py::test_support_birth_split_residual_cap_slack_target_points_prefer_errorful_uncovered_pixels
    tests/test_star_uvt_feature_target_adapter.py::test_support_birth_split_cap_slack_target_points_avoid_loaded_tiles
    tests/test_star_uvt_feature_target_adapter.py::test_rgb_probe_config_requires_target_grid_materialization
    tests/test_star_uvt_support_birthsplit_preflight.py -q`

    Result: `8 passed`.

Ran footprint-residual-cap-slack guarded repair:
    Config:
    `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit16_multicenter_k16_r40_o04_footprintresidualcapslack_from1500_lr001_50step_media.jsonc`

    Outputs:
    - `outputs/checkpoints/2026-05-25_star_uvt_feature_targetgrid_birthsplit16_multicenter_k16_r40_o04_footprintresidualcapslack_repair4g2_from1500_lr001_50step.pt`
    - `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_multicenter_k16_n16_r40_o04_footprintresidualcapslack_repair4g2_50step_media.json`
    - offline W&B `wandb/offline-run-20260526_165629-p3a8rfkj`

    Result:
    `pass=true`, final `tile_overflow_sum=0`, max `127/128`, loss
    `0.752912 -> 0.748672`, feature `0.608350 -> 0.607417`, RGB-probe PSNR
    `24.420 -> 24.521`, dense RGB PSNR `6.481`. It selected footprint-smoothed
    points with selected residual `0.755`, selected alpha `0.0549`, tile load
    mean/max `17.698/25`, tile slack mean/min `0.862/0.805`, and
    `footprint_radius_samples=5`. Repair again dropped four born tubes
    (`[37,85,194,732]`) and cleared the initial `4` overflowing tiles /
    `7` excess refs to post-repair max `126`.

Dense diagnostic:
    Wrote:
    - `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_footprintresidualcapslack_repair4g2_dense_support.json`
    - `outputs/benchmarks/2026-05-25_star_uvt_birthsplit_footprintresidualcapslack_repair4g2_dense_support.md`

    Dense table:
    - `start1500`: normal `6.035`, forced-alpha `10.702`, oracle `16.787`,
      alpha `>0.1` `0.647`
    - `n8_r40_90`: normal `6.462`, forced-alpha `14.012`, oracle `21.579`,
      alpha `>0.1` `0.652`
    - `residualcapslack_repair4g2_50`: normal `6.486`, forced-alpha `14.019`,
      oracle `21.579`, alpha `>0.1` `0.655`
    - `footprintresidualcapslack_repair4g2_50`: normal `6.481`,
      forced-alpha `14.021`, oracle `21.576`, alpha `>0.1` `0.655`

Updated decision:
    Footprint-aware residual scoring is the best K16 cap-safe scalar row so far,
    but it does not move dense support. This is an important negative/plateau:
    target scoring has mostly exhausted itself unless the born tubes get a
    better feature/RGB handoff, a direct alpha/composition objective, or a
    visibility-prefix/compositing tape. Do not spend another loop on target
    picker variants before changing that downstream contract.

Added target-grid feature init for born support:
    Added `support_birth_split.feature_init_mode=target_group_mean`. When the
    mode is enabled, the trainer samples the normalized target-grid feature at
    selected support-birth points and initializes each new center group to the
    mean target feature for that group. This changes what the born tubes carry
    immediately; default `preserve` keeps old behavior. Added CPU tests for
    target-grid feature sampling, target-group-mean raw-feature initialization,
    and config validation.

Focused gates before train:
    `PYTHONPATH=src/train uv run --with pytest python -m pytest
    tests/test_star_uvt_feature_target_adapter.py::test_support_birth_split_samples_target_grid_features_at_points
    tests/test_star_uvt_feature_target_adapter.py::test_support_birth_split_target_group_mean_initializes_reallocated_features
    tests/test_star_uvt_feature_target_adapter.py::test_support_birth_split_reallocates_low_opacity_tubes_and_preserves_budget
    tests/test_star_uvt_feature_target_adapter.py::test_rgb_probe_config_requires_target_grid_materialization -q`

    Result: `4 passed`.

Ran footprint-residual-cap-slack target-init guarded repair:
    Config:
    `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit16_multicenter_k16_r40_o04_footprintresidualcapslack_targetinit_from1500_lr001_50step_media.jsonc`

    Outputs:
    - `outputs/checkpoints/2026-05-26_star_uvt_feature_targetgrid_birthsplit16_multicenter_k16_r40_o04_footprintresidualcapslack_targetinit_repair4g2_from1500_lr001_50step.pt`
    - `outputs/benchmarks/2026-05-26_star_uvt_birthsplit_multicenter_k16_n16_r40_o04_footprintresidualcapslack_targetinit_repair4g2_50step_media.json`
    - offline W&B `wandb/offline-run-20260526_171233-mvj5hto2`

    Result:
    `pass=true`, final `tile_overflow_sum=0`, max `127/128`, loss
    `0.752454 -> 0.748504`, feature `0.608332 -> 0.607351`, RGB-probe PSNR
    `24.433 -> 24.524`, dense RGB PSNR `6.488`. Feature init applied:
    selected born-tube feature abs mean moved `0.123 -> 0.416`. Repair again
    dropped `[37,85,194,732]` and cleared the initial `4` overflowing tiles /
    `7` excess refs to post-repair max `126`.

Dense diagnostic:
    Wrote:
    - `outputs/benchmarks/2026-05-26_star_uvt_birthsplit_targetinit_repair4g2_dense_support.json`
    - `outputs/benchmarks/2026-05-26_star_uvt_birthsplit_targetinit_repair4g2_dense_support.md`

    Dense table:
    - `start1500`: normal `6.035`, forced-alpha `10.702`, oracle `16.787`,
      alpha `>0.1` `0.647`
    - `n8_r40_90`: normal `6.462`, forced-alpha `14.012`, oracle `21.579`,
      alpha `>0.1` `0.652`
    - `footprintresidualcapslack_repair4g2_50`: normal `6.481`,
      forced-alpha `14.021`, oracle `21.576`, alpha `>0.1` `0.655`
    - `targetinit_repair4g2_50`: normal `6.488`, forced-alpha `14.054`,
      oracle `21.629`, alpha `>0.1` `0.655`

Updated decision:
    Target-grid feature init is the first small positive model-handoff row:
    content/oracle improves, and the scalar row is the best K16 cap-safe branch
    so far. But alpha coverage is unchanged, so the next bridge should not be
    another target picker or feature init variant. Move to direct alpha/
    composition or visibility-prefix/compositing behavior.

Added support-target alpha bridge:
    Added `support_birth_split.target_alpha_loss_weight`,
    `support_birth_split.target_alpha_target`, and
    `support_birth_split.target_alpha_max_points`. When enabled, the trainer
    takes the selected support-birth target points, maps them to sparse chunk
    pixel ids, renders sparse alpha with cached bins, and sends an F1 alpha VJP
    through `direct_atomic_feature_sparse_pixels_backward_cached_bins`. This is
    intentionally pointwise: it asks the born support target pixels to move
    toward a target alpha, without yet changing target-area composition or
    order/prefix semantics.

Focused gates before train:
    - `.venv/bin/python -m py_compile src/train/star_uvt_visibility_support.py
      src/train/star_uvt_feature_config.py
      src/train/star_uvt_feature_overfit_trainer.py`
    - `PYTHONPATH=src/train uv run --with pytest python -m pytest
      tests/test_star_uvt_feature_target_adapter.py -q`

    Result: `45 passed`. The test update covers chunk-local target pixel ids,
    target-alpha config defaults/validation, and the prior target-grid feature
    init behavior.

Ran footprint-residual-cap-slack target-init plus support-target alpha:
    Config:
    `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit16_multicenter_k16_r40_o04_footprintresidualcapslack_targetinit_targetalpha_from1500_lr001_50step_media.jsonc`

    Outputs:
    - `outputs/checkpoints/2026-05-26_star_uvt_feature_targetgrid_birthsplit16_multicenter_k16_r40_o04_footprintresidualcapslack_targetinit_targetalpha025a075_repair4g2_from1500_lr001_50step.pt`
    - `outputs/benchmarks/2026-05-26_star_uvt_birthsplit_multicenter_k16_n16_r40_o04_footprintresidualcapslack_targetinit_targetalpha025a075_repair4g2_50step_media.json`
    - offline W&B `wandb/offline-run-20260526_180007-jq7px654`

    Result:
    `pass=true`, final `tile_overflow_sum=0`, max `127/128`, total loss
    `0.875695 -> 0.868414`, feature `0.608332 -> 0.607645`, RGB-probe PSNR
    `24.433 -> 24.524`, dense RGB PSNR `6.508`. The new local term learned:
    support-target alpha loss `0.492962 -> 0.478448`, `2048` samples/step.
    Mean support-target alpha timing was high (`1160.7ms`), so this route is
    useful as evidence but not an obvious performance primitive.

Dense diagnostic:
    Wrote:
    - `outputs/benchmarks/2026-05-26_star_uvt_birthsplit_targetalpha_repair4g2_dense_support.json`
    - `outputs/benchmarks/2026-05-26_star_uvt_birthsplit_targetalpha_repair4g2_dense_support.md`

    Dense table:
    - `start1500`: normal `6.035`, forced-alpha `10.702`, oracle `16.787`,
      alpha `>0.1` `0.647`
    - `n8_r40_90`: normal `6.462`, forced-alpha `14.012`, oracle `21.579`,
      alpha `>0.1` `0.652`
    - `targetinit_repair4g2_50`: normal `6.488`, forced-alpha `14.054`,
      oracle `21.629`, alpha `>0.1` `0.655`
    - `targetalpha_repair4g2_50`: normal `6.508`, forced-alpha `14.084`,
      oracle `21.626`, alpha `>0.1` `0.657`

Updated decision:
    The direct pointwise target-alpha bridge is a small positive but not a
    breakthrough. It proves the selected birth targets can receive alpha
    gradients and move their sampled alpha objective, but the dense
    normal/forced/oracle gap barely changes. The next bridge should use
    target-area composition, visibility-prefix/compositing tape, or a support
    mechanism that changes where transmittance accumulates across neighborhoods,
    not just another per-target alpha pressure term.

Added support-target-area patch bridge:
    Added `support_birth_split.target_area_loss_weight`,
    `support_birth_split.target_area_patch_shape`,
    `support_birth_split.target_area_max_points`,
    `support_birth_split.target_area_vjp_mode`, and
    `support_birth_split.target_area_composition`. This bridge maps selected
    support-birth target points to chunk-local pixel patches, renders the sparse
    pixels, and applies the existing sparse-visual patch-mean RGB/composition
    VJP. The first config uses 2x2 patches around `1024` selected birth targets,
    black-background composition, and `manual_hidden64_star_only`.

Focused gates before train:
    - `.venv/bin/python -m py_compile src/train/star_uvt_visibility_support.py
      src/train/star_uvt_feature_config.py
      src/train/star_uvt_feature_overfit_trainer.py`
    - `PYTHONPATH=src/train uv run --with pytest python -m pytest
      tests/test_star_uvt_feature_target_adapter.py -q`

    Result: `46 passed`. The new helper test checks chunk-local patch grouping
    and clamped 2D patches; config validation covers target-area defaults and
    invalid modes/shapes.

Ran footprint-residual-cap-slack target-init plus support-target-area patches:
    Config:
    `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit16_multicenter_k16_r40_o04_footprintresidualcapslack_targetinit_targetarea2_from1500_lr001_50step_media.jsonc`

    Outputs:
    - `outputs/checkpoints/2026-05-26_star_uvt_feature_targetgrid_birthsplit16_multicenter_k16_r40_o04_footprintresidualcapslack_targetinit_targetarea2w05_repair4g2_from1500_lr001_50step.pt`
    - `outputs/benchmarks/2026-05-26_star_uvt_birthsplit_multicenter_k16_n16_r40_o04_footprintresidualcapslack_targetinit_targetarea2w05_repair4g2_50step_media.json`
    - offline W&B `wandb/offline-run-20260526_182358-vcjfr5sh`

    Result:
    `pass=true`, final `tile_overflow_sum=0`, max `127/128`, total loss
    `1.051414 -> 1.040208`, feature `0.608309 -> 0.608125`, RGB-probe PSNR
    `24.433 -> 24.520`, dense RGB PSNR `6.507`. The new local term learned:
    support-target-area loss `0.597970 -> 0.581641`, `1024` target cells/step,
    mean support-target-area timing `208.7ms`.

Dense diagnostic:
    Wrote:
    - `outputs/benchmarks/2026-05-26_star_uvt_birthsplit_targetarea2_repair4g2_dense_support.json`
    - `outputs/benchmarks/2026-05-26_star_uvt_birthsplit_targetarea2_repair4g2_dense_support.md`

    Dense table:
    - `targetarea2_repair4g2_50`: normal `6.507`, forced-alpha `14.085`,
      oracle `21.627`, alpha `>0.1` `0.657`

Updated decision:
    Support-target-area patches are a cheaper local-positive bridge than
    pointwise support-target alpha, but they land on the same dense plateau and
    weaken feature loss compared with target-init. This closes the "maybe local
    target-area composition around born support is enough" branch. Next work
    should move to visibility-prefix/compositing tape behavior or a support
    mechanism that changes transmittance over ordered neighborhoods, not another
    pointwise or tiny-patch pressure objective.

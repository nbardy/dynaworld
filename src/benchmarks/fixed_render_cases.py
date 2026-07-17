from __future__ import annotations

import torch

from runtime_types import GaussianSequence
from trainer_capabilities import trainer_uses_multicam_phase
from fixed_render_graph import FixedRenderCase, FixedRenderChunk, PhaseTimer


def detach_gaussian_sequence(sequence: GaussianSequence) -> GaussianSequence:
    return GaussianSequence(
        xyz=sequence.xyz.detach(),
        scales=sequence.scales.detach(),
        quats=sequence.quats.detach(),
        opacities=sequence.opacities.detach(),
        rgbs=sequence.rgbs.detach(),
        cameras=sequence.cameras,
        camera_state=None,
        auxiliary=sequence.auxiliary,
    )


def prepare_heldout_fixed_render_case(trainer) -> FixedRenderCase:
    if not trainer_uses_multicam_phase(trainer):
        raise ValueError("--target heldout is only supported for multicam trainers.")
    if trainer.multicam_bundle.heldout_frames is None:
        raise ValueError("--target heldout requested, but the config has no heldout frames.")
    setup_timer = PhaseTimer(trainer.device)
    with setup_timer.measure("sample"):
        sequence_data, clip_indices, clip_frames, clip_times, _views = trainer.sample_multicam_clip()
    with setup_timer.measure("encode"):
        decoded = trainer._decode_clip(sequence_data, clip_frames, clip_times)
    background = trainer.rgb_objective.sample_background(
        phase="train",
        like=trainer.multicam_bundle.heldout_frames[0, clip_indices],
        frame_count=len(clip_indices),
    )
    chunks = []
    heldout_count = int(trainer.multicam_bundle.heldout_frames.shape[0])
    for view in range(heldout_count):
        camera_names = trainer.multicam_bundle.heldout_camera_names or []
        camera_name = camera_names[view] if view < len(camera_names) else f"heldout_{view}"
        target = trainer.make_target_view(
            view_id=f"heldout_view_{view}",
            frames=trainer.multicam_bundle.heldout_frames[int(view), clip_indices],
            frame_indices=clip_indices,
            frame_times=trainer.frame_times_for_indices(clip_indices),
            cameras=trainer.camera_rig.heldout_cameras_for(view, clip_indices),
            role="heldout",
            camera_role="heldout",
            camera_owner="external_rig",
            camera_name=camera_name,
            metrics_prefix=f"Heldout{view}_{camera_name}",
        )
        chunks.append(FixedRenderChunk(sequence=detach_gaussian_sequence(decoded), target=target))
    return FixedRenderCase(
        chunks=tuple(chunks),
        background=background.detach() if torch.is_tensor(background) else background,
        total_frames=heldout_count * int(len(clip_indices)),
        setup_phases_ms={phase: float(setup_timer.elapsed_ms.get(phase, 0.0)) for phase in ("sample", "encode")},
    )

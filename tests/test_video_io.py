from __future__ import annotations

from pathlib import Path

import torch

import video_io


def test_rgb_alpha_preview_builds_target_render_alpha_triptych() -> None:
    target = torch.zeros(3, 4, 5)
    render = torch.ones(3, 4, 5)
    alpha = torch.full((4, 5), 0.25)

    preview = video_io.rgb_alpha_preview(target, render, alpha)

    assert preview.shape == (3, 4, 15)
    assert torch.equal(preview[..., :5], target)
    assert torch.equal(preview[..., 5:10], render)
    assert torch.equal(preview[..., 10:], torch.full((3, 4, 5), 0.25))


def test_video_fps_from_config_defaults_and_coerces() -> None:
    assert video_io.video_fps_from_config({}) == 4.0
    assert video_io.video_fps_from_config({"video_fps": "7.5"}) == 7.5


def test_save_render_side_by_side_videos_uses_stable_powerfoam_names(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[Path, torch.Tensor, float]] = []

    def fake_save_mp4(path: Path, frames: torch.Tensor, fps: float) -> None:
        calls.append((path, frames.clone(), fps))

    monkeypatch.setattr(video_io, "save_mp4", fake_save_mp4)
    renders = torch.ones(2, 3, 4, 5)
    targets = torch.zeros(2, 3, 4, 5)

    video_io.save_render_side_by_side_videos(tmp_path, 12, renders, targets, fps=6.0, prefix="heldout_")

    assert [path.name for path, _, _ in calls] == [
        "heldout_render_step_0012.mp4",
        "heldout_side_by_side_step_0012.mp4",
    ]
    assert calls[0][2] == 6.0
    assert torch.equal(calls[0][1], renders)
    assert calls[1][1].shape == (2, 3, 4, 10)
    assert torch.equal(calls[1][1][..., :5], targets)
    assert torch.equal(calls[1][1][..., 5:], renders)


def test_save_rgb_alpha_eval_media_writes_preview_and_optional_heldout_videos(
    monkeypatch,
    tmp_path: Path,
) -> None:
    previews: list[tuple[Path, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    videos: list[tuple[Path, int, torch.Tensor, torch.Tensor, float, str]] = []

    def fake_preview(path: Path, target: torch.Tensor, render: torch.Tensor, alpha: torch.Tensor) -> None:
        previews.append((path, target.clone(), render.clone(), alpha.clone()))

    def fake_videos(
        output_dir: Path,
        step: int,
        renders: torch.Tensor,
        targets: torch.Tensor,
        *,
        fps: float,
        prefix: str = "",
    ) -> None:
        videos.append((output_dir, step, renders.clone(), targets.clone(), fps, prefix))

    monkeypatch.setattr(video_io, "save_rgb_alpha_preview", fake_preview)
    monkeypatch.setattr(video_io, "save_render_side_by_side_videos", fake_videos)
    renders = torch.ones(2, 3, 4, 5)
    targets = torch.zeros(2, 3, 4, 5)
    alphas = torch.full((2, 4, 5), 0.25)
    heldout_renders = renders + 1.0
    heldout_targets = targets + 0.5
    heldout_alphas = torch.full((2, 4, 5), 0.75)

    video_io.save_rgb_alpha_eval_media(
        tmp_path,
        3,
        renders,
        targets,
        alphas,
        fps=12.0,
        save_videos=True,
        heldout_renders=heldout_renders,
        heldout_targets=heldout_targets,
        heldout_alphas=heldout_alphas,
    )

    assert [path.name for path, *_ in previews] == [
        "preview_step_0003.png",
        "heldout_preview_step_0003.png",
    ]
    assert torch.equal(previews[0][1], targets[0])
    assert torch.equal(previews[1][1], heldout_targets[0])
    assert [(step, fps, prefix) for _, step, _, _, fps, prefix in videos] == [
        (3, 12.0, ""),
        (3, 12.0, "heldout_"),
    ]

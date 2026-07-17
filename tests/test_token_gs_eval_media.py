from __future__ import annotations

from pathlib import Path

import torch

import token_gs_trainer


def test_save_token_gs_eval_media_writes_preview_and_optional_videos(
    monkeypatch,
    tmp_path: Path,
) -> None:
    previews: list[tuple[Path, torch.Tensor]] = []
    videos: list[tuple[Path, int, torch.Tensor, torch.Tensor, float]] = []

    def fake_save_png(path: Path, image: torch.Tensor) -> None:
        previews.append((path, image.clone()))

    def fake_save_render_side_by_side_videos(
        output_dir: Path,
        step: int,
        renders: torch.Tensor,
        targets: torch.Tensor,
        *,
        fps: float,
    ) -> None:
        videos.append((output_dir, step, renders.clone(), targets.clone(), fps))

    monkeypatch.setattr(token_gs_trainer, "save_png", fake_save_png)
    monkeypatch.setattr(token_gs_trainer, "save_render_side_by_side_videos", fake_save_render_side_by_side_videos)
    renders = torch.ones(2, 3, 4, 5)
    targets = torch.zeros(2, 3, 4, 5)

    token_gs_trainer.save_token_gs_eval_media(
        tmp_path,
        20,
        renders,
        targets,
        fps=4.0,
        save_videos=True,
    )

    assert [path.name for path, _ in previews] == ["preview_step_0020.png"]
    assert previews[0][1].shape == (3, 4, 10)
    assert torch.equal(previews[0][1][..., :5], targets[0])
    assert torch.equal(previews[0][1][..., 5:], renders[0])
    assert [(step, fps) for _, step, _, _, fps in videos] == [(20, 4.0)]
    assert torch.equal(videos[0][2], renders)
    assert torch.equal(videos[0][3], targets)

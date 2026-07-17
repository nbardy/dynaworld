from __future__ import annotations

import torch

import wandb_media
from wandb_media import (
    add_existing_wandb_media,
    alpha_to_rgb_video,
    build_rgb_alpha_eval_media_payload,
    build_rgb_alpha_validation_video_payload,
    build_validation_video_payload,
    make_step_preview_image,
    make_wandb_image,
)


def test_build_validation_video_payload_uses_render_and_gt_side_by_side(monkeypatch) -> None:
    calls: list[tuple[torch.Tensor, float]] = []

    def fake_video(sequence: torch.Tensor, fps: float) -> str:
        calls.append((sequence.clone(), fps))
        return f"video-{len(calls)}"

    monkeypatch.setattr(wandb_media, "make_wandb_video", fake_video)
    rendered = torch.ones(2, 3, 4, 5)
    target = torch.zeros(2, 3, 4, 5)

    payload = build_validation_video_payload(rendered, target, fps=7.5)

    assert payload == {
        "Render_Video": "video-1",
        "Render_GT_Video": "video-2",
    }
    assert len(calls) == 2
    assert calls[0][1] == 7.5
    assert torch.equal(calls[0][0], rendered)
    side_by_side = calls[1][0]
    assert side_by_side.shape == (2, 3, 4, 10)
    assert torch.equal(side_by_side[..., :5], target)
    assert torch.equal(side_by_side[..., 5:], rendered)


def test_build_rgb_alpha_validation_video_payload_adds_gt_and_alpha(monkeypatch) -> None:
    calls: list[tuple[torch.Tensor, float]] = []

    def fake_video(sequence: torch.Tensor, fps: float) -> str:
        calls.append((sequence.clone(), fps))
        return f"video-{len(calls)}"

    monkeypatch.setattr(wandb_media, "make_wandb_video", fake_video)
    rendered = torch.ones(2, 3, 4, 5)
    target = torch.zeros(2, 3, 4, 5)
    alpha = torch.full((2, 4, 5), 0.25)

    payload = build_rgb_alpha_validation_video_payload(rendered, target, alpha, fps=8.0)

    assert payload == {
        "Render_Video": "video-1",
        "Render_GT_Video": "video-2",
        "GT_Video": "video-3",
        "Alpha_Video": "video-4",
    }
    assert len(calls) == 4
    assert calls[2][1] == 8.0
    assert torch.equal(calls[2][0], target)
    assert calls[3][0].shape == (2, 3, 4, 5)
    assert torch.equal(calls[3][0], torch.full((2, 3, 4, 5), 0.25))


def test_build_rgb_alpha_eval_media_payload_adds_preview_and_optional_videos(monkeypatch) -> None:
    preview_calls: list[tuple[torch.Tensor, torch.Tensor, int]] = []
    video_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]] = []

    def fake_preview(target: torch.Tensor, render: torch.Tensor, step: int) -> str:
        preview_calls.append((target.clone(), render.clone(), step))
        return "preview"

    def fake_video_payload(render: torch.Tensor, target: torch.Tensor, alpha: torch.Tensor, fps: float) -> dict[str, str]:
        video_calls.append((render.clone(), target.clone(), alpha.clone(), fps))
        return {"Render_Video": "render-video"}

    monkeypatch.setattr(wandb_media, "make_step_preview_image", fake_preview)
    monkeypatch.setattr(wandb_media, "build_rgb_alpha_validation_video_payload", fake_video_payload)
    rendered = torch.ones(2, 3, 4, 5)
    target = torch.zeros(2, 3, 4, 5)
    alpha = torch.full((2, 4, 5), 0.25)

    payload = build_rgb_alpha_eval_media_payload(
        rendered,
        target,
        alpha,
        step=9,
        fps=12.0,
        include_videos=True,
    )
    no_video_payload = build_rgb_alpha_eval_media_payload(
        rendered,
        target,
        alpha,
        step=10,
        fps=12.0,
        include_videos=False,
    )

    assert payload == {"Preview": "preview", "Render_Video": "render-video"}
    assert no_video_payload == {"Preview": "preview"}
    assert [call[2] for call in preview_calls] == [9, 10]
    assert len(video_calls) == 1
    assert video_calls[0][3] == 12.0


def test_alpha_to_rgb_video_accepts_single_channel_alpha() -> None:
    alpha = torch.full((2, 1, 4, 5), 0.5)

    rgb_alpha = alpha_to_rgb_video(alpha)

    assert rgb_alpha.shape == (2, 3, 4, 5)
    assert torch.equal(rgb_alpha, torch.full((2, 3, 4, 5), 0.5))


def test_make_step_preview_image_uses_shared_caption(monkeypatch) -> None:
    calls: list[tuple[torch.Tensor, torch.Tensor, str]] = []

    def fake_preview(target: torch.Tensor, render: torch.Tensor, caption: str) -> str:
        calls.append((target, render, caption))
        return "image"

    monkeypatch.setattr(wandb_media, "make_preview_image", fake_preview)
    target = torch.zeros(3, 4, 5)
    render = torch.ones(3, 4, 5)

    image = make_step_preview_image(target, render, step=12)

    assert image == "image"
    assert calls == [(target, render, "step 12: GT | render")]


def test_make_wandb_image_forwards_image_and_caption(monkeypatch) -> None:
    calls: list[tuple[object, str]] = []

    class FakeImage:
        def __init__(self, image: object, *, caption: str) -> None:
            calls.append((image, caption))

    monkeypatch.setattr(wandb_media.wandb, "Image", FakeImage)
    image = object()

    result = make_wandb_image(image, caption="Feature PCA")

    assert isinstance(result, FakeImage)
    assert calls == [(image, "Feature PCA")]


def test_add_existing_wandb_media_adds_only_present_paths(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "preview.png"
    video_path = tmp_path / "video.mp4"
    image_path.write_bytes(b"png")
    video_path.write_bytes(b"mp4")
    calls: list[tuple[str, str, str | None]] = []

    class FakeImage:
        def __init__(self, path: str) -> None:
            calls.append(("image", path, None))

    class FakeVideo:
        def __init__(self, path: str, *, format: str) -> None:
            calls.append(("video", path, format))

    monkeypatch.setattr(wandb_media.wandb, "Image", FakeImage)
    monkeypatch.setattr(wandb_media.wandb, "Video", FakeVideo)
    payload: dict[str, object] = {}

    add_existing_wandb_media(
        payload,
        {
            "preview": image_path,
            "video": video_path,
            "missing": tmp_path / "missing.mp4",
        },
        image_outputs=(("preview", "Preview"),),
        video_outputs=(("video", "Video"), ("missing", "Missing")),
    )

    assert set(payload) == {"Preview", "Video"}
    assert calls == [
        ("image", str(image_path), None),
        ("video", str(video_path), "mp4"),
    ]

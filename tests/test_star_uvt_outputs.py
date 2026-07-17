from __future__ import annotations

import json

import torch

from star_uvt_outputs import (
    log_star_uvt_row_outputs,
    side_by_side_fps,
    write_prediction_media,
    write_row_json,
    write_row_json_and_print,
)


def test_side_by_side_fps_prefers_output_then_data_then_default() -> None:
    assert side_by_side_fps({"side_by_side_fps": 12}, {"fps": 24}) == 12.0
    assert side_by_side_fps({"side_by_side_fps": None}, {"fps": 24}) == 24.0
    assert side_by_side_fps({"side_by_side_fps": None}, {"fps": None}) == 30.0


def test_write_prediction_media_uses_shared_fps_fallback(tmp_path) -> None:
    calls: list[tuple[str, object]] = []

    def contact_writer(path, target, pred, alpha, *, max_frames, mode):
        calls.append(("contact", (path, target.shape, pred.shape, alpha, max_frames, mode)))

    def video_writer(path, target, pred, alpha, *, fps):
        calls.append(("video", (path, target.shape, pred.shape, alpha, fps)))

    target = torch.zeros((2, 4, 4, 3))
    pred = torch.ones((2, 4, 4, 3))
    contact_path, video_path = write_prediction_media(
        target_thwc=target,
        pred_thwc=pred,
        output_cfg={
            "contact_sheet": str(tmp_path / "contact.png"),
            "side_by_side_video": str(tmp_path / "video.mp4"),
            "side_by_side_fps": None,
            "contact_sheet_frames": 2,
            "contact_sheet_mode": "first",
        },
        data_cfg={"fps": 7.5},
        contact_sheet_writer=contact_writer,
        side_by_side_writer=video_writer,
    )

    assert contact_path == tmp_path / "contact.png"
    assert video_path == tmp_path / "video.mp4"
    assert calls[0][0] == "contact"
    assert calls[0][1][-2:] == (2, "first")
    assert calls[1][0] == "video"
    assert calls[1][1][-1] == 7.5


def test_write_row_json_serializes_sorted_pretty_json(tmp_path) -> None:
    path = write_row_json({"b": 2, "a": 1}, tmp_path / "row.json")

    assert path == tmp_path / "row.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    assert path.read_text(encoding="utf-8").endswith("\n")


def test_write_row_json_and_print_uses_same_sorted_payload(tmp_path, capsys) -> None:
    path = write_row_json_and_print({"b": 2, "a": 1}, tmp_path / "row.json")

    assert path == tmp_path / "row.json"
    assert path.read_text(encoding="utf-8") == '{\n  "a": 1,\n  "b": 2\n}\n'
    assert capsys.readouterr().out == '{\n  "a": 1,\n  "b": 2\n}\n'


def test_log_star_uvt_row_outputs_uses_standard_media_keys(monkeypatch) -> None:
    calls = []

    def fake_log(row, cfg, *, metric_prefix, image_outputs, video_outputs):
        calls.append((row, cfg, metric_prefix, image_outputs, video_outputs))

    monkeypatch.setattr("star_uvt_outputs.log_wandb_row_outputs", fake_log)
    row = {"loss": 1.0}
    cfg = {"output": {}}

    log_star_uvt_row_outputs(row, cfg, metric_prefix="star_uvt")

    assert calls == [
        (
            row,
            cfg,
            "star_uvt",
            (("contact_sheet", "media/contact_sheet"),),
            (("side_by_side_video", "media/side_by_side_video"),),
        )
    ]


def test_log_star_uvt_row_outputs_allows_extra_feature_probe_media(monkeypatch) -> None:
    calls = []

    def fake_log(row, cfg, *, metric_prefix, image_outputs, video_outputs):
        calls.append((metric_prefix, image_outputs, video_outputs))

    monkeypatch.setattr("star_uvt_outputs.log_wandb_row_outputs", fake_log)

    log_star_uvt_row_outputs(
        {"loss": 1.0},
        {"output": {}},
        metric_prefix="star_uvt_feature",
        image_outputs=(
            ("contact_sheet", "media/contact_sheet"),
            ("rgb_probe_contact_sheet", "media/rgb_probe_contact_sheet"),
        ),
        video_outputs=(
            ("side_by_side_video", "media/side_by_side_video"),
            ("rgb_probe_side_by_side_video", "media/rgb_probe_side_by_side_video"),
        ),
    )

    assert calls == [
        (
            "star_uvt_feature",
            (
                ("contact_sheet", "media/contact_sheet"),
                ("rgb_probe_contact_sheet", "media/rgb_probe_contact_sheet"),
            ),
            (
                ("side_by_side_video", "media/side_by_side_video"),
                ("rgb_probe_side_by_side_video", "media/rgb_probe_side_by_side_video"),
            ),
        )
    ]

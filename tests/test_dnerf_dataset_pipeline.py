from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from dataset_pipeline.dnerf import inspect


def _write_split(scene: Path, split: str) -> None:
    frames = []
    for index, time in enumerate((0.0, 1.0)):
        relative = f"{split}/r_{index}"
        image_path = scene / f"{relative}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 6)).save(image_path)
        frames.append(
            {
                "file_path": relative,
                "time": time,
                "transform_matrix": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        )
    (scene / f"transforms_{split}.json").write_text(
        json.dumps({"camera_angle_x": 0.7, "frames": frames}),
        encoding="utf-8",
    )


def test_inspect_validates_controlled_dnerf_splits(tmp_path: Path) -> None:
    root = tmp_path / "dnerf"
    scene = root / "extracted" / "data" / "bouncingballs"
    scene.mkdir(parents=True)
    for split in ("train", "val", "test"):
        _write_split(scene, split)

    inventory = inspect(
        {
            "dataset_name": "dnerf_test",
            "download_url": "https://example.test/data.zip",
            "controlled_scenes": ["bouncingballs"],
        },
        root,
    )

    assert inventory["controlled_scenes"] == ["bouncingballs"]
    assert [row["frame_count"] for row in inventory["scenes"][0]["splits"]] == [2, 2, 2]
    assert inventory["scenes"][0]["splits"][0]["image_size"] == [8, 6]
    assert (root / "metadata" / "controlled_scene_inventory.json").exists()

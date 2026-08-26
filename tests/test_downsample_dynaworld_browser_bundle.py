from __future__ import annotations

import json
from pathlib import Path

import pytest
from downsample_dynaworld_browser_bundle import downsample_browser_bundle
from PIL import Image


def test_downsample_browser_bundle_preserves_contract_and_frame_boundaries(tmp_path: Path) -> None:
    source_atlas = tmp_path / "source_cam00.png"
    atlas = Image.new("RGB", (8, 2))
    atlas.paste((255, 0, 0), (0, 0, 4, 2))
    atlas.paste((0, 255, 0), (4, 0, 8, 2))
    atlas.save(source_atlas)
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "name": "fixture 4x2",
                "decode_size": [4, 2],
                "frame_count": 2,
                "dataset_contract": {"split": "train1_holdout0"},
                "cameras": [
                    {
                        "name": "cam00",
                        "role": "train",
                        "intrinsics": [1.0, 1.0, 0.5, 0.5],
                        "frame_atlas_url": "./source_cam00.png",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "derived.json"
    downsample_browser_bundle(source, output, width=2, height=1)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["decode_size"] == [2, 1]
    assert payload["dataset_contract"] == {"split": "train1_holdout0"}
    assert payload["cameras"][0]["intrinsics"] == [1.0, 1.0, 0.5, 0.5]
    with Image.open(tmp_path / "derived_cam00.png") as reduced:
        assert reduced.size == (4, 1)
        assert reduced.getpixel((0, 0))[0] > 240
        assert reduced.getpixel((1, 0))[1] < 10
        assert reduced.getpixel((2, 0))[1] > 240
        assert reduced.getpixel((2, 0))[0] < 10


def test_downsample_browser_bundle_rejects_upscaling(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"decode_size": [4, 2], "frame_count": 1}), encoding="utf-8")

    with pytest.raises(ValueError, match="only supports downsampling"):
        downsample_browser_bundle(source, tmp_path / "output.json", width=8, height=4)

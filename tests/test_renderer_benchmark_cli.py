from __future__ import annotations

from pathlib import Path

import pytest
import torch

from renderer_benchmark_cli import (
    apply_save_image_cli_overrides,
    deep_merge,
    normalize_resolution,
    parse_csv_floats,
    parse_csv_ints,
    parse_csv_resolutions,
    parse_csv_strings,
    resolve_image_save_target,
    resolve_project_path,
    row_matches_image_save_target,
    safe_filename_part,
    save_chw_image,
    torch_dtype_from_name,
)


def test_parse_resolution_values() -> None:
    assert normalize_resolution(64) == (64, 64)
    assert normalize_resolution("128x96") == (128, 96)
    assert normalize_resolution([32, 24]) == (32, 24)
    assert parse_csv_resolutions("64,128x96") == [(64, 64), (128, 96)]


def test_deep_merge_preserves_base_and_merges_nested_values() -> None:
    base = {"render": {"tile_size": 16, "alpha": 0.1}, "device": "auto"}
    merged = deep_merge(base, {"render": {"alpha": 0.2}})

    assert merged == {"render": {"tile_size": 16, "alpha": 0.2}, "device": "auto"}
    assert base == {"render": {"tile_size": 16, "alpha": 0.1}, "device": "auto"}


def test_parse_csv_values() -> None:
    assert parse_csv_ints("1, 4,16") == [1, 4, 16]
    assert parse_csv_floats("0.5, 2,3.25") == [0.5, 2.0, 3.25]
    assert parse_csv_strings("taichi, raw_metal") == ["taichi", "raw_metal"]


def test_apply_save_image_cli_overrides() -> None:
    cfg = {"save_images": {"enabled": False, "directory": "old"}}

    returned = apply_save_image_cli_overrides(
        cfg,
        save_images=Path("new/previews"),
        no_save_images=False,
    )

    assert returned is cfg
    assert cfg["save_images"] == {"enabled": True, "directory": "new/previews"}
    apply_save_image_cli_overrides(cfg, save_images=None, no_save_images=True)
    assert cfg["save_images"] == {"enabled": False, "directory": "new/previews"}


def test_torch_dtype_from_name() -> None:
    assert torch_dtype_from_name("float32") is torch.float32
    with pytest.raises(ValueError, match="Taichi precision sentinel"):
        torch_dtype_from_name("input", input_sentinel_error=True)


def test_project_path_and_safe_filename() -> None:
    root = Path("/tmp/dynaworld")
    assert resolve_project_path("outputs/result.json", root) == root / "outputs/result.json"
    assert resolve_project_path("/tmp/absolute.json", root) == Path("/tmp/absolute.json")
    assert safe_filename_part("raw metal/v5") == "raw_metal_v5"
    assert safe_filename_part("frame.001/v5", allow_dot=False) == "frame_001_v5"


def test_image_save_target_matches_largest_case_and_status() -> None:
    target = resolve_image_save_target(
        {"largest_resolution_only": True, "largest_splat_count_only": True, "set_index": 2},
        [(32, 32), (64, 16), (64, 64)],
        [16, 32],
    )
    matching_row = {"status": "ok", "height": 64, "width": 64, "splat_count": 32, "set_index": 2}

    assert target.resolutions == {(64, 64)}
    assert target.splat_counts == {32}
    assert row_matches_image_save_target(matching_row, target, required_status="ok")
    assert not row_matches_image_save_target({**matching_row, "status": "error"}, target, required_status="ok")
    assert not row_matches_image_save_target({**matching_row, "set_index": 1}, target, required_status="ok")


def test_save_chw_image_creates_parent_and_writes_png(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "preview.png"

    written = save_chw_image(torch.ones(1, 2, 3), path)

    assert written == path
    assert path.read_bytes().startswith(b"\x89PNG")

#!/usr/bin/env python3
"""Merge simple ASCII XYZRGB PLY point clouds for PowerFoam init probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PLY_PROPERTIES = (
    "property float x",
    "property float y",
    "property float z",
    "property uchar red",
    "property uchar green",
    "property uchar blue",
)


def read_ascii_xyzrgb_ply(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        header: list[str] = []
        for line in f:
            header.append(line.rstrip("\n"))
            if line.rstrip("\n") == "end_header":
                break
        else:
            raise ValueError(f"{path} has no end_header.")

        if header[:2] != ["ply", "format ascii 1.0"]:
            raise ValueError(f"{path} is not an ASCII PLY file.")
        try:
            vertex_line = next(line for line in header if line.startswith("element vertex "))
        except StopIteration as exc:
            raise ValueError(f"{path} has no vertex count.") from exc
        vertex_count = int(vertex_line.split()[-1])
        property_lines = [line for line in header if line.startswith("property ")]
        if tuple(property_lines[: len(PLY_PROPERTIES)]) != PLY_PROPERTIES:
            raise ValueError(f"{path} must use x/y/z uchar RGB vertex properties.")

        rows = [line.rstrip("\n") for line in f if line.strip()]
    if len(rows) != vertex_count:
        raise ValueError(f"{path} declared {vertex_count} vertices but contains {len(rows)} rows.")
    return rows


def load_metadata(path: Path) -> dict[str, Any] | None:
    metadata_path = path.with_suffix(".json")
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def source_summary(path: Path, rows: list[str]) -> dict[str, Any]:
    metadata = load_metadata(path)
    summary: dict[str, Any] = {
        "path": str(path),
        "vertex_count": len(rows),
        "metadata_path": str(path.with_suffix(".json")) if path.with_suffix(".json").exists() else None,
    }
    if metadata is None:
        return summary
    for key in (
        "frame_index",
        "point_count",
        "target_size",
        "database_num_verified_image_pairs",
        "database_num_keypoints",
        "filtered_reproj_error",
        "filtered_track_length",
        "filtered_track_length_histogram",
    ):
        if key in metadata:
            summary[key] = metadata[key]
    return summary


def write_ascii_xyzrgb_ply(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(rows)}\n")
        for property_line in PLY_PROPERTIES:
            f.write(f"{property_line}\n")
        f.write("end_header\n")
        for row in rows:
            f.write(f"{row}\n")


def build_summary(output: Path, sources: list[dict[str, Any]], rows: list[str]) -> dict[str, Any]:
    frame_indices = [source.get("frame_index") for source in sources if source.get("frame_index") is not None]
    return {
        "output": str(output),
        "point_count": len(rows),
        "coordinate_frame": "model",
        "merge_mode": "concat_no_dedup",
        "source_count": len(sources),
        "frame_indices": frame_indices,
        "sources": sources,
        "note": "Concatenated clean known-pose per-frame pycolmap PLYs; this increases init points but does not create longer multiview tracks.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", type=Path, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_rows: list[str] = []
    sources: list[dict[str, Any]] = []
    for path in args.inputs:
        rows = read_ascii_xyzrgb_ply(path)
        all_rows.extend(rows)
        sources.append(source_summary(path, rows))

    write_ascii_xyzrgb_ply(args.output, all_rows)
    summary = build_summary(args.output, sources, all_rows)
    with args.output.with_suffix(".json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from PIL import Image


def downsample_browser_bundle(source_path: Path, output_path: Path, *, width: int, height: int) -> Path:
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    source_width, source_height = (int(value) for value in payload["decode_size"])
    frame_count = int(payload["frame_count"])
    if min(width, height, frame_count) < 1:
        raise ValueError("Target size and frame count must be positive.")
    if width > source_width or height > source_height:
        raise ValueError("Bundle derivation only supports downsampling.")

    result = deepcopy(payload)
    result["decode_size"] = [width, height]
    result["name"] = str(result["name"]).replace(
        f"{source_width}x{source_height}", f"{width}x{height}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for camera in result["cameras"]:
        source_atlas = source_path.parent / str(camera["frame_atlas_url"]).removeprefix("./")
        output_atlas = output_path.with_name(f"{output_path.stem}_{camera['name']}.png")
        with Image.open(source_atlas) as atlas:
            expected_size = (source_width * frame_count, source_height)
            if atlas.size != expected_size:
                raise ValueError(f"Atlas {source_atlas} is {atlas.size}, expected {expected_size}.")
            # Resize frames independently so Lanczos cannot blend across atlas seams.
            reduced = Image.new("RGB", (width * frame_count, height))
            for frame_index in range(frame_count):
                frame = atlas.crop(
                    (frame_index * source_width, 0, (frame_index + 1) * source_width, source_height)
                )
                reduced.paste(frame.resize((width, height), Image.Resampling.LANCZOS), (frame_index * width, 0))
            reduced.save(output_atlas, optimize=True)
        camera["frame_atlas_url"] = f"./{output_atlas.name}"

    output_path.write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive a smaller exact Dynaworld browser dataset bundle.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    args = parser.parse_args()
    print(downsample_browser_bundle(args.source, args.output, width=args.width, height=args.height))


if __name__ == "__main__":
    main()

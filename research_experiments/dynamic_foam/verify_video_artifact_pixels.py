from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def ffprobe_stream(path: Path) -> dict[str, Any]:
    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,codec_tag_string,width,height,pix_fmt,r_frame_rate,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    return json.loads(output)["streams"][0]


def extract_first_frame(path: Path, output: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(path),
            "-frames:v",
            "1",
            str(output),
        ],
        check=True,
    )


def frame_stats(path: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="dynaworld_video_check_") as tmp:
        frame_path = Path(tmp) / "frame0.png"
        extract_first_frame(path, frame_path)
        frame = np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.float32) / 255.0
    pixels = frame.reshape(-1, 3)
    mean = pixels.mean(axis=0)
    var = pixels.var(axis=0)
    sample_stride = max(1, pixels.shape[0] // 20000)
    unique_sample = np.unique((pixels[::sample_stride] * 255.0).astype(np.uint8), axis=0).shape[0]
    green_both = np.mean((frame[..., 1] > frame[..., 0] + 0.05) & (frame[..., 1] > frame[..., 2] + 0.05))
    return {
        "mean_rgb": [float(value) for value in mean],
        "var_rgb": [float(value) for value in var],
        "var_min": float(var.min()),
        "green_both_plus_005_fraction": float(green_both),
        "sample_unique_rgb": int(unique_sample),
    }


def verify_path(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    stream = ffprobe_stream(path)
    stats = frame_stats(path)
    checks = [
        {
            "name": "first_frame_has_variance",
            "passed": stats["var_min"] >= float(args.min_channel_variance),
            "evidence": stats["var_min"],
        },
        {
            "name": "first_frame_has_unique_colors",
            "passed": stats["sample_unique_rgb"] >= int(args.min_unique_rgb),
            "evidence": stats["sample_unique_rgb"],
        },
        {
            "name": "not_solid_green",
            "passed": stats["green_both_plus_005_fraction"] <= float(args.max_green_dominance_fraction),
            "evidence": stats["green_both_plus_005_fraction"],
        },
    ]
    if args.require_h264:
        checks.append(
            {
                "name": "h264_avc1",
                "passed": stream.get("codec_name") == "h264" and stream.get("codec_tag_string") == "avc1",
                "evidence": {
                    "codec_name": stream.get("codec_name"),
                    "codec_tag_string": stream.get("codec_tag_string"),
                },
            }
        )
    ok = all(check["passed"] for check in checks)
    return {
        "path": str(path),
        "ok": ok,
        "stream": stream,
        "first_frame": stats,
        "checks": checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify saved MP4 first-frame pixels are not flat/green.")
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--require-h264", action="store_true")
    parser.add_argument("--min-channel-variance", type=float, default=1.0e-4)
    parser.add_argument("--min-unique-rgb", type=int, default=32)
    parser.add_argument("--max-green-dominance-fraction", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = [verify_path(path, args) for path in args.paths]
    payload = {
        "schema_version": "video_artifact_pixel_check_v1",
        "ok": all(result["ok"] for result in results),
        "results": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

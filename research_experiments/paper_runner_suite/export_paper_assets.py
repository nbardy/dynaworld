#!/usr/bin/env python3
from __future__ import annotations

"""Export paper SVGs through one fail-closed, manifest-driven contract.

The exporter deliberately rasterizes both PNG and PDF outputs.  That gives the
two paper packages the same venue-portable contract: one opaque, 8-bit RGB
image with explicit pixel dimensions and physical resolution.  The verifier
checks the produced bytes rather than trusting ImageMagick's requested flags.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
RECORD_SCHEMA_VERSION = 1
MAX_DIMENSION_PX = 20_000
MAX_PIXELS = 100_000_000
MAX_DPI = 1_200
ASPECT_RELATIVE_TOLERANCE = 1e-6
DPI_TOLERANCE = 0.05
PDF_POINT_TOLERANCE = 0.1
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
SVG_TAG = "{http://www.w3.org/2000/svg}svg"
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
HEX_COLOR_PATTERN = re.compile(r"^#[0-9A-Fa-f]{6}$")
DIMENSION_PATTERN = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)(?:px)?\s*$")

TOP_LEVEL_KEYS = frozenset(
    {"schema_version", "source_root", "output_root", "record", "assets"}
)
ASSET_KEYS = frozenset(
    {
        "id",
        "source",
        "output",
        "format",
        "width_px",
        "height_px",
        "dpi",
        "background",
    }
)


class PaperAssetError(RuntimeError):
    """Raised when an export cannot satisfy the paper-asset contract."""


@dataclass(frozen=True)
class AssetSpec:
    asset_id: str
    source_label: str
    output_label: str
    source_path: Path
    output_path: Path
    output_format: str
    width_px: int
    height_px: int
    dpi: int
    background: str
    source_width: float
    source_height: float

    @property
    def render_density_dpi(self) -> float:
        scale = max(
            self.width_px / self.source_width,
            self.height_px / self.source_height,
        )
        return max(96.0, 96.0 * scale)


@dataclass(frozen=True)
class ExportManifest:
    path: Path
    source_root: Path
    output_root: Path
    record_path: Path
    assets: tuple[AssetSpec, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PaperAssetError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PaperAssetError(f"could not read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PaperAssetError(f"expected a JSON object in {path}")
    return value


def _reject_unknown_keys(
    value: dict[str, Any], allowed: frozenset[str], *, context: str
) -> None:
    unknown = sorted(set(value) - allowed)
    missing = sorted(allowed - set(value))
    if unknown:
        raise PaperAssetError(f"unknown {context} keys: {', '.join(unknown)}")
    if missing:
        raise PaperAssetError(f"missing {context} keys: {', '.join(missing)}")


def _required_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PaperAssetError(f"{field} must be a non-empty string")
    return value


def _required_int(value: Any, *, field: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise PaperAssetError(f"{field} must be an integer in [1, {maximum}]")
    return value


def _resolve_root(manifest_dir: Path, label: str, *, field: str) -> Path:
    path = Path(_required_string(label, field=field))
    if path.is_absolute():
        raise PaperAssetError(f"{field} must be relative to the manifest")
    return (manifest_dir / path).resolve()


def _resolve_beneath(root: Path, label: str, *, field: str) -> Path:
    relative = Path(_required_string(label, field=field))
    if relative.is_absolute():
        raise PaperAssetError(f"{field} must be relative to its declared root")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PaperAssetError(f"{field} escapes its declared root: {label}") from exc
    return resolved


def _numeric_svg_dimension(value: str | None, *, field: str) -> float:
    if value is None:
        raise PaperAssetError(f"SVG is missing {field}")
    match = DIMENSION_PATTERN.fullmatch(value)
    if match is None:
        raise PaperAssetError(
            f"SVG {field} must be a unitless or px numeric value, got {value!r}"
        )
    dimension = float(match.group(1))
    if dimension <= 0:
        raise PaperAssetError(f"SVG {field} must be positive")
    return dimension


def _svg_dimensions(path: Path) -> tuple[float, float]:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise PaperAssetError(f"could not parse SVG {path}: {exc}") from exc
    if root.tag != SVG_TAG:
        raise PaperAssetError(f"source is not an SVG document: {path}")
    width = _numeric_svg_dimension(root.attrib.get("width"), field="width")
    height = _numeric_svg_dimension(root.attrib.get("height"), field="height")
    return width, height


def load_manifest(path: Path) -> ExportManifest:
    manifest_path = path.resolve()
    raw = _load_json_object(manifest_path)
    _reject_unknown_keys(raw, TOP_LEVEL_KEYS, context="manifest")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise PaperAssetError(
            f"unsupported manifest schema_version {raw['schema_version']!r}; "
            f"expected {SCHEMA_VERSION}"
        )

    manifest_dir = manifest_path.parent
    source_root = _resolve_root(
        manifest_dir, raw["source_root"], field="source_root"
    )
    output_root = _resolve_root(
        manifest_dir, raw["output_root"], field="output_root"
    )
    record_path = _resolve_beneath(output_root, raw["record"], field="record")
    rows = raw["assets"]
    if not isinstance(rows, list) or not rows:
        raise PaperAssetError("assets must be a non-empty list")

    assets: list[AssetSpec] = []
    seen_ids: set[str] = set()
    seen_outputs: set[Path] = set()
    for index, row in enumerate(rows):
        context = f"asset[{index}]"
        if not isinstance(row, dict):
            raise PaperAssetError(f"{context} must be an object")
        _reject_unknown_keys(row, ASSET_KEYS, context=context)

        asset_id = _required_string(row["id"], field=f"{context}.id")
        if ID_PATTERN.fullmatch(asset_id) is None:
            raise PaperAssetError(f"{context}.id has unsupported characters: {asset_id}")
        if asset_id in seen_ids:
            raise PaperAssetError(f"duplicate asset id: {asset_id}")
        seen_ids.add(asset_id)

        source_label = _required_string(row["source"], field=f"{context}.source")
        output_label = _required_string(row["output"], field=f"{context}.output")
        source_path = _resolve_beneath(
            source_root, source_label, field=f"{context}.source"
        )
        output_path = _resolve_beneath(
            output_root, output_label, field=f"{context}.output"
        )
        if not source_path.is_file():
            raise PaperAssetError(f"missing SVG source: {source_path}")
        if source_path.suffix.lower() != ".svg":
            raise PaperAssetError(f"source must have .svg suffix: {source_path}")

        output_format = _required_string(
            row["format"], field=f"{context}.format"
        ).lower()
        if output_format not in {"png", "pdf"}:
            raise PaperAssetError(f"{context}.format must be png or pdf")
        if output_path.suffix.lower() != f".{output_format}":
            raise PaperAssetError(
                f"{context}.output suffix does not match format {output_format}"
            )
        if output_path in seen_outputs:
            raise PaperAssetError(f"duplicate output path: {output_path}")
        if output_path == record_path:
            raise PaperAssetError(f"asset output collides with record: {output_path}")
        seen_outputs.add(output_path)

        width_px = _required_int(
            row["width_px"], field=f"{context}.width_px", maximum=MAX_DIMENSION_PX
        )
        height_px = _required_int(
            row["height_px"], field=f"{context}.height_px", maximum=MAX_DIMENSION_PX
        )
        if width_px * height_px > MAX_PIXELS:
            raise PaperAssetError(
                f"{context} exceeds the {MAX_PIXELS:,}-pixel safety limit"
            )
        dpi = _required_int(row["dpi"], field=f"{context}.dpi", maximum=MAX_DPI)
        background = _required_string(
            row["background"], field=f"{context}.background"
        )
        if HEX_COLOR_PATTERN.fullmatch(background) is None:
            raise PaperAssetError(
                f"{context}.background must be an opaque #RRGGBB color"
            )

        source_width, source_height = _svg_dimensions(source_path)
        source_aspect = source_width / source_height
        output_aspect = width_px / height_px
        relative_error = abs(source_aspect - output_aspect) / source_aspect
        if relative_error > ASPECT_RELATIVE_TOLERANCE:
            raise PaperAssetError(
                f"{context} changes aspect ratio: SVG {source_width:g}x"
                f"{source_height:g}, export {width_px}x{height_px}"
            )

        assets.append(
            AssetSpec(
                asset_id=asset_id,
                source_label=source_label,
                output_label=output_label,
                source_path=source_path,
                output_path=output_path,
                output_format=output_format,
                width_px=width_px,
                height_px=height_px,
                dpi=dpi,
                background=background.lower(),
                source_width=source_width,
                source_height=source_height,
            )
        )

    return ExportManifest(
        path=manifest_path,
        source_root=source_root,
        output_root=output_root,
        record_path=record_path,
        assets=tuple(assets),
    )


def _require_tool(name: str) -> Path:
    resolved = shutil.which(name)
    if resolved is None:
        raise PaperAssetError(f"required paper-asset tool is unavailable: {name}")
    return Path(resolved).resolve()


def _run_tool(
    argv: list[str], *, timeout_seconds: int = 120, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PaperAssetError(f"tool invocation failed: {argv[0]}: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise PaperAssetError(
            f"tool returned {result.returncode}: {' '.join(argv)}\n{detail}"
        )
    return result


def _tool_identity(path: Path, *version_args: str) -> dict[str, str]:
    result = _run_tool([str(path), *version_args], timeout_seconds=30)
    lines = [line.strip() for line in (result.stdout + result.stderr).splitlines()]
    version = next((line for line in lines if line), "unknown")
    return {"executable": str(path), "version": version}


def _png_chunks(path: Path) -> list[tuple[bytes, bytes]]:
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise PaperAssetError(f"not a PNG file: {path}")
    chunks: list[tuple[bytes, bytes]] = []
    offset = len(PNG_SIGNATURE)
    saw_iend = False
    while offset < len(data):
        if offset + 12 > len(data):
            raise PaperAssetError(f"truncated PNG chunk in {path}")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        end = offset + 12 + length
        if end > len(data):
            raise PaperAssetError(f"truncated PNG payload in {path}")
        payload = data[offset + 8 : offset + 8 + length]
        stored_crc = struct.unpack(">I", data[offset + 8 + length : end])[0]
        computed_crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
        if stored_crc != computed_crc:
            raise PaperAssetError(f"PNG CRC mismatch in {path}")
        chunks.append((chunk_type, payload))
        offset = end
        if chunk_type == b"IEND":
            saw_iend = True
            break
    if not saw_iend:
        raise PaperAssetError(f"PNG is missing IEND: {path}")
    if offset != len(data):
        raise PaperAssetError(f"PNG has trailing bytes after IEND: {path}")
    return chunks


def _verify_png(path: Path, spec: AssetSpec) -> dict[str, Any]:
    chunks = _png_chunks(path)
    if not chunks or chunks[0][0] != b"IHDR" or len(chunks[0][1]) != 13:
        raise PaperAssetError(f"PNG has invalid IHDR: {path}")
    width, height, depth, color_type, compression, filtering, interlace = (
        struct.unpack(">IIBBBBB", chunks[0][1])
    )
    if (width, height) != (spec.width_px, spec.height_px):
        raise PaperAssetError(
            f"PNG size mismatch for {spec.asset_id}: {width}x{height}, expected "
            f"{spec.width_px}x{spec.height_px}"
        )
    if depth != 8:
        raise PaperAssetError(
            f"PNG must be 8-bit for {spec.asset_id}, observed {depth}-bit"
        )
    if color_type in {4, 6} or any(kind == b"tRNS" for kind, _ in chunks):
        raise PaperAssetError(f"PNG alpha/transparency is forbidden: {path}")
    if color_type != 2:
        raise PaperAssetError(
            f"PNG must be direct RGB (color type 2), observed {color_type}: {path}"
        )
    if (compression, filtering, interlace) != (0, 0, 0):
        raise PaperAssetError(f"PNG uses unsupported encoding flags: {path}")

    physical = [payload for kind, payload in chunks if kind == b"pHYs"]
    if len(physical) != 1 or len(physical[0]) != 9:
        raise PaperAssetError(f"PNG must carry exactly one pHYs DPI record: {path}")
    x_ppm, y_ppm, unit = struct.unpack(">IIB", physical[0])
    x_dpi, y_dpi = x_ppm * 0.0254, y_ppm * 0.0254
    if unit != 1 or max(abs(x_dpi - spec.dpi), abs(y_dpi - spec.dpi)) > DPI_TOLERANCE:
        raise PaperAssetError(
            f"PNG DPI mismatch for {spec.asset_id}: {x_dpi:.4f}x{y_dpi:.4f}, "
            f"expected {spec.dpi}"
        )
    if not any(kind == b"IDAT" and payload for kind, payload in chunks):
        raise PaperAssetError(f"PNG has no image payload: {path}")
    return {
        "width_px": width,
        "height_px": height,
        "dpi_x": round(x_dpi, 4),
        "dpi_y": round(y_dpi, 4),
        "color_model": "RGB",
        "components": 3,
        "bit_depth": depth,
        "opaque": True,
        "alpha_channel": False,
    }


def _parse_pdfinfo(path: Path, pdfinfo: Path) -> dict[str, str]:
    result = _run_tool([str(pdfinfo), str(path)], timeout_seconds=30)
    values: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    return values


def _pdf_image_rows(path: Path, pdfimages: Path) -> list[list[str]]:
    result = _run_tool([str(pdfimages), "-list", str(path)], timeout_seconds=30)
    rows: list[list[str]] = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 14 and fields[0].isdigit() and fields[1].isdigit():
            rows.append(fields)
    return rows


def _verify_pdf(
    path: Path, spec: AssetSpec, *, pdfinfo: Path, pdfimages: Path
) -> dict[str, Any]:
    if not path.read_bytes().startswith(b"%PDF-"):
        raise PaperAssetError(f"not a PDF file: {path}")
    info = _parse_pdfinfo(path, pdfinfo)
    if info.get("Pages") != "1":
        raise PaperAssetError(f"PDF must contain exactly one page: {path}")
    if info.get("Encrypted", "").lower() != "no":
        raise PaperAssetError(f"PDF must be unencrypted: {path}")
    page_size = info.get("Page size", "")
    match = re.match(r"^([0-9.]+) x ([0-9.]+) pts", page_size)
    if match is None:
        raise PaperAssetError(f"could not verify PDF page size: {path}")
    width_points, height_points = float(match.group(1)), float(match.group(2))
    expected_width = 72.0 * spec.width_px / spec.dpi
    expected_height = 72.0 * spec.height_px / spec.dpi
    if max(
        abs(width_points - expected_width), abs(height_points - expected_height)
    ) > PDF_POINT_TOLERANCE:
        raise PaperAssetError(
            f"PDF page size mismatch for {spec.asset_id}: {width_points:g}x"
            f"{height_points:g} pt, expected {expected_width:g}x{expected_height:g} pt"
        )

    rows = _pdf_image_rows(path, pdfimages)
    if len(rows) != 1:
        raise PaperAssetError(
            f"PDF must contain one flattened image, observed {len(rows)}: {path}"
        )
    row = rows[0]
    image_type = row[2].lower()
    width, height = int(row[3]), int(row[4])
    color, components, depth = row[5].lower(), int(row[6]), int(row[7])
    x_dpi, y_dpi = float(row[12]), float(row[13])
    if image_type != "image":
        raise PaperAssetError(f"PDF contains an alpha mask or non-image row: {path}")
    if (width, height) != (spec.width_px, spec.height_px):
        raise PaperAssetError(f"PDF embedded image dimensions drifted: {path}")
    if color != "rgb" or components != 3 or depth != 8:
        raise PaperAssetError(
            f"PDF must embed opaque 8-bit RGB, observed {color}/{components}/{depth}: "
            f"{path}"
        )
    if max(abs(x_dpi - spec.dpi), abs(y_dpi - spec.dpi)) > DPI_TOLERANCE:
        raise PaperAssetError(
            f"PDF embedded image DPI mismatch: {x_dpi:g}x{y_dpi:g}, expected "
            f"{spec.dpi}: {path}"
        )
    return {
        "width_px": width,
        "height_px": height,
        "dpi_x": x_dpi,
        "dpi_y": y_dpi,
        "page_width_points": width_points,
        "page_height_points": height_points,
        "color_model": "RGB",
        "components": components,
        "bit_depth": depth,
        "opaque": True,
        "alpha_channel": False,
    }


def _validate_output(
    path: Path,
    spec: AssetSpec,
    *,
    pdfinfo: Path | None = None,
    pdfimages: Path | None = None,
) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise PaperAssetError(f"missing or empty export: {path}")
    if spec.output_format == "png":
        return _verify_png(path, spec)
    if pdfinfo is None or pdfimages is None:
        raise PaperAssetError("PDF verification requires pdfinfo and pdfimages")
    return _verify_pdf(path, spec, pdfinfo=pdfinfo, pdfimages=pdfimages)


def _conversion_environment() -> dict[str, str]:
    return {
        **os.environ,
        "MAGICK_MEMORY_LIMIT": "512MiB",
        "MAGICK_MAP_LIMIT": "1GiB",
        "MAGICK_DISK_LIMIT": "2GiB",
        "MAGICK_THREAD_LIMIT": "2",
    }


def _convert_svg(spec: AssetSpec, *, magick: Path, temporary_path: Path) -> None:
    output_prefix = "PNG24" if spec.output_format == "png" else "PDF"
    argv = [
        str(magick),
        "-density",
        f"{spec.render_density_dpi:.8g}",
        str(spec.source_path),
        "-background",
        spec.background,
        "-alpha",
        "remove",
        "-alpha",
        "off",
        "-filter",
        "Lanczos",
        "-resize",
        f"{spec.width_px}x{spec.height_px}!",
        "-units",
        "PixelsPerInch",
        "-density",
        str(spec.dpi),
        "-colorspace",
        "sRGB",
        "-depth",
        "8",
        "-type",
        "TrueColor",
        f"{output_prefix}:{temporary_path}",
    ]
    _run_tool(argv, env=_conversion_environment())


def _atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary_path = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def export_assets(manifest_path: Path) -> Path:
    manifest = load_manifest(manifest_path)
    magick = _require_tool("magick")
    needs_pdf = any(spec.output_format == "pdf" for spec in manifest.assets)
    pdfinfo = _require_tool("pdfinfo") if needs_pdf else None
    pdfimages = _require_tool("pdfimages") if needs_pdf else None

    tools: dict[str, Any] = {
        "converter": _tool_identity(magick, "-version"),
        "python": {
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
    }
    if needs_pdf:
        assert pdfinfo is not None and pdfimages is not None
        tools["pdfinfo"] = _tool_identity(pdfinfo, "-v")
        tools["pdfimages"] = _tool_identity(pdfimages, "-v")

    asset_records: list[dict[str, Any]] = []
    for spec in manifest.assets:
        source_hash_before = _sha256(spec.source_path)
        spec.output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=spec.output_path.parent,
            prefix=f".{spec.output_path.stem}.",
            suffix=spec.output_path.suffix,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        try:
            _convert_svg(spec, magick=magick, temporary_path=temporary_path)
            observed = _validate_output(
                temporary_path,
                spec,
                pdfinfo=pdfinfo,
                pdfimages=pdfimages,
            )
            if _sha256(spec.source_path) != source_hash_before:
                raise PaperAssetError(
                    f"SVG source changed during export: {spec.source_path}"
                )
            os.replace(temporary_path, spec.output_path)
        finally:
            temporary_path.unlink(missing_ok=True)

        asset_records.append(
            {
                "id": spec.asset_id,
                "source": {
                    "path": spec.source_label,
                    "sha256": source_hash_before,
                    "width": spec.source_width,
                    "height": spec.source_height,
                },
                "export": {
                    "path": spec.output_label,
                    "format": spec.output_format,
                    "sha256": _sha256(spec.output_path),
                    "byte_size": spec.output_path.stat().st_size,
                    **observed,
                },
                "contract": {
                    "width_px": spec.width_px,
                    "height_px": spec.height_px,
                    "dpi": spec.dpi,
                    "background": spec.background,
                    "render_density_dpi": round(spec.render_density_dpi, 8),
                    "alpha_policy": "flatten_then_remove",
                    "color_model": "sRGB",
                    "bit_depth": 8,
                },
            }
        )

    record = {
        "schema_version": RECORD_SCHEMA_VERSION,
        "status": "accepted",
        "manifest": {
            "path": manifest.path.name,
            "sha256": _sha256(manifest.path),
        },
        "resource_limits": {
            "memory": "512MiB",
            "map": "1GiB",
            "disk": "2GiB",
            "threads": 2,
        },
        "tools": tools,
        "assets": asset_records,
    }
    _atomic_json_write(manifest.record_path, record)
    errors = verify_exports(manifest.path)
    if errors:
        raise PaperAssetError("export verification failed:\n- " + "\n- ".join(errors))
    return manifest.record_path


def _record_rows_by_id(record: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    rows = record.get("assets")
    if not isinstance(rows, list):
        return {}, ["record assets must be a list"]
    by_id: dict[str, Any] = {}
    errors: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            errors.append(f"record asset[{index}] is malformed")
            continue
        if row["id"] in by_id:
            errors.append(f"record repeats asset id {row['id']}")
            continue
        by_id[row["id"]] = row
    return by_id, errors


def verify_exports(manifest_path: Path) -> list[str]:
    try:
        manifest = load_manifest(manifest_path)
    except PaperAssetError as exc:
        return [str(exc)]
    if not manifest.record_path.is_file():
        return [f"missing export record: {manifest.record_path}"]
    try:
        record = _load_json_object(manifest.record_path)
    except PaperAssetError as exc:
        return [str(exc)]

    errors: list[str] = []
    if record.get("schema_version") != RECORD_SCHEMA_VERSION:
        errors.append("record schema_version mismatch")
    if record.get("status") != "accepted":
        errors.append("record status is not accepted")
    manifest_record = record.get("manifest")
    if not isinstance(manifest_record, dict):
        errors.append("record manifest provenance is missing")
    else:
        if manifest_record.get("sha256") != _sha256(manifest.path):
            errors.append("manifest SHA-256 mismatch")
        if manifest_record.get("path") != manifest.path.name:
            errors.append("manifest path provenance mismatch")
    tools = record.get("tools")
    if not isinstance(tools, dict) or not isinstance(tools.get("converter"), dict):
        errors.append("converter tool identity is missing")
    else:
        converter = tools["converter"]
        if not converter.get("executable") or not converter.get("version"):
            errors.append("converter tool identity is incomplete")

    needs_pdf = any(spec.output_format == "pdf" for spec in manifest.assets)
    pdfinfo: Path | None = None
    pdfimages: Path | None = None
    if needs_pdf:
        for name in ("pdfinfo", "pdfimages"):
            identity = tools.get(name) if isinstance(tools, dict) else None
            if not isinstance(identity, dict) or not identity.get("version"):
                errors.append(f"{name} tool identity is missing")
        try:
            pdfinfo = _require_tool("pdfinfo")
            pdfimages = _require_tool("pdfimages")
        except PaperAssetError as exc:
            errors.append(str(exc))

    rows_by_id, row_errors = _record_rows_by_id(record)
    errors.extend(row_errors)
    expected_ids = {spec.asset_id for spec in manifest.assets}
    if set(rows_by_id) != expected_ids:
        errors.append(
            "record asset ids differ from manifest: "
            f"record={sorted(rows_by_id)}, manifest={sorted(expected_ids)}"
        )

    for spec in manifest.assets:
        row = rows_by_id.get(spec.asset_id)
        if not isinstance(row, dict):
            continue
        source = row.get("source")
        export = row.get("export")
        contract = row.get("contract")
        if not isinstance(source, dict):
            errors.append(f"{spec.asset_id}: source record is missing")
        else:
            if source.get("path") != spec.source_label:
                errors.append(f"{spec.asset_id}: source path record mismatch")
            if source.get("sha256") != _sha256(spec.source_path):
                errors.append(f"{spec.asset_id}: source SHA-256 mismatch")
        if not isinstance(export, dict):
            errors.append(f"{spec.asset_id}: export record is missing")
            continue
        if export.get("path") != spec.output_label:
            errors.append(f"{spec.asset_id}: export path record mismatch")
        if export.get("format") != spec.output_format:
            errors.append(f"{spec.asset_id}: export format record mismatch")
        if not spec.output_path.is_file():
            errors.append(f"{spec.asset_id}: export is missing: {spec.output_path}")
            continue
        if export.get("sha256") != _sha256(spec.output_path):
            errors.append(f"{spec.asset_id}: export SHA-256 mismatch")
        if export.get("byte_size") != spec.output_path.stat().st_size:
            errors.append(f"{spec.asset_id}: export byte size mismatch")
        expected_contract = {
            "width_px": spec.width_px,
            "height_px": spec.height_px,
            "dpi": spec.dpi,
            "background": spec.background,
            "render_density_dpi": round(spec.render_density_dpi, 8),
            "alpha_policy": "flatten_then_remove",
            "color_model": "sRGB",
            "bit_depth": 8,
        }
        if contract != expected_contract:
            errors.append(f"{spec.asset_id}: conversion contract record mismatch")
        try:
            observed = _validate_output(
                spec.output_path,
                spec,
                pdfinfo=pdfinfo,
                pdfimages=pdfimages,
            )
        except (OSError, PaperAssetError, ValueError) as exc:
            errors.append(f"{spec.asset_id}: {exc}")
            continue
        for key, expected in observed.items():
            if export.get(key) != expected:
                errors.append(f"{spec.asset_id}: recorded {key} does not match bytes")
    return errors


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("export", "verify"))
    parser.add_argument("manifest", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "export":
        try:
            record_path = export_assets(args.manifest)
        except PaperAssetError as exc:
            parser.exit(2, f"paper asset export failed: {exc}\n")
        print(f"paper asset export accepted: {record_path}")
        return 0
    errors = verify_exports(args.manifest)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"paper asset verification accepted: {args.manifest.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from research_experiments.paper_runner_suite.export_paper_assets import (
    PaperAssetError,
    export_assets,
    load_manifest,
    verify_exports,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "research_experiments"
    / "paper_runner_suite"
    / "export_paper_assets.py"
)
DEFAULT_MANIFEST = (
    ROOT
    / "research_experiments"
    / "paper_runner_suite"
    / "paper_asset_export_manifest.json"
)
HAS_EXPORT_TOOLS = all(
    shutil.which(name) for name in ("magick", "pdfinfo", "pdfimages")
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture_manifest(tmp_path: Path, *, include_pdf: bool) -> Path:
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    (source_dir / "tiny.svg").write_text(
        """<svg xmlns="http://www.w3.org/2000/svg" width="64" height="32" viewBox="0 0 64 32">
<rect width="64" height="32" fill="#ffffff"/>
<rect x="4" y="4" width="56" height="24" fill="#2563eb"/>
</svg>
""",
        encoding="utf-8",
    )
    assets = [
        {
            "id": "tiny_png",
            "source": "tiny.svg",
            "output": "tiny.png",
            "format": "png",
            "width_px": 128,
            "height_px": 64,
            "dpi": 300,
            "background": "#ffffff",
        }
    ]
    if include_pdf:
        assets.append(
            {
                "id": "tiny_pdf",
                "source": "tiny.svg",
                "output": "tiny.pdf",
                "format": "pdf",
                "width_px": 128,
                "height_px": 64,
                "dpi": 300,
                "background": "#ffffff",
            }
        )
    manifest = {
        "schema_version": 1,
        "source_root": "sources",
        "output_root": "exports",
        "record": "record.json",
        "assets": assets,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_checked_in_manifest_is_shared_and_declares_venue_geometry() -> None:
    manifest = load_manifest(DEFAULT_MANIFEST)
    ids = {asset.asset_id for asset in manifest.assets}
    assert any(asset_id.startswith("world_tubes_") for asset_id in ids)
    assert any(asset_id.startswith("worldfoam_") for asset_id in ids)
    assert "worldfoam_material_family_loss" in ids
    assert "worldfoam_foundation_error_summary" in ids
    assert all("placeholder" not in asset.source_label for asset in manifest.assets)
    assert all("placeholder" not in asset.output_label for asset in manifest.assets)
    assert all(asset.dpi == 300 for asset in manifest.assets)
    assert all(asset.output_format in {"png", "pdf"} for asset in manifest.assets)


@pytest.mark.skipif(not HAS_EXPORT_TOOLS, reason="paper asset tools unavailable")
def test_shared_export_records_hashes_tools_and_opaque_rgb_bytes(
    tmp_path: Path,
) -> None:
    manifest_path = _write_fixture_manifest(tmp_path, include_pdf=True)
    record_path = export_assets(manifest_path)
    record = json.loads(record_path.read_text(encoding="utf-8"))

    assert record["status"] == "accepted"
    assert record["tools"]["converter"]["version"].startswith("Version: ImageMagick")
    assert record["tools"]["pdfinfo"]["version"].startswith("pdfinfo version")
    assert record["tools"]["pdfimages"]["version"].startswith("pdfimages version")
    assert verify_exports(manifest_path) == []

    for row in record["assets"]:
        source = tmp_path / "sources" / row["source"]["path"]
        output = tmp_path / "exports" / row["export"]["path"]
        assert row["source"]["sha256"] == _sha256(source)
        assert row["export"]["sha256"] == _sha256(output)
        assert row["export"]["color_model"] == "RGB"
        assert row["export"]["bit_depth"] == 8
        assert row["export"]["opaque"] is True
        assert row["export"]["alpha_channel"] is False
        assert row["export"]["width_px"] == 128
        assert row["export"]["height_px"] == 64


@pytest.mark.skipif(not HAS_EXPORT_TOOLS, reason="paper asset tools unavailable")
def test_verifier_rejects_export_tampering(tmp_path: Path) -> None:
    manifest_path = _write_fixture_manifest(tmp_path, include_pdf=False)
    export_assets(manifest_path)
    output = tmp_path / "exports" / "tiny.png"
    output.write_bytes(output.read_bytes() + b"tamper")

    errors = verify_exports(manifest_path)
    assert any("export SHA-256 mismatch" in error for error in errors)
    assert any("trailing bytes after IEND" in error for error in errors)


@pytest.mark.skipif(not HAS_EXPORT_TOOLS, reason="paper asset tools unavailable")
def test_verifier_rejects_alpha_even_if_record_hash_is_rewritten(
    tmp_path: Path,
) -> None:
    manifest_path = _write_fixture_manifest(tmp_path, include_pdf=False)
    record_path = export_assets(manifest_path)
    output = tmp_path / "exports" / "tiny.png"
    subprocess.run(
        [
            shutil.which("magick") or "magick",
            "-size",
            "128x64",
            "xc:none",
            f"PNG32:{output}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["assets"][0]["export"]["sha256"] = _sha256(output)
    record["assets"][0]["export"]["byte_size"] = output.stat().st_size
    record_path.write_text(json.dumps(record), encoding="utf-8")

    errors = verify_exports(manifest_path)
    assert any("alpha/transparency is forbidden" in error for error in errors)


def test_manifest_rejects_unknown_keys_and_aspect_drift(tmp_path: Path) -> None:
    manifest_path = _write_fixture_manifest(tmp_path, include_pdf=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["assets"][0]["height_px"] = 65
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PaperAssetError, match="changes aspect ratio"):
        load_manifest(manifest_path)

    manifest["assets"][0]["height_px"] = 64
    manifest["assets"][0]["typo_dpi"] = 300
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PaperAssetError, match=r"unknown asset\[0\] keys"):
        load_manifest(manifest_path)


@pytest.mark.skipif(not HAS_EXPORT_TOOLS, reason="paper asset tools unavailable")
def test_cli_exports_then_verifies_without_regeneration(tmp_path: Path) -> None:
    manifest_path = _write_fixture_manifest(tmp_path, include_pdf=False)
    for command in ("export", "verify"):
        subprocess.run(
            [sys.executable, str(SCRIPT), command, str(manifest_path)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

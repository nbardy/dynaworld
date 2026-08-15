from __future__ import annotations

import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from research_experiments.paper_runner_suite.generate_world_tubes_concept_figures import (
    FIGURE_FILENAMES,
    PROJECTIVE_REQUIRED_LABELS,
    SYSTEM_REQUIRED_LABELS,
    expected_figures,
    verify_figure_dir,
    write_figures,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "research_experiments"
    / "paper_runner_suite"
    / "generate_world_tubes_concept_figures.py"
)
SVG_NAMESPACE = "{http://www.w3.org/2000/svg}"
FULL_TEXT_WIDTH_INCHES = 6.5
MIN_EFFECTIVE_FONT_POINTS = 7.0


def _effective_text_sizes_points(source: str) -> list[float]:
    root = ET.fromstring(source)
    width = float(root.attrib["width"])
    view_box = tuple(float(value) for value in root.attrib["viewBox"].split())
    assert view_box[:2] == (0.0, 0.0)
    assert view_box[2] == width

    css = root.findtext(f"{SVG_NAMESPACE}style") or ""
    font_sizes = {
        class_name: float(font_size)
        for class_name, font_size in re.findall(
            r"\.([\w-]+)\s*\{[^}]*font-size:\s*([\d.]+)px",
            css,
        )
    }
    printed_sizes = []
    for text_node in root.iter(f"{SVG_NAMESPACE}text"):
        class_name = text_node.attrib.get("class")
        assert class_name in font_sizes, f"text has no calculable font size: {class_name}"
        printed_sizes.append(
            font_sizes[class_name] * FULL_TEXT_WIDTH_INCHES * 72.0 / width
        )
    assert printed_sizes
    return printed_sizes


def test_concept_figures_are_deterministic_valid_semantic_svgs(tmp_path: Path) -> None:
    expected = expected_figures()
    first_paths = write_figures(tmp_path / "first")
    first_bytes = {path.name: path.read_bytes() for path in first_paths}
    second_paths = write_figures(tmp_path / "second")
    second_bytes = {path.name: path.read_bytes() for path in second_paths}

    assert tuple(path.name for path in first_paths) == FIGURE_FILENAMES
    assert tuple(path.name for path in second_paths) == FIGURE_FILENAMES
    assert first_bytes == second_bytes
    assert first_bytes == {
        filename: source.encode("utf-8") for filename, source in expected.items()
    }
    assert verify_figure_dir(tmp_path / "first") == []
    assert verify_figure_dir(tmp_path / "second") == []

    required_by_figure = {
        FIGURE_FILENAMES[0]: SYSTEM_REQUIRED_LABELS,
        FIGURE_FILENAMES[1]: PROJECTIVE_REQUIRED_LABELS,
    }
    for filename, source in expected.items():
        root = ET.fromstring(source)
        assert root.tag == f"{SVG_NAMESPACE}svg"
        assert root.findtext(f"{SVG_NAMESPACE}title")
        assert root.findtext(f"{SVG_NAMESPACE}desc")
        for label in required_by_figure[filename]:
            assert label in source
        assert "PSNR" not in source
        assert "SSIM" not in source
        assert min(_effective_text_sizes_points(source)) >= MIN_EFFECTIVE_FONT_POINTS


def test_concept_figure_verifier_rejects_missing_and_exact_byte_drift(
    tmp_path: Path,
) -> None:
    write_figures(tmp_path)
    missing_path = tmp_path / FIGURE_FILENAMES[1]
    missing_path.unlink()
    assert any("missing figure" in error for error in verify_figure_dir(tmp_path))

    write_figures(tmp_path)
    drifted_path = tmp_path / FIGURE_FILENAMES[0]
    drifted_path.write_bytes(drifted_path.read_bytes().replace(b"\n", b"\r\n"))
    errors = verify_figure_dir(tmp_path)
    assert any("figure bytes drifted" in error for error in errors)


def test_concept_figure_cli_runs_with_isolated_stdlib(tmp_path: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(SCRIPT),
            "--out-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert verify_figure_dir(tmp_path) == []

from __future__ import annotations

import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from research_experiments.paper_runner_suite.generate_worldfoam_concept_figures import (
    ATLAS_REQUIRED_LABELS,
    FIGURE_FILENAMES,
    FORBIDDEN_RESULT_TOKENS,
    REPRESENTATION_REQUIRED_LABELS,
    expected_figures,
    verify_figure_dir,
    write_figures,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "research_experiments"
    / "paper_runner_suite"
    / "generate_worldfoam_concept_figures.py"
)
PAPER = ROOT / "research_notes" / "worldfoam_paper" / "WORLD_FOAM_ICLR_MAIN_DRAFT.md"
BIBLIOGRAPHY = ROOT / "research_notes" / "worldfoam_paper" / "WORLD_FOAM_REFERENCES.bib"
SVG_NAMESPACE = "{http://www.w3.org/2000/svg}"


def test_worldfoam_figures_are_deterministic_result_free_semantic_svgs(
    tmp_path: Path,
) -> None:
    expected = expected_figures()
    first_paths = write_figures(tmp_path / "first")
    second_paths = write_figures(tmp_path / "second")
    first_bytes = {path.name: path.read_bytes() for path in first_paths}
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
        FIGURE_FILENAMES[0]: REPRESENTATION_REQUIRED_LABELS,
        FIGURE_FILENAMES[1]: ATLAS_REQUIRED_LABELS,
    }
    for filename, source in expected.items():
        root = ET.fromstring(source)
        assert root.tag == f"{SVG_NAMESPACE}svg"
        assert root.attrib["width"] == "1200"
        assert root.findtext(f"{SVG_NAMESPACE}title")
        assert root.findtext(f"{SVG_NAMESPACE}desc")
        for label in required_by_figure[filename]:
            assert label in source
        for token in FORBIDDEN_RESULT_TOKENS:
            assert token not in source

        font_sizes = tuple(
            int(value) for value in re.findall(r"font-size:\s*(\d+)px", source)
        )
        assert font_sizes
        assert min(font_sizes) >= 18


def test_worldfoam_figure_verifier_rejects_missing_and_byte_drift(
    tmp_path: Path,
) -> None:
    write_figures(tmp_path)
    (tmp_path / FIGURE_FILENAMES[1]).unlink()
    assert any("missing figure" in error for error in verify_figure_dir(tmp_path))

    write_figures(tmp_path)
    drifted = tmp_path / FIGURE_FILENAMES[0]
    drifted.write_bytes(drifted.read_bytes().replace(b"\n", b"\r\n"))
    assert any("figure bytes drifted" in error for error in verify_figure_dir(tmp_path))


def test_worldfoam_concept_figure_cli_runs_with_isolated_stdlib(
    tmp_path: Path,
) -> None:
    subprocess.run(
        [sys.executable, "-I", "-S", str(SCRIPT), "--out-dir", str(tmp_path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(SCRIPT),
            "--verify-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_worldfoam_submission_source_links_figures_and_resolves_citations() -> None:
    paper = PAPER.read_text(encoding="utf-8")
    bibliography = BIBLIOGRAPHY.read_text(encoding="utf-8")

    for filename in FIGURE_FILENAMES:
        assert f"research_notes/worldfoam_paper/figures/{filename}" in paper
    for gate in ("G0", "G1", "G2", "G3", "G4", "G5", "G6"):
        assert gate in paper
    assert "all 21 required measured" in paper
    assert "same-representation sequential per-frame replay" in paper

    cited_keys = set(re.findall(r"@([A-Za-z0-9_:-]+)", paper))
    bibliography_keys = set(
        re.findall(r"^@[A-Za-z]+\{([^,]+),", bibliography, flags=re.MULTILINE)
    )
    assert cited_keys
    assert cited_keys <= bibliography_keys

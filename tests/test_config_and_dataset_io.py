from __future__ import annotations

from pathlib import Path

import pytest

from config_utils import load_config_file, path_or_none, require_config_keys
from download_utils import fetch_json_url, validate_http_url
from json_io import load_json, load_jsonl_objects
from multicam_val import read_jsonl as read_multicam_jsonl
from youtube_curated_spans import read_jsonl as read_curated_jsonl
from youtube_ingest import read_jsonl as read_ingest_jsonl


def test_load_config_file_reports_jsonc_path(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.jsonc"
    config_path.write_text('{"model": true,,}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"Invalid JSONC config .*bad\.jsonc"):
        load_config_file(config_path)


def test_require_config_keys_reports_missing_section_keys() -> None:
    with pytest.raises(KeyError, match=r"Missing required train config key\(s\): lr, steps"):
        require_config_keys("train", {"seed": 1}, ("lr", "steps"))


def test_path_or_none_preserves_none_and_converts_paths() -> None:
    assert path_or_none(None) is None
    assert path_or_none("outputs/run.json") == Path("outputs/run.json")


def test_load_json_reads_pathlike_values(tmp_path: Path) -> None:
    path = tmp_path / "payload.json"
    path.write_text('{"a": 1, "b": [2]}\n', encoding="utf-8")

    assert load_json(path) == {"a": 1, "b": [2]}


def test_load_jsonl_objects_reports_file_and_line(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"ok": 1}\n\n{"also_ok": 2}\n', encoding="utf-8")
    assert load_jsonl_objects(manifest) == [{"ok": 1}, {"also_ok": 2}]

    bad_manifest = tmp_path / "bad.jsonl"
    bad_manifest.write_text('{"ok": 1}\n[]\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"Expected object on .*bad\.jsonl:2"):
        load_jsonl_objects(bad_manifest)


@pytest.mark.parametrize(
    "reader",
    [read_ingest_jsonl, read_curated_jsonl, read_multicam_jsonl],
)
def test_jsonl_readers_report_file_and_line(reader, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"ok": 1}\n\n{"broken": }\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"manifest\.jsonl:3"):
        reader(manifest)


def test_download_url_validation_rejects_non_http_schemes() -> None:
    assert validate_http_url("https://example.com/data.zip") == "https://example.com/data.zip"

    with pytest.raises(ValueError, match="http\\(s\\) URLs"):
        validate_http_url("file:///tmp/data.zip")


class _JsonResponse:
    text = "{broken"

    def raise_for_status(self) -> None:
        return None


def test_fetch_json_url_reports_url_on_malformed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, **_kwargs) -> _JsonResponse:
        assert url == "https://example.com/release.json"
        return _JsonResponse()

    monkeypatch.setattr("download_utils.requests.get", fake_get)

    with pytest.raises(ValueError, match=r"Invalid JSON response from 'https://example\.com/release\.json'"):
        fetch_json_url("https://example.com/release.json")

from __future__ import annotations

from pathlib import Path

import pytest

from config_utils import load_config_file
from download_utils import fetch_json_url, validate_http_url
from multicam_val import read_jsonl as read_multicam_jsonl
from youtube_curated_spans import read_jsonl as read_curated_jsonl
from youtube_ingest import read_jsonl as read_ingest_jsonl


def test_load_config_file_reports_jsonc_path(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.jsonc"
    config_path.write_text('{"model": true,,}\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"Invalid JSONC config .*bad\.jsonc"):
        load_config_file(config_path)


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

from __future__ import annotations

from pathlib import Path

import pytest
import requests

from config_utils import load_config_file, path_or_none, require_config_keys
from download_utils import download_url, fetch_json_url, validate_http_url
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


class _DownloadResponse:
    def __init__(self, body: bytes, *, status_code: int, headers: dict[str, str]) -> None:
        self.body = body
        self.status_code = status_code
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, *, chunk_size: int):
        assert chunk_size == 1024 * 1024
        yield self.body


class _InterruptedDownloadResponse(_DownloadResponse):
    def iter_content(self, *, chunk_size: int):
        yield self.body
        raise requests.ConnectionError("transient stream failure")


def test_download_url_resumes_partial_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "scene.zip"
    output.with_suffix(".zip.part").write_bytes(b"first-")

    def fake_get(_url: str, **kwargs):
        assert kwargs["headers"]["Range"] == "bytes=6-"
        return _DownloadResponse(b"second", status_code=206, headers={"Content-Range": "bytes 6-11/12"})

    monkeypatch.setattr("download_utils.requests.get", fake_get)
    assert download_url(
        "https://example.com/scene.zip",
        output,
        overwrite=True,
        user_agent="test",
    )
    assert output.read_bytes() == b"first-second"


def test_download_url_restarts_when_server_ignores_range(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "scene.zip"
    output.with_suffix(".zip.part").write_bytes(b"stale")
    monkeypatch.setattr(
        "download_utils.requests.get",
        lambda *_args, **_kwargs: _DownloadResponse(b"fresh", status_code=200, headers={}),
    )

    download_url("https://example.com/scene.zip", output, overwrite=True, user_agent="test")

    assert output.read_bytes() == b"fresh"


def test_download_url_retries_and_resumes_transient_stream_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "scene.zip"
    calls = 0

    def fake_get(_url: str, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            assert "Range" not in kwargs["headers"]
            return _InterruptedDownloadResponse(b"first", status_code=200, headers={})
        assert kwargs["headers"]["Range"] == "bytes=5-"
        return _DownloadResponse(b"second", status_code=206, headers={"Content-Range": "bytes 5-10/11"})

    monkeypatch.setattr("download_utils.requests.get", fake_get)
    monkeypatch.setattr("download_utils.time.sleep", lambda _seconds: None)

    download_url(
        "https://example.com/scene.zip",
        output,
        overwrite=True,
        user_agent="test",
        max_attempts=2,
    )

    assert output.read_bytes() == b"firstsecond"

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import requests


def validate_http_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Only http(s) URLs are supported for dataset downloads: {url!r}")
    if not parsed.netloc:
        raise ValueError(f"Dataset download URL is missing a host: {url!r}")
    return url


def fetch_json_url(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    response = requests.get(
        validate_http_url(url),
        headers=dict(headers or {}),
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    try:
        payload = json.loads(response.text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON response from {url!r}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object from {url!r}, got {type(payload).__name__}.")
    return payload


def download_url(
    url: str,
    output_path: Path,
    *,
    overwrite: bool,
    user_agent: str,
    timeout_seconds: float = 60.0,
) -> bool:
    if output_path.exists() and not overwrite:
        print(f"Already exists: {output_path}")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    with requests.get(
        validate_http_url(url),
        headers={"User-Agent": user_agent},
        stream=True,
        timeout=timeout_seconds,
    ) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    tmp_path.replace(output_path)
    print(f"Downloaded: {output_path}")
    return True

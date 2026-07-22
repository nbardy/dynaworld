from __future__ import annotations

import json
import time
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
    max_attempts: int = 5,
) -> bool:
    if output_path.exists() and not overwrite:
        print(f"Already exists: {output_path}")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    for attempt in range(1, max_attempts + 1):
        resume_from = tmp_path.stat().st_size if tmp_path.exists() else 0
        headers = {"User-Agent": user_agent}
        if resume_from:
            headers["Range"] = f"bytes={resume_from}-"
        try:
            with requests.get(
                validate_http_url(url),
                headers=headers,
                stream=True,
                timeout=timeout_seconds,
            ) as response:
                status_code = int(response.status_code)
                if resume_from and status_code == 416:
                    total = str(response.headers.get("Content-Range", "")).removeprefix("bytes */")
                    if total.isdigit() and int(total) == resume_from:
                        tmp_path.replace(output_path)
                        print(f"Downloaded: {output_path}")
                        return True
                response.raise_for_status()
                append = resume_from > 0 and status_code == 206
                if append:
                    content_range = str(response.headers.get("Content-Range", ""))
                    if not content_range.startswith(f"bytes {resume_from}-"):
                        raise RuntimeError(
                            f"Download server returned an invalid resume range for {url!r}: {content_range!r}"
                        )
                with tmp_path.open("ab" if append else "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            break
        except requests.RequestException:
            if attempt == max_attempts:
                raise
            print(f"Download interrupted; resuming attempt {attempt + 1}/{max_attempts}: {output_path}")
            time.sleep(min(2 ** (attempt - 1), 8))
    tmp_path.replace(output_path)
    print(f"Downloaded: {output_path}")
    return True

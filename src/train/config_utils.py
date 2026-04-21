from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any


def strip_jsonc_comments(text: str) -> str:
    output = []
    index = 0
    in_string = False
    escaped = False

    while index < len(text):
        char = text[index]

        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            output.append(char)
            index += 1
            continue

        if char == "/" and index + 1 < len(text):
            next_char = text[index + 1]
            if next_char == "/":
                index += 2
                while index < len(text) and text[index] not in "\r\n":
                    index += 1
                continue
            if next_char == "*":
                index += 2
                while index + 1 < len(text) and not (text[index] == "*" and text[index + 1] == "/"):
                    if text[index] in "\r\n":
                        output.append(text[index])
                    index += 1
                if index + 1 >= len(text):
                    raise ValueError("Unterminated block comment in JSONC config.")
                index += 2
                continue

        output.append(char)
        index += 1

    return "".join(output)


def load_config_file(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    data = json.loads(strip_jsonc_comments(config_path.read_text()))
    if not isinstance(data, dict):
        raise ValueError(f"Expected config object in {config_path}, got {type(data).__name__}.")
    return data


def require_config_sections(config: dict[str, Any], sections: tuple[str, ...]) -> None:
    for section in sections:
        if section not in config:
            raise KeyError(f"Missing required config section: {section}")


def path_or_none(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    return Path(value)


def resolved_config(config: dict[str, Any], sections: tuple[str, ...]) -> dict[str, Any]:
    cfg = deepcopy(config)
    require_config_sections(cfg, sections)
    return cfg


def apply_defaults(section: dict[str, Any], defaults: Mapping[str, Any]) -> None:
    for key, value in defaults.items():
        if key not in section:
            section[key] = deepcopy(value)


def select_keys(values: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {key: values[key] for key in keys}


def format_key_values(values: Mapping[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in values.items())


def serialize_config_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: serialize_config_value(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_config_value(inner) for inner in value]
    return value

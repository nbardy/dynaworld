from __future__ import annotations

import hashlib
import json
import math
import shutil
import struct
from pathlib import Path
from typing import Any, Callable, Mapping


FORMAT = "world_tubes_projective_trace_atlas_v1"
MAGIC = b"WTATLAS1\n"
LOGICAL_PAYLOAD_DEFINITION = (
    "cumulative logical tensor-element work volume; replay recounts shared "
    "color/opacity at every frame and compiled covers trace tensors only; "
    "this proxy excludes topology and is neither retained storage nor peak memory"
)
RETAINED_STORAGE_DEFINITION = (
    "actual serialized route-owned evaluator state; compiled bytes include "
    "every retained tensor and discrete cell topology; the shared frozen-world "
    "checkpoint is reported separately and excluded from both route totals"
)
REPLAY_STORAGE_REASON = (
    "replay recomputes one-frame projection/binning and retains no lowered "
    "per-frame evaluator between calls"
)
ROUTE_MEMORY_DEFINITION = (
    "absolute device allocator peaks sampled at 5 ms and synchronized phase "
    "boundaries during the correctness routes; increments are relative to a "
    "garbage-collected, cache-cleared route baseline"
)
ROUTE_MEMORY_MEASUREMENT_SOURCE = (
    "correctness_pass_route_scoped_device_allocator_samples_v1"
)
TENSOR_NAMES = (
    "coeffs",
    "opacity",
    "color",
    "opacity_time_coeffs",
    "spatial_precision_uv",
    "depth_affine_uv",
)
REQUIRED_TENSOR_NAMES = frozenset(("coeffs", "opacity", "color"))
DTYPE_BYTES = {
    "float16": 2,
    "float32": 4,
    "float64": 8,
    "int8": 1,
    "uint8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "bool": 1,
}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_number(value: Any, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _validate_integer_sequence(
    value: Any,
    *,
    name: str,
    length: int,
) -> None:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} integers")
    for index, item in enumerate(value):
        _integer(item, name=f"{name}[{index}]")


def validate_topology(
    topology: Mapping[str, Any],
    *,
    trace_count: int,
    cell_count: int,
) -> None:
    expected_keys = {
        "source_window_indices",
        "source_primitive_ids",
        "active_start",
        "active_stop",
        "cells",
    }
    if set(topology) != expected_keys:
        raise ValueError("retained atlas topology fields are incomplete")
    for name in (
        "source_window_indices",
        "source_primitive_ids",
        "active_start",
        "active_stop",
    ):
        _validate_integer_sequence(
            topology[name],
            name=name,
            length=trace_count,
        )
    for index, (start, stop) in enumerate(
        zip(topology["active_start"], topology["active_stop"], strict=True)
    ):
        if stop <= start:
            raise ValueError(
                f"retained atlas active interval {index} is not ordered"
            )
    cells = topology["cells"]
    if not isinstance(cells, list) or len(cells) != cell_count:
        raise ValueError(
            f"retained atlas topology must contain exactly {cell_count} cells"
        )
    cell_keys = {
        "tile_u",
        "tile_v",
        "start",
        "stop",
        "primitive_ids",
        "ordered_primitive_ids",
        "depth_intervals",
        "fallback",
        "fallback_reasons",
    }
    for cell_index, cell in enumerate(cells):
        if not isinstance(cell, Mapping) or set(cell) != cell_keys:
            raise ValueError(
                f"retained atlas cell {cell_index} fields are incomplete"
            )
        for name in ("tile_u", "tile_v", "start", "stop"):
            _integer(cell[name], name=f"cells[{cell_index}].{name}")
        if cell["stop"] <= cell["start"]:
            raise ValueError(
                f"retained atlas cell {cell_index} interval is not ordered"
            )
        primitive_ids = cell["primitive_ids"]
        ordered_ids = cell["ordered_primitive_ids"]
        intervals = cell["depth_intervals"]
        if (
            not isinstance(primitive_ids, list)
            or not isinstance(ordered_ids, list)
            or not isinstance(intervals, list)
            or len(ordered_ids) != len(primitive_ids)
            or len(intervals) != len(primitive_ids)
        ):
            raise ValueError(
                f"retained atlas cell {cell_index} trace lists disagree"
            )
        _validate_integer_sequence(
            primitive_ids,
            name=f"cells[{cell_index}].primitive_ids",
            length=len(primitive_ids),
        )
        _validate_integer_sequence(
            ordered_ids,
            name=f"cells[{cell_index}].ordered_primitive_ids",
            length=len(ordered_ids),
        )
        if sorted(primitive_ids) != sorted(ordered_ids):
            raise ValueError(
                f"retained atlas cell {cell_index} ordering is not a permutation"
            )
        for interval_index, interval in enumerate(intervals):
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError(
                    f"retained atlas cell {cell_index} depth interval is invalid"
                )
            lower = _finite_number(
                interval[0],
                name=(
                    f"cells[{cell_index}].depth_intervals[{interval_index}][0]"
                ),
            )
            upper = _finite_number(
                interval[1],
                name=(
                    f"cells[{cell_index}].depth_intervals[{interval_index}][1]"
                ),
            )
            if upper < lower:
                raise ValueError(
                    f"retained atlas cell {cell_index} depth interval is reversed"
                )
        if not isinstance(cell["fallback"], bool):
            raise ValueError(
                f"retained atlas cell {cell_index} fallback flag is invalid"
            )
        reasons = cell["fallback_reasons"]
        if (
            not isinstance(reasons, list)
            or any(not isinstance(reason, str) or not reason for reason in reasons)
        ):
            raise ValueError(
                f"retained atlas cell {cell_index} fallback reasons are invalid"
            )


def write_retained_storage_artifact(
    path: Path,
    *,
    frame_count: int,
    trace_count: int,
    cell_count: int,
    tensors: Mapping[
        str,
        tuple[str, tuple[int, ...], bytes | Callable[[], bytes]] | None,
    ],
    topology: Mapping[str, Any],
) -> dict[str, Any]:
    """Serialize the exact compiled evaluator tensors and discrete topology.

    Tensor offsets are relative to the binary payload following the canonical
    JSON header. The returned byte count is an on-disk retained-state measure,
    not an estimate based on Python object sizes.
    """

    frame_count = _integer(frame_count, name="frame_count", minimum=1)
    trace_count = _integer(trace_count, name="trace_count", minimum=1)
    cell_count = _integer(cell_count, name="cell_count", minimum=1)
    if set(tensors) != set(TENSOR_NAMES):
        raise ValueError("retained atlas tensor set is incomplete")
    normalized_topology = json.loads(_canonical_json_bytes(topology))
    validate_topology(
        normalized_topology,
        trace_count=trace_count,
        cell_count=cell_count,
    )
    topology_bytes = _canonical_json_bytes(normalized_topology)

    records: list[dict[str, Any]] = []
    payload_offset = 0
    payload_path = path.with_name(path.name + ".payload.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with payload_path.open("wb") as payload_handle:
            for name in TENSOR_NAMES:
                tensor = tensors[name]
                if tensor is None:
                    if name in REQUIRED_TENSOR_NAMES:
                        raise ValueError(
                            f"required retained atlas tensor is absent: {name}"
                        )
                    records.append({"name": name, "present": False})
                    continue
                dtype, raw_shape, data_source = tensor
                if dtype not in DTYPE_BYTES:
                    raise ValueError(
                        f"unsupported retained atlas tensor dtype: {dtype}"
                    )
                data = data_source() if callable(data_source) else data_source
                if not isinstance(data, bytes):
                    raise ValueError(
                        f"retained atlas tensor {name} bytes are invalid"
                    )
                shape = tuple(
                    _integer(dimension, name=f"{name}.shape", minimum=0)
                    for dimension in raw_shape
                )
                if not shape or shape[0] != trace_count:
                    raise ValueError(
                        f"retained atlas tensor {name} leading dimension drifted"
                    )
                element_count = math.prod(shape)
                expected_bytes = element_count * DTYPE_BYTES[dtype]
                if len(data) != expected_bytes:
                    raise ValueError(
                        f"retained atlas tensor {name} byte count drifted"
                    )
                records.append(
                    {
                        "name": name,
                        "present": True,
                        "dtype": dtype,
                        "shape": list(shape),
                        "offset": payload_offset,
                        "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                )
                payload_handle.write(data)
                payload_offset += len(data)
                del data

        header = {
            "schema_version": 1,
            "format": FORMAT,
            "frame_count": frame_count,
            "trace_count": trace_count,
            "cell_count": cell_count,
            "tensor_payload_bytes": payload_offset,
            "topology_sha256": hashlib.sha256(topology_bytes).hexdigest(),
            "tensors": records,
            "topology": normalized_topology,
        }
        header_bytes = _canonical_json_bytes(header)
        with path.open("wb") as handle:
            handle.write(MAGIC)
            handle.write(struct.pack("<Q", len(header_bytes)))
            handle.write(header_bytes)
            with payload_path.open("rb") as payload_handle:
                shutil.copyfileobj(payload_handle, handle, length=1024 * 1024)
    finally:
        payload_path.unlink(missing_ok=True)
    total_bytes = path.stat().st_size
    topology_and_container_bytes = total_bytes - payload_offset
    identity = {
        "schema_version": 1,
        "format": FORMAT,
        "path": str(path.resolve()),
        "sha256": _file_sha256(path),
        "bytes": total_bytes,
        "tensor_payload_bytes": payload_offset,
        "topology_and_container_bytes": topology_and_container_bytes,
        "topology_sha256": header["topology_sha256"],
        "topology_included": True,
        "trace_count": trace_count,
        "cell_count": cell_count,
        "tensor_count": sum(record["present"] is True for record in records),
    }
    verify_retained_storage_artifact(
        identity,
        expected_frame_count=frame_count,
        expected_trace_count=trace_count,
        expected_cell_count=cell_count,
    )
    return identity


def verify_retained_storage_artifact(
    identity: Mapping[str, Any],
    *,
    expected_frame_count: int,
    expected_trace_count: int,
    expected_cell_count: int,
) -> Mapping[str, Any]:
    if (
        int(identity.get("schema_version", -1)) != 1
        or identity.get("format") != FORMAT
        or identity.get("topology_included") is not True
    ):
        raise ValueError("retained atlas artifact identity is stale")
    path = Path(str(identity.get("path", "")))
    if not path.is_file():
        raise ValueError("retained atlas artifact is missing")
    prefix_bytes = len(MAGIC) + 8
    with path.open("rb") as handle:
        magic = handle.read(len(MAGIC))
        length_bytes = handle.read(8)
    if magic != MAGIC or len(length_bytes) != 8:
        raise ValueError("retained atlas artifact magic is invalid")
    header_length = struct.unpack("<Q", length_bytes)[0]
    header_stop = prefix_bytes + header_length
    file_bytes = path.stat().st_size
    if header_length < 2 or header_stop > file_bytes:
        raise ValueError("retained atlas artifact header length is invalid")
    with path.open("rb") as handle:
        handle.seek(prefix_bytes)
        header_bytes = handle.read(header_length)
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("retained atlas artifact header is invalid") from error
    if _canonical_json_bytes(header) != header_bytes:
        raise ValueError("retained atlas artifact header is not canonical")
    if (
        int(header.get("schema_version", -1)) != 1
        or header.get("format") != FORMAT
        or int(header.get("frame_count", 0)) != expected_frame_count
        or int(header.get("trace_count", 0)) != expected_trace_count
        or int(header.get("cell_count", 0)) != expected_cell_count
    ):
        raise ValueError("retained atlas artifact contract drifted")
    topology = header.get("topology")
    if not isinstance(topology, Mapping):
        raise ValueError("retained atlas topology is missing")
    validate_topology(
        topology,
        trace_count=expected_trace_count,
        cell_count=expected_cell_count,
    )
    topology_sha256 = hashlib.sha256(_canonical_json_bytes(topology)).hexdigest()
    if (
        header.get("topology_sha256") != topology_sha256
        or identity.get("topology_sha256") != topology_sha256
    ):
        raise ValueError("retained atlas topology hash drifted")

    records = header.get("tensors")
    if (
        not isinstance(records, list)
        or [record.get("name") for record in records if isinstance(record, Mapping)]
        != list(TENSOR_NAMES)
    ):
        raise ValueError("retained atlas tensor records are incomplete")
    payload_bytes = file_bytes - header_stop
    expected_offset = 0
    present_count = 0
    with path.open("rb") as handle:
        handle.seek(header_stop)
        for record in records:
            if not isinstance(record, Mapping):
                raise ValueError("retained atlas tensor record is invalid")
            name = record["name"]
            present = record.get("present")
            if present is False:
                if (
                    name in REQUIRED_TENSOR_NAMES
                    or set(record) != {"name", "present"}
                ):
                    raise ValueError(
                        f"retained atlas tensor absence is invalid: {name}"
                    )
                continue
            if present is not True:
                raise ValueError(
                    f"retained atlas tensor presence is invalid: {name}"
                )
            dtype = record.get("dtype")
            shape = record.get("shape")
            byte_count = _integer(record.get("bytes"), name=f"{name}.bytes")
            offset = _integer(record.get("offset"), name=f"{name}.offset")
            if dtype not in DTYPE_BYTES or not isinstance(shape, list) or not shape:
                raise ValueError(
                    f"retained atlas tensor schema is invalid: {name}"
                )
            dimensions = tuple(
                _integer(value, name=f"{name}.shape", minimum=0)
                for value in shape
            )
            if dimensions[0] != expected_trace_count:
                raise ValueError(
                    f"retained atlas tensor trace count drifted: {name}"
                )
            if byte_count != math.prod(dimensions) * DTYPE_BYTES[dtype]:
                raise ValueError(
                    f"retained atlas tensor byte count drifted: {name}"
                )
            if offset != expected_offset or offset + byte_count > payload_bytes:
                raise ValueError(
                    f"retained atlas tensor offset drifted: {name}"
                )
            digest = hashlib.sha256()
            remaining = byte_count
            while remaining:
                chunk = handle.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise ValueError(
                        f"retained atlas tensor is truncated: {name}"
                    )
                digest.update(chunk)
                remaining -= len(chunk)
            if digest.hexdigest() != record.get("sha256"):
                raise ValueError(f"retained atlas tensor hash drifted: {name}")
            expected_offset += byte_count
            present_count += 1
    if expected_offset != payload_bytes:
        raise ValueError("retained atlas binary payload has trailing bytes")
    if int(header.get("tensor_payload_bytes", -1)) != payload_bytes:
        raise ValueError("retained atlas header payload count drifted")
    if (
        int(identity.get("bytes", -1)) != file_bytes
        or identity.get("sha256") != _file_sha256(path)
        or int(identity.get("tensor_payload_bytes", -1)) != payload_bytes
        or int(identity.get("topology_and_container_bytes", -1)) != header_stop
        or int(identity.get("trace_count", -1)) != expected_trace_count
        or int(identity.get("cell_count", -1)) != expected_cell_count
        or int(identity.get("tensor_count", -1)) != present_count
    ):
        raise ValueError("retained atlas artifact identity drifted")
    return header


__all__ = [
    "FORMAT",
    "LOGICAL_PAYLOAD_DEFINITION",
    "RETAINED_STORAGE_DEFINITION",
    "REPLAY_STORAGE_REASON",
    "ROUTE_MEMORY_DEFINITION",
    "ROUTE_MEMORY_MEASUREMENT_SOURCE",
    "TENSOR_NAMES",
    "validate_topology",
    "verify_retained_storage_artifact",
    "write_retained_storage_artifact",
]

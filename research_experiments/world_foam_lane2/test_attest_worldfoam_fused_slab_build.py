from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import attest_worldfoam_fused_slab_build as attest
from verify_worldfoam_native_variant_sources import (
    _load_native_build_contract,
    _schema_inventory,
)


def test_source_contract_is_exact_133_schema_build_boundary() -> None:
    snapshot = attest._source_snapshot(attest.DEFAULT_VARIANT_DIR)

    assert snapshot["schema_count"] == 133
    assert snapshot["required_post_103_schema_count"] == 30
    assert (
        snapshot["schema_name_inventory_sha256"]
        == "818d42fd3c45c89cc55fb886f16be0d7a6a9479ba66867bdac3dc77fe4a810d8"
    )
    assert (
        snapshot["full_schema_inventory_sha256"]
        == "4296969b4943bf685d3e4e7fec5a211c5a2f85dff5f07d71821c4252c5f91168"
    )
    assert snapshot["translation_units"] == [
        "csrc/bindings.cpp",
        "csrc/metal/world_foam_lane2_metal.mm",
    ]
    assert set(snapshot["runtime_metal_sources"]) == {
        "csrc/metal/world_foam_lane2_power_boundary_tensor.metal",
        "csrc/metal/world_foam_lane2_shared_replay_tensor.metal",
    }


def test_exact_inventory_comparison_rejects_pre_30_registration_binary() -> None:
    bindings = (
        attest.DEFAULT_VARIANT_DIR / "csrc" / "bindings.cpp"
    ).read_text(encoding="utf-8")
    schemas = _schema_inventory(bindings)
    contract, _, failures = _load_native_build_contract(attest.DEFAULT_VARIANT_DIR)
    assert not failures and contract is not None
    missing = set(contract.REQUIRED_POST_103_SCHEMA_NAMES)
    retained = sorted(set(schemas) - missing)
    compiled_schemas = [schemas[name] for name in retained]
    compiled = {
        "dispatch_key": attest.DISPATCH_KEY,
        "schema_count": len(retained),
        "schema_names": retained,
        "schemas": compiled_schemas,
        "schema_name_inventory_sha256": attest._inventory_sha256(retained),
        "full_schema_inventory_sha256": attest._inventory_sha256(compiled_schemas),
        "missing_dispatch_kernels": [],
    }

    failures = attest._compare_source_and_compiled(
        attest._source_snapshot(attest.DEFAULT_VARIANT_DIR),
        compiled,
        bindings,
    )

    assert len(retained) == 103
    assert any("schema-name mismatch" in failure for failure in failures)
    assert any("schema_count mismatch" in failure for failure in failures)


def test_receipt_digest_is_canonical_and_tamper_evident() -> None:
    payload = {
        "schema_version": 1,
        "kind": attest.ATTESTATION_KIND,
        "status": "accepted",
        "nested": {"b": 2, "a": 1},
    }
    payload["receipt_payload_sha256"] = attest._payload_sha256(payload)
    reordered = json.loads(json.dumps(payload, sort_keys=False))

    assert attest._payload_sha256(reordered) == payload["receipt_payload_sha256"]
    reordered["nested"]["a"] = 3
    assert attest._payload_sha256(reordered) != payload["receipt_payload_sha256"]


def test_source_only_cli_does_not_require_site_packages_or_import_extension() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(Path(attest.__file__).resolve()),
            "--source-only",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "accepted"
    assert payload["scope"] == "source_only_no_extension_import_no_metal_dispatch"
    assert payload["source"]["schema_count"] == 133


def test_receipt_json_schema_pins_python311_darwin_binary_and_133_ops() -> None:
    schema = json.loads(attest.RECEIPT_SCHEMA_PATH.read_text(encoding="utf-8"))

    extension = schema["properties"]["extension"]["properties"]
    compiled = schema["properties"]["compiled_operator_inventory"]["properties"]
    runtime = schema["properties"]["runtime"]["properties"]
    assert extension["basename"]["const"] == "_C.cpython-311-darwin.so"
    assert compiled["schema_count"]["const"] == 133
    assert compiled["missing_dispatch_kernels"]["const"] == []
    assert runtime["platform_system"]["const"] == "Darwin"
    assert runtime["python_version_info"]["prefixItems"][:2] == [
        {"const": 3},
        {"const": 11},
    ]


def test_setup_rejects_parent_project_metadata_leak_and_accepts_variant_cwd() -> None:
    setup_py = attest.DEFAULT_VARIANT_DIR / "setup.py"
    wrong_cwd = subprocess.run(
        [sys.executable, str(setup_py), "--name"],
        cwd=attest.DYNAWORLD,
        check=False,
        capture_output=True,
        text=True,
    )
    correct_cwd = subprocess.run(
        [sys.executable, "setup.py", "--name"],
        cwd=attest.DEFAULT_VARIANT_DIR,
        check=False,
        capture_output=True,
        text=True,
    )

    assert wrong_cwd.returncode != 0
    assert "parent pyproject metadata cannot leak in" in wrong_cwd.stderr
    assert correct_cwd.returncode == 0, correct_cwd.stderr
    assert correct_cwd.stdout.strip() == "torch-world-foam-lane2-fused-slab-v0"

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import verify_worldfoam_native_variant_imports as verify_mod


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class VerifyWorldFoamNativeVariantImportsTests(unittest.TestCase):
    def test_real_worldfoam_variants_import_and_register_compiled_schemas(self) -> None:
        result = verify_mod.verify()

        self.assertEqual(result["variant_count"], 1)
        row = result["variants"][0]
        self.assertEqual(row["schema_count"], 133)
        self.assertEqual(row["build_contract_schema_count"], 133)
        self.assertEqual(
            row["source_schema_inventory_sha256"],
            "4296969b4943bf685d3e4e7fec5a211c5a2f85dff5f07d71821c4252c5f91168",
        )
        self.assertEqual(row["extension_load_error"], "")
        self.assertTrue(row["kinetic_compiled_abi_attestation_present"])
        self.assertTrue(row["extension_library"].endswith("_C.cpython-311-darwin.so"))

        # This checkout intentionally retains the pre-30-registration binary
        # until an operator-approved quiet-host rebuild.  The same assertion
        # also accepts the future fully attested state, but never a partial or
        # silently stale registration set.
        if row["registered_schema_count"] == 133:
            self.assertTrue(row["exact_schema_inventory_match"])
            self.assertFalse(row["rebuild_required"])
            self.assertEqual(row["missing_registered_schemas"], [])
            self.assertEqual(row["unexpected_registered_schemas"], [])
            self.assertEqual(row["mismatched_registered_schema_signatures"], [])
            self.assertEqual(row["missing_dispatch_kernels"], [])
            self.assertIsNotNone(row["attestation"])
            self.assertEqual(row["attestation"]["status"], "accepted")
            self.assertEqual(result["status"], "ok", result["failures"])
        else:
            self.assertEqual(row["registered_schema_count"], 103)
            self.assertEqual(len(row["missing_registered_schemas"]), 30)
            self.assertFalse(row["exact_schema_inventory_match"])
            self.assertTrue(row["rebuild_required"])
            self.assertEqual(result["status"], "failed")
            self.assertIn(
                "native build attestation is missing",
                "\n".join(result["failures"]),
            )

    def test_rejects_variant_without_built_extension_library(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            variant_dir = root / "unit_variant"
            _write(
                variant_dir / "csrc" / "bindings.cpp",
                """
TORCH_LIBRARY(unit_variant, m) {
  m.def("foo(Tensor x) -> Tensor");
}
TORCH_LIBRARY_IMPL(unit_variant, CompositeExplicitAutograd, m) {
  m.impl("foo", unit::foo_dispatch);
}
""",
            )
            _write(
                variant_dir / "torch_unit_variant" / "__init__.py",
                "from . import ops\n",
            )
            _write(
                variant_dir / "torch_unit_variant" / "ops.py",
                "_EXTENSION_LOAD_ERROR = None\n",
            )

            result = verify_mod.verify(root, variants=(("unit_variant", "torch_unit_variant"),))

        self.assertEqual(result["status"], "failed")
        self.assertIn("missing built extension library", "\n".join(result["failures"]))


if __name__ == "__main__":
    unittest.main()

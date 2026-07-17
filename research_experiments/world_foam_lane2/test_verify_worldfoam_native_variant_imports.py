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

        self.assertEqual(result["status"], "ok", result["failures"])
        self.assertEqual(result["variant_count"], 3)
        for row in result["variants"]:
            self.assertEqual(row["status"], "ok", row["failures"])
            self.assertGreater(row["schema_count"], 0)
            self.assertEqual(row["registered_schema_count"], row["schema_count"])
            self.assertEqual(row["missing_registered_schemas"], [])
            self.assertEqual(row["extension_load_error"], "")
            self.assertTrue(row["extension_library"].endswith(".so"))

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

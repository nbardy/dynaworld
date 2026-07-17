from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import verify_worldfoam_native_variant_sources as verify_mod


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _minimal_variant(root: Path, *, variant: str = "unit_variant", package: str = "torch_unit_variant") -> None:
    variant_dir = root / variant
    _write(
        variant_dir / "csrc" / "bindings.cpp",
        """
namespace unit {
void foo_dispatch() {}
}

TORCH_LIBRARY(unit_variant, m) {
  m.def(
      "foo(Tensor x) -> Tensor");
}
TORCH_LIBRARY_IMPL(unit_variant, CompositeExplicitAutograd, m) {
  m.impl(
      "foo",
      unit::foo_dispatch);
}
""",
    )
    _write(
        variant_dir / "csrc" / "metal" / "world_foam_lane2_metal.mm",
        """
struct MetalKernels {
  std::shared_ptr<MetalKernelFunction> unit_kernel;
};

MetalKernels& kernels() {
  NSString* unitPath = [metalPath stringByAppendingPathComponent:@"unit.metal"];
  static MetalKernels out;
  out.unit_kernel = lib->getKernelFunction("wf2_unit_kernel");
  return out;
}

void launch() {
  kernels().unit_kernel;
}
""",
    )
    _write(
        variant_dir / "csrc" / "metal" / "unit.metal",
        """
#include <metal_stdlib>
using namespace metal;
kernel void wf2_unit_kernel(device float* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  out[tid] = 0.0f;
}
""",
    )
    _write(
        variant_dir / package / "ops.py",
        """
import torch
ops = torch.ops.unit_variant

def foo(x):
    return ops.foo(x)
""",
    )


class VerifyWorldFoamNativeVariantSourcesTests(unittest.TestCase):
    def test_torch_ops_load_library_is_not_treated_as_custom_op_reference(self) -> None:
        refs = verify_mod._torch_ops_refs(
            """
import torch
torch.ops.load_library("_C.so")
ops = torch.ops.unit_variant
ops.foo(x)
"""
        )

        self.assertEqual(refs, {"foo"})

    def test_real_worldfoam_variants_have_consistent_source_wiring(self) -> None:
        result = verify_mod.verify()

        self.assertEqual(result["status"], "ok", result["failures"])
        self.assertEqual(result["variant_count"], 1)
        for row in result["variants"]:
            self.assertEqual(row["status"], "ok", row["failures"])
            self.assertGreater(row["schema_count"], 0)
            self.assertGreater(row["impl_count"], 0)
            self.assertEqual(row["impl_target_count"], row["impl_count"])
            self.assertGreater(row["host_kernel_ref_count"], 0)
            self.assertGreater(row["host_kernel_field_count"], 0)
            self.assertEqual(row["initialized_kernel_field_count"], row["host_kernel_field_count"])
            self.assertGreater(row["metal_kernel_count"], 0)

    def test_rejects_host_kernel_reference_without_metal_declaration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _minimal_variant(root)
            host = root / "unit_variant" / "csrc" / "metal" / "world_foam_lane2_metal.mm"
            host.write_text('lib->getKernelFunction("wf2_missing_kernel");\n', encoding="utf-8")

            result = verify_mod.verify(root, variants=(("unit_variant", "torch_unit_variant"),))

        self.assertEqual(result["status"], "failed")
        self.assertIn("host loads Metal kernels that are not declared", "\n".join(result["failures"]))

    def test_rejects_host_kernel_reference_absent_from_loaded_metal_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _minimal_variant(root)
            host = root / "unit_variant" / "csrc" / "metal" / "world_foam_lane2_metal.mm"
            host.write_text(host.read_text(encoding="utf-8").replace("unit.metal", "other.metal"), encoding="utf-8")
            _write(
                root / "unit_variant" / "csrc" / "metal" / "other.metal",
                """
#include <metal_stdlib>
using namespace metal;
kernel void wf2_other_kernel(device float* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
  out[tid] = 0.0f;
}
""",
            )

            result = verify_mod.verify(root, variants=(("unit_variant", "torch_unit_variant"),))

        self.assertEqual(result["status"], "failed")
        self.assertIn(
            "host loads Metal kernels absent from dynamically loaded Metal sources",
            "\n".join(result["failures"]),
        )

    def test_rejects_python_op_reference_without_schema_or_impl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _minimal_variant(root)
            ops = root / "unit_variant" / "torch_unit_variant" / "ops.py"
            ops.write_text(
                """
import torch
ops = torch.ops.unit_variant

def bar(x):
    return ops.bar(x)
""",
                encoding="utf-8",
            )

            result = verify_mod.verify(root, variants=(("unit_variant", "torch_unit_variant"),))

        failures = "\n".join(result["failures"])
        self.assertEqual(result["status"], "failed")
        self.assertIn("Python ops references without schema definitions", failures)
        self.assertIn("Python ops references without native implementations", failures)

    def test_rejects_impl_target_without_source_definition(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _minimal_variant(root)
            bindings = root / "unit_variant" / "csrc" / "bindings.cpp"
            bindings.write_text(
                """
TORCH_LIBRARY(unit_variant, m) {
  m.def("foo(Tensor x) -> Tensor");
}
TORCH_LIBRARY_IMPL(unit_variant, CompositeExplicitAutograd, m) {
  m.impl("foo", unit::missing_dispatch);
}
""",
                encoding="utf-8",
            )

            result = verify_mod.verify(root, variants=(("unit_variant", "torch_unit_variant"),))

        self.assertEqual(result["status"], "failed")
        self.assertIn("native dispatch targets without source definitions", "\n".join(result["failures"]))

    def test_rejects_declared_metal_kernel_field_without_initializer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _minimal_variant(root)
            host = root / "unit_variant" / "csrc" / "metal" / "world_foam_lane2_metal.mm"
            host.write_text(
                """
struct MetalKernels {
  std::shared_ptr<MetalKernelFunction> unit_kernel;
};

MetalKernels& kernels() {
  static MetalKernels out;
  return out;
}
""",
                encoding="utf-8",
            )

            result = verify_mod.verify(root, variants=(("unit_variant", "torch_unit_variant"),))

        self.assertEqual(result["status"], "failed")
        self.assertIn("declared Metal kernel fields without initializers", "\n".join(result["failures"]))

    def test_rejects_initializer_for_undeclared_metal_kernel_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _minimal_variant(root)
            host = root / "unit_variant" / "csrc" / "metal" / "world_foam_lane2_metal.mm"
            host.write_text(
                """
struct MetalKernels {
  std::shared_ptr<MetalKernelFunction> unit_kernel;
};

MetalKernels& kernels() {
  static MetalKernels out;
  out.other_kernel = lib->getKernelFunction("wf2_unit_kernel");
  return out;
}
""",
                encoding="utf-8",
            )

            result = verify_mod.verify(root, variants=(("unit_variant", "torch_unit_variant"),))

        failures = "\n".join(result["failures"])
        self.assertEqual(result["status"], "failed")
        self.assertIn("host initializes Metal kernel fields that are not declared", failures)
        self.assertIn("declared Metal kernel fields without initializers", failures)


if __name__ == "__main__":
    unittest.main()

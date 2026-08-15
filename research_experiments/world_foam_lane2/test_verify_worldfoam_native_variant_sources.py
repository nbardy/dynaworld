from __future__ import annotations

import re
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import torch
import verify_worldfoam_native_variant_sources as verify_mod


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _braced_definition(source: str, signature: str) -> str:
    """Extract exactly one C++/Metal definition, including nested lambdas."""

    start = source.index(signature)
    opening = source.index("{", start)
    depth = 0
    for cursor in range(opening, len(source)):
        if source[cursor] == "{":
            depth += 1
        elif source[cursor] == "}":
            depth -= 1
            if depth == 0:
                return source[start : cursor + 1]
    raise AssertionError(f"unterminated definition: {signature}")


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


def _native_continuous_binding_fixture():
    from compiled_lie_world_adjoint import (
        AdaptiveCompiledLieWorldAtlas,
        AdaptiveLieWorldCompilePolicy,
        compile_lie_world_atlas,
    )
    from compiled_transfer_adjoint import make_stable_cell_word, power_boundary_parameters
    from continuous_adaptive_lie_acceptance import ContinuousAdaptiveLieCertificationPolicy
    from prepared_track_block import prepare_worldfoam_track_block
    from staged_compiled_lie_adjoint import prepare_compact_staged_lie_world_snapshot
    from torch_world_foam_lane2_fused_slab.certificate_binding import (
        certify_and_bind_native_fixed_word_p0,
    )

    sites = torch.tensor(
        [
            [0.0, 0.0, 0.2, -0.05, 0.01],
            [0.03, 0.0, 0.6, 0.05, -0.02],
        ],
        dtype=torch.float64,
    )
    pairs = torch.tensor([[0, 1]], dtype=torch.int64)
    boundary = power_boundary_parameters(sites, pairs)
    rays = torch.tensor(
        [[0.01, 0.0, 0.0, 0.005, 0.0, 0.005, 0.0, 0.0, 1.0, 0.005, 0.0, 0.02]],
        dtype=torch.float64,
    )
    density = torch.tensor([0.5, 0.7], dtype=torch.float64)
    color = torch.tensor([[0.2, 0.4, 0.8], [0.8, 0.3, 0.2]], dtype=torch.float64)
    words = (make_stable_cell_word([0, 1], [-1, 0], [0, -2]),)
    chart = compile_lie_world_atlas(
        boundary=boundary,
        ray_coefficients=rays,
        words=words,
        site_density=density,
        site_color=color,
        t_min=-0.5,
        t_max=0.5,
        near=0.1,
        far=0.9,
        node_count=4,
    )
    atlas = AdaptiveCompiledLieWorldAtlas(
        charts=(chart,),
        selections=(),
        policy=AdaptiveLieWorldCompilePolicy(),
        supplied_word_ordering_check=chart.supplied_word_ordering_check,
    )
    topology = prepare_worldfoam_track_block(
        words,
        pairs,
        site_count=2,
        track_start=0,
        track_end=1,
    )
    prepared = prepare_compact_staged_lie_world_snapshot(
        atlas,
        topology,
        site_geometry=sites,
        ray_coefficients=rays,
        site_density=density,
        site_color=color,
    )
    binding = certify_and_bind_native_fixed_word_p0(
        prepared,
        policy=ContinuousAdaptiveLieCertificationPolicy(
            transfer_tolerance=0.01,
            world_jacobian_tolerance=0.2,
            site_geometry_jacobian_tolerance=0.25,
            max_split_depth=2,
            max_leaves_per_chart=16,
            max_interval_jet_work_units_per_chart=10_000,
            arithmetic_fraction_bits=64,
            owner_identity_tolerance=1.0e-9,
            owner_max_split_depth=12,
            owner_max_leaves_per_chart=4096,
            owner_max_work_units_per_chart=100_000,
        ),
    )
    return binding, prepared


class VerifyWorldFoamNativeVariantSourcesTests(unittest.TestCase):
    def test_torch_ops_load_library_is_not_treated_as_custom_op_reference(self) -> None:
        refs = verify_mod._torch_ops_refs(
            """
import torch
torch.ops.load_library("_C.so")
ops = torch.ops.unit_variant
ops.foo(x)
torch.ops.unit_variant.bar(x)
"""
        )

        self.assertEqual(refs, {"bar", "foo"})

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
            contract = row["kinetic_memory_contract"]
            self.assertIsNotNone(contract)
            self.assertEqual(contract["status"], "ok", contract["failures"])
            self.assertTrue(contract["source_contract_only"])
            self.assertFalse(contract["allocator_peak_measured"])
            self.assertFalse(contract["invocation_frequency_verified"])
            build_contract = row["build_contract"]
            self.assertTrue(row["build_contract_module_loaded"])
            self.assertEqual(build_contract["schema_count"], 133)
            self.assertEqual(build_contract["required_post_103_schema_count"], 30)
            self.assertEqual(
                build_contract["schema_name_inventory_sha256"],
                "818d42fd3c45c89cc55fb886f16be0d7a6a9479ba66867bdac3dc77fe4a810d8",
            )
            self.assertEqual(
                build_contract["full_schema_inventory_sha256"],
                "4296969b4943bf685d3e4e7fec5a211c5a2f85dff5f07d71821c4252c5f91168",
            )

    def test_fixed_word_launch_and_stage_schemas_parse(self) -> None:
        variant = verify_mod.DEFAULT_VARIANT_ROOT / "world_foam_lane2_fused_slab_v0"
        bindings_source = (variant / "csrc" / "bindings.cpp").read_text(encoding="utf-8")
        schemas = {
            schema.split("(", 1)[0]: schema
            for schema in re.findall(
                r'm\.def\(\s*"([^"]+)"',
                bindings_source,
                flags=re.DOTALL,
            )
        }
        expected = {
            "fixed_word_p0_compiled_lie_transfer_mse_vjp_sparse_mobius_boundary_launch_only",
            "sparse_power_boundary_from_sites_launch_only",
            "fixed_word_p0_sparse_mobius_lower_launch_only",
            "fixed_word_p0_lie_node_forward_launch_only",
            "kinetic_precompiled_length_p0_lie_node_forward_launch_only",
            "kinetic_precompiled_length_p0_lie_node_forward_into_launch_only_v1",
            "fixed_word_p0_lie_sample_state_init_launch_only",
            "fixed_word_p0_lie_sample_accumulate_launch_only",
            "fixed_word_p0_lie_world_grad_init_launch_only",
            "fixed_word_p0_lie_material_world_grad_init_launch_only",
            "fixed_word_p0_lie_node_vjp_accumulate_launch_only",
            "kinetic_precompiled_length_p0_lie_node_vjp_accumulate_launch_only",
            "kinetic_precompiled_length_p0_lie_material_node_vjp_accumulate_launch_only",
            "fixed_word_p0_lie_material_node_vjp_accumulate_launch_only",
            "fixed_word_p0_sparse_mobius_boundary_finalize_launch_only",
            "sparse_power_boundary_vjp_to_sites_launch_only",
        }
        self.assertLessEqual(expected, schemas.keys())
        for name in expected:
            torch._C.parse_schema(schemas[name])

    def test_constant_state_factorized_bridge_has_frame_independent_reverse_state_and_boundary_vjp_abi(self) -> None:
        variant = verify_mod.DEFAULT_VARIANT_ROOT / "world_foam_lane2_fused_slab_v0"
        metal_source = (variant / "csrc" / "metal" / "world_foam_lane2_shared_replay_tensor.metal").read_text(
            encoding="utf-8"
        )
        kernel_name = (
            "wf2_endpoint_record_delta_replace_factorized_packed_framegroup16_"
            "constant_state_mse_vjp_direct_atomic_rgb_boundary_tensor"
        )
        kernel_start = metal_source.index(f"kernel void {kernel_name}(")
        kernel_end = metal_source.find("\nkernel void ", kernel_start + 1)
        kernel_source = metal_source[kernel_start : kernel_end if kernel_end >= 0 else None]

        self.assertNotIn("uint owners[", kernel_source)
        self.assertNotIn("float lengths[", kernel_source)
        self.assertNotIn("float trans_before[", kernel_source)
        self.assertNotIn("if (!(depth_length > 1.0e-8f)", kernel_source)
        self.assertGreaterEqual(kernel_source.count("if (!(physical_length > 1.0e-8f)"), 2)
        self.assertGreaterEqual(kernel_source.count("for (uint cursor"), 2)
        self.assertGreaterEqual(
            kernel_source.count("wf2_endpoint_record_factorized_cut_depth_boundary_jacobian"),
            4,
        )
        for required in (
            "processed_run_count",
            "prefix_transmittance",
            "prefix_rgb",
            "wf2_endpoint_record_factorized_cut_depth_boundary_jacobian",
            "const float fiber_speed = length(ray_direction);",
            "const float physical_length = fiber_speed * depth_length;",
            "grad_rgba.w = physical_length * tau_bar;",
            "const float endpoint_depth_bar = fiber_speed * density * tau_bar;",
            "device atomic_float* grad_boundary_f32 [[buffer(16)]]",
            "wf2_atomic_add5",
        ):
            self.assertIn(required, kernel_source)
        for required in (
            "const float denominator_scale = length(normal) * length(direction);",
            "fabs(denominator) < invalid_epsilon * denominator_scale",
        ):
            self.assertIn(required, metal_source)

        op_name = (
            "endpoint_record_delta_replace_factorized_packed_framegroup16_"
            "constant_state_mse_vjp_direct_atomic_rgb_boundary"
        )
        bindings_source = (variant / "csrc" / "bindings.cpp").read_text(encoding="utf-8")
        self.assertIn(f'"{op_name}(', bindings_source)
        schema_tail = bindings_source.split(f'"{op_name}(', 1)[1].split('");', 1)[0]
        self.assertIn("-> (Tensor, Tensor, Tensor)", schema_tail)

        host_source = (variant / "csrc" / "metal" / "world_foam_lane2_metal.mm").read_text(encoding="utf-8")
        self.assertIn("auto grad_boundary = torch::empty({boundary_count, 5}", host_source)
        self.assertIn("return std::make_tuple(loss, grad_site_rgba, grad_boundary);", host_source)

        package_source = (variant / "torch_world_foam_lane2_fused_slab" / "__init__.py").read_text(encoding="utf-8")
        self.assertIn(op_name, package_source)

    def test_sparse_mobius_p0_bridge_stages_incidence_lowering_replay_and_boundary_vjp(self) -> None:
        variant = verify_mod.DEFAULT_VARIANT_ROOT / "world_foam_lane2_fused_slab_v0"
        metal_source = (variant / "csrc" / "metal" / "world_foam_lane2_shared_replay_tensor.metal").read_text(
            encoding="utf-8"
        )
        kernel_name = (
            "wf2_endpoint_record_delta_replace_factorized_packed_framegroup16_"
            "constant_state_p0_mse_vjp_sparse_mobius_rgb_tensor"
        )
        kernel_start = metal_source.index(f"kernel void {kernel_name}(")
        kernel_end = metal_source.find("\nkernel void ", kernel_start + 1)
        kernel_source = metal_source[kernel_start : kernel_end if kernel_end >= 0 else None]

        self.assertNotIn("uint owners[", kernel_source)
        self.assertNotIn("float lengths[", kernel_source)
        self.assertNotIn("float trans_before[", kernel_source)
        self.assertNotIn("boundary_f32", kernel_source)
        self.assertNotIn("incidence_boundary_i32", kernel_source)
        for required in (
            "device const float* mobius_coeff_f32 [[buffer(0)]]",
            "device const int* track_incidence_offsets_i32 [[buffer(10)]]",
            "device atomic_float* grad_mobius_coeff_f32 [[buffer(17)]]",
            "const float physical_length = fiber_speed * depth_length;",
            "grad_rgba.w = physical_length * tau_bar;",
            "wf2_atomic_add4(\n              grad_mobius_coeff_f32",
        ):
            self.assertIn(required, kernel_source)

        helper_start = metal_source.index("static inline bool wf2_endpoint_record_sparse_mobius_cut_depth_jacobian(")
        helper_end = metal_source.index("\nstatic inline ", helper_start + 1)
        helper_source = metal_source[helper_start:helper_end]
        for required in (
            "const uint incidence_id = incidence_begin + local_incidence_id;",
            "const float coefficient_a = mobius_coeff_f32[coefficient_base + 0u];",
            "const float denominator_scale = max(1.0f, fabs(coefficient_c) + fabs(coefficient_d));",
            "fabs(denominator) <= invalid_epsilon * denominator_scale",
            "-numerator * inv_denominator * inv_denominator",
        ):
            self.assertIn(required, helper_source)
        self.assertNotIn("boundary_f32", helper_source)
        self.assertNotIn("track_ray_coeff_f32", helper_source)

        lower_start = metal_source.index("kernel void wf2_sparse_mobius_incidence_lower_tensor(")
        lower_end = metal_source.index("\nkernel void ", lower_start + 1)
        lower_source = metal_source[lower_start:lower_end]
        for required in (
            "for (uint incidence_id = incidence_begin; incidence_id < incidence_end; ++incidence_id)",
            "-dot(origin_base, normal) - boundary_f32[boundary_base + 4u]",
            "-dot(origin_slope, normal) - boundary_f32[boundary_base + 3u]",
            "mobius_coeff_f32[coefficient_base + 2u] = dot(direction_base, normal);",
            "mobius_coeff_f32[coefficient_base + 3u] = dot(direction_slope, normal);",
        ):
            self.assertIn(required, lower_source)

        vjp_start = metal_source.index("kernel void wf2_sparse_mobius_incidence_boundary_vjp_tensor(")
        vjp_end = metal_source.index("\nkernel void ", vjp_start + 1)
        vjp_source = metal_source[vjp_start:vjp_end]
        for required in (
            "for (uint incidence_id = incidence_begin; incidence_id < incidence_end; ++incidence_id)",
            "-grad_a * origin_base - grad_b * origin_slope +",
            "grad_c * direction_base + grad_d * direction_slope",
            "grad_normal,\n        -grad_b,\n        -grad_a",
        ):
            self.assertIn(required, vjp_source)

        op_name = (
            "endpoint_record_delta_replace_factorized_packed_framegroup16_"
            "constant_state_p0_mse_vjp_sparse_mobius_rgb_boundary"
        )
        bindings_source = (variant / "csrc" / "bindings.cpp").read_text(encoding="utf-8")
        self.assertIn(f'"{op_name}(', bindings_source)
        schema_tail = bindings_source.split(f'"{op_name}(', 1)[1].split('");', 1)[0]
        self.assertIn("Tensor track_incidence_offsets_i32", schema_tail)
        self.assertIn("Tensor incidence_boundary_i32", schema_tail)
        self.assertIn("-> (Tensor, Tensor, Tensor, Tensor)", schema_tail)

        host_source = (variant / "csrc" / "metal" / "world_foam_lane2_metal.mm").read_text(encoding="utf-8")
        function_start = host_source.index(f"metal_{op_name}(")
        function_end = host_source.index("\nstd::tuple<", function_start + 1)
        function_source = host_source[function_start:function_end]
        for required in (
            "auto mobius_coeff = torch::empty({incidence_count, 4}",
            "auto grad_mobius_coeff = torch::empty({incidence_count, 4}",
            "check_track_boundary_incidence_csr_cpu(",
            "check_packed_endpoint_incidence_delta_records_cpu(",
            "return std::make_tuple(loss, grad_site_rgba, grad_mobius_coeff, grad_boundary);",
        ):
            self.assertIn(required, function_source)
        lower_launch = function_source.index("launch(k.sparse_mobius_incidence_lower")
        replay_launch = function_source.index("launch(framegroup_kernel")
        vjp_launch = function_source.index("launch(k.sparse_mobius_incidence_boundary_vjp")
        self.assertLess(lower_launch, replay_launch)
        self.assertLess(replay_launch, vjp_launch)

        ops_source = (variant / "torch_world_foam_lane2_fused_slab" / "ops.py").read_text(encoding="utf-8")
        for required in (
            "def recode_packed_endpoint_delta_records_to_track_incidence(",
            "each incidence CSR row must contain strictly increasing unique boundary ids",
            "boundary_to_local = {",
            f"def {op_name}(",
        ):
            self.assertIn(required, ops_source)
        package_source = (variant / "torch_world_foam_lane2_fused_slab" / "__init__.py").read_text(encoding="utf-8")
        self.assertIn(op_name, package_source)
        self.assertIn("recode_packed_endpoint_delta_records_to_track_incidence", package_source)

    def test_sparse_mobius_record_recoder_uses_row_local_ids_and_fails_on_missing_incidence(self) -> None:
        variant = verify_mod.DEFAULT_VARIANT_ROOT / "world_foam_lane2_fused_slab_v0"
        sys.path.insert(0, str(variant))
        try:
            from torch_world_foam_lane2_fused_slab.ops import (
                recode_packed_endpoint_delta_records_to_track_incidence,
            )
        finally:
            sys.path.remove(str(variant))

        def pack(owner: int, left: int, right: int) -> int:
            left_code = left + 2 if left >= 0 else 0 if left == -1 else 1
            right_code = right + 2 if right >= 0 else 0 if right == -1 else 1
            value = owner | (left_code << 8) | (right_code << 20)
            return value if value < (1 << 31) else value - (1 << 32)

        base_records = torch.tensor(
            [pack(0, -1, 1), pack(1, 4, -2), pack(2, -1, 2)],
            dtype=torch.int32,
        )
        inputs = {
            "track_incidence_offsets_i32": torch.tensor([0, 2, 3], dtype=torch.int32),
            "incidence_boundary_i32": torch.tensor([1, 4, 2], dtype=torch.int32),
            "base_offsets_i16": torch.tensor([0, 2, 3], dtype=torch.int16),
            "base_record_i32": base_records,
            "track_change_offsets_i16": torch.tensor([0, 0, 0], dtype=torch.int16),
            "change_offsets_i16": torch.tensor([0], dtype=torch.int16),
            "change_record_i32": torch.empty(0, dtype=torch.int32),
        }
        base_recoded, change_recoded = recode_packed_endpoint_delta_records_to_track_incidence(
            **inputs,
            track_count=2,
            boundary_count=5,
            site_count=3,
        )
        packed_u32 = base_recoded.to(torch.int64) & 0xFFFFFFFF
        self.assertEqual(((packed_u32 >> 8) & 4095).tolist(), [0, 3, 0])
        self.assertEqual(((packed_u32 >> 20) & 4095).tolist(), [2, 1, 2])
        self.assertEqual(change_recoded.numel(), 0)

        with self.assertRaisesRegex(ValueError, "absent from its track incidence row"):
            recode_packed_endpoint_delta_records_to_track_incidence(
                **{
                    **inputs,
                    "track_incidence_offsets_i32": torch.tensor([0, 1, 2], dtype=torch.int32),
                    "incidence_boundary_i32": torch.tensor([1, 2], dtype=torch.int32),
                },
                track_count=2,
                boundary_count=5,
                site_count=3,
            )

    def test_fixed_word_compiled_lie_bridge_has_staged_sample_independent_world_adjoint(self) -> None:
        variant = verify_mod.DEFAULT_VARIANT_ROOT / "world_foam_lane2_fused_slab_v0"
        metal_source = (variant / "csrc" / "metal" / "world_foam_lane2_shared_replay_tensor.metal").read_text(
            encoding="utf-8"
        )

        def kernel_source(name: str) -> str:
            start = metal_source.index(f"kernel void {name}(")
            cursor = metal_source.index("{", start)
            depth = 0
            while cursor < len(metal_source):
                if metal_source[cursor] == "{":
                    depth += 1
                elif metal_source[cursor] == "}":
                    depth -= 1
                    if depth == 0:
                        return metal_source[start : cursor + 1]
                cursor += 1
            raise AssertionError(f"unterminated Metal kernel: {name}")

        forward_source = kernel_source("wf2_fixed_word_p0_lie_node_forward_tensor")
        sample_source = kernel_source("wf2_fixed_word_p0_lie_sample_mse_vjp_tensor")
        sample_helper_start = metal_source.index("inline bool wf2_fixed_word_p0_lie_sample_mse_vjp(")
        sample_helper_source = metal_source[
            sample_helper_start : metal_source.index("kernel void wf2_fixed_word_p0_lie_sample_mse_vjp_tensor(")
        ]
        sample_math_source = sample_helper_source + sample_source
        vjp_helper_start = metal_source.index("static inline void wf2_fixed_word_p0_lie_node_vjp_impl(")
        vjp_helper_end = metal_source.index(
            "\nkernel void wf2_fixed_word_p0_lie_node_vjp_tensor(",
            vjp_helper_start,
        )
        vjp_source = metal_source[vjp_helper_start:vjp_helper_end] + kernel_source(
            "wf2_fixed_word_p0_lie_node_vjp_tensor"
        )
        for source in (forward_source, vjp_source):
            for forbidden in (
                "uint owners[",
                "float lengths[",
                "float trans_before[",
                "boundary_f32",
                "incidence_boundary_i32",
                "frame_count",
            ):
                self.assertNotIn(forbidden, source)
            for required in (
                "word_offsets_i32",
                "word_owner_i32",
                "word_left_incidence_i32",
                "word_right_incidence_i32",
                "const float segment_alpha = -expm1(-optical_depth);",
                "const uint incidence_count = uint(config_i32[6]);",
            ):
                self.assertIn(required, source)
            self.assertNotIn("1.0f - segment_beta", source)
        for required in (
            "for (int cursor = word_begin_raw; valid && cursor < word_end_raw; ++cursor)",
            "wf2_lie_inverse_phi_and_derivative(total_kappa, inverse_phi, inverse_phi_prime);",
            "total_m += total_beta * segment_alpha * rgb;",
            "node_chart_f32[chart_base + 0u] = total_kappa;",
        ):
            self.assertIn(required, forward_source)
        for required in (
            "for (uint node_id = 0u; node_id < node_count; ++node_id)",
            "const float cone_violation = max(",
            "const float loss_scale = config_f32[5];",
            "dot(diff, diff) * loss_scale",
            "const float3 grad_prediction = (2.0f * loss_scale) * diff;",
            "all(isfinite(target))",
            "atomic_fetch_add_explicit(&cone_diagnostic_i32[0], 1",
            "atomic_fetch_add_explicit(&loss_f32[0], invalid",
            "wf2_atomic_add4(\n        grad_node_chart_f32",
        ):
            self.assertIn(required, sample_math_source)
        self.assertNotIn("inv_element_count", sample_math_source)
        for required in (
            "const float3 bar_m = inverse_phi * grad_chart.yzw;",
            "const float bar_kappa_word =",
            "dot(bar_m, prefix_m + prefix_beta * rgb - total_m) + bar_kappa_word;",
            "prefix_beta * segment_alpha * bar_m",
            "prefix_m += prefix_beta * segment_alpha * rgb;",
        ):
            self.assertIn(required, vjp_source)
        helper_start = metal_source.index("static inline void wf2_lie_phi_and_derivative(")
        helper_end = metal_source.index("\nstatic inline bool wf2_bitset_has_boundary(", helper_start)
        helper_source = metal_source[helper_start:helper_end]
        for required in (
            "if (fabs(kappa) < 1.0e-4f)",
            "const float numerator = -expm1(-kappa);",
            "const float denominator = -expm1(-kappa);",
            "inverse_phi_prime = 0.5f + kappa / 6.0f",
        ):
            self.assertIn(required, helper_source)

        op_name = "fixed_word_p0_compiled_lie_transfer_mse_vjp_sparse_mobius_boundary"
        bindings_source = (variant / "csrc" / "bindings.cpp").read_text(encoding="utf-8")
        schema_tail = bindings_source.split(f'"{op_name}(', 1)[1].split('");', 1)[0]
        self.assertIn("Tensor sample_to_node_f32", schema_tail)
        self.assertIn("Tensor background_rgb_f32", schema_tail)
        self.assertIn(
            "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)",
            schema_tail,
        )
        host_source = (variant / "csrc" / "metal" / "world_foam_lane2_metal.mm").read_text(encoding="utf-8")
        function_start = host_source.index(f"metal_{op_name}(")
        function_end = host_source.index("\nFixedWordP0CompiledLieResult\nmetal_", function_start)
        function_source = host_source[function_start:function_end]
        for required in (
            'check_i32_mps_1d(config_i32, "config_i32", 8);',
            'config_f32.size(0) == 6, "config_f32 must have shape [6]"',
            "const int64_t word_count = config[5];",
            "const int64_t incidence_count = config[6];",
            "config[7] == incidence_count",
            "check_fixed_word_incidence_csr_cpu(",
            "auto config_i32_cpu = config_i32.cpu();",
            "auto config_f32_cpu = config_f32.cpu();",
            "scalar_config[5] > 0.0f",
            "return launch_fixed_word_p0_compiled_lie_prevalidated(",
        ):
            self.assertIn(required, function_source)

        core_start = host_source.index("launch_fixed_word_p0_compiled_lie_prevalidated(")
        core_end = host_source.index(f"\nmetal_{op_name}(", core_start)
        core_source = host_source[core_start:core_end]
        lower_launch = core_source.index("launch(k.sparse_mobius_incidence_lower")
        forward_launch = core_source.index("launch(k.fixed_word_p0_lie_node_forward")
        sample_launch = core_source.index("launch(k.fixed_word_p0_lie_sample_mse_vjp")
        node_vjp_launch = core_source.index("launch(k.fixed_word_p0_lie_node_vjp")
        boundary_vjp_launch = core_source.index("launch(k.sparse_mobius_incidence_boundary_vjp")
        self.assertLess(lower_launch, forward_launch)
        self.assertLess(forward_launch, sample_launch)
        self.assertLess(sample_launch, node_vjp_launch)
        self.assertLess(node_vjp_launch, boundary_vjp_launch)
        self.assertNotIn(".cpu()", core_source)
        self.assertNotIn("TORCH_CHECK", core_source)
        self.assertNotIn("frame_count", core_source)

        stage_names = (
            "fixed_word_p0_sparse_mobius_lower_launch_only",
            "fixed_word_p0_lie_node_forward_launch_only",
            "fixed_word_p0_lie_sample_state_init_launch_only",
            "fixed_word_p0_lie_sample_accumulate_launch_only",
            "fixed_word_p0_lie_sample_accumulate_loss_only_launch_only",
            "fixed_word_p0_lie_world_grad_init_launch_only",
            "fixed_word_p0_lie_material_world_grad_init_launch_only",
            "fixed_word_p0_lie_node_vjp_accumulate_launch_only",
            "fixed_word_p0_lie_material_node_vjp_accumulate_launch_only",
            "fixed_word_p0_sparse_mobius_boundary_finalize_launch_only",
        )
        for stage_name in stage_names:
            self.assertIn(f'"{stage_name}(', bindings_source)
            self.assertIn(f'"{stage_name}"', bindings_source)
            self.assertIn(f"metal_{stage_name}(", host_source)
        staged_host_source = "\n".join(
            _braced_definition(host_source, f"metal_{stage_name}(")
            for stage_name in stage_names
        )
        self.assertNotIn(".cpu()", staged_host_source)
        self.assertNotIn("TORCH_CHECK", staged_host_source)
        sample_stage_start = host_source.index("metal_fixed_word_p0_lie_sample_accumulate_launch_only(")
        sample_stage_end = host_source.index(
            "\nstd::tuple<torch::Tensor, torch::Tensor, torch::Tensor>\n"
            "metal_fixed_word_p0_lie_world_grad_init_launch_only(",
            sample_stage_start,
        )
        sample_stage_source = host_source[sample_stage_start:sample_stage_end]
        self.assertIn("launch(k.fixed_word_p0_lie_sample_mse_vjp", sample_stage_source)
        for forbidden in (
            "boundary_f32",
            "sparse_mobius_incidence_lower",
            "fixed_word_p0_lie_node_forward",
            "fixed_word_p0_lie_node_vjp",
            "sparse_mobius_incidence_boundary_vjp",
        ):
            self.assertNotIn(forbidden, sample_stage_source)
        ops_source = (variant / "torch_world_foam_lane2_fused_slab" / "ops.py").read_text(encoding="utf-8")
        self.assertIn(f"def {op_name}(", ops_source)
        self.assertIn("compiled Lie transfer scalar configuration must be finite", ops_source)
        preparation_start = ops_source.index(
            "def prepare_fixed_word_p0_compiled_lie_transfer_mse_vjp_sparse_mobius_boundary("
        )
        preparation_end = ops_source.index("\ndef ", preparation_start + 5)
        preparation_source = ops_source[preparation_start:preparation_end]
        for required in (
            "loss_scale: float",
            "cone_tolerance,\n        loss_scale,",
            "_CheckedFixedWordP0FusedCall(",
        ):
            self.assertIn(required, preparation_source)
        self.assertNotIn("site_rgba_f32[:, 3].detach().cpu()", preparation_source)
        warm_name = "launch_prepared_fixed_word_p0_compiled_lie_transfer_mse_vjp_sparse_mobius_boundary"
        warm_start = ops_source.index(f"def {warm_name}(")
        warm_end = ops_source.index("\ndef ", warm_start + 5)
        warm_source = ops_source[warm_start:warm_end]
        for forbidden in (".cpu()", ".contiguous()", "_require", "_validate", "hasattr"):
            self.assertNotIn(forbidden, warm_source)
        self.assertIn(
            "fixed_word_p0_compiled_lie_transfer_mse_vjp_sparse_mobius_boundary_launch_only(",
            warm_source,
        )
        for token_name in (
            "FixedWordP0TopologyToken",
            "FixedWordP0WorldRefreshToken",
            "FixedWordP0ChartToken",
            "FixedWordP0SampleStateToken",
            "FixedWordP0SampleBlockToken",
            "FixedWordP0WorldGradToken",
        ):
            self.assertIn(f"class {token_name}:", ops_source)
        topology_class_start = ops_source.index("class FixedWordP0TopologyToken:")
        topology_class_end = ops_source.index("\n@dataclass", topology_class_start)
        topology_class_source = ops_source[topology_class_start:topology_class_end]
        self.assertIn("every power site", topology_class_source)
        self.assertIn("topology_generation_id", topology_class_source)
        certificate_source = (variant / "torch_world_foam_lane2_fused_slab" / "certificate_binding.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("class NativeFixedWordP0ContinuousCertificateBinding:", certificate_source)
        self.assertIn("certify_prepared_adaptive_lie_world(", certificate_source)
        self.assertIn(
            'owner_identity_scope="all_competitor_sites_continuously_certified"',
            certificate_source,
        )
        self.assertIn("owner_identity_certified=True", certificate_source)
        self.assertIn("owner_identity_certificate_digest", certificate_source)
        self.assertIn("maximum_owner_difference_upper_bound", certificate_source)
        self.assertIn("total_owner_certificate_leaves", certificate_source)
        self.assertIn("class NativeFixedWordP0TrainingTopologyBinding:", certificate_source)
        self.assertIn('binding_mode="training_owner_topology_only"', certificate_source)
        self.assertIn("transfer_jacobian_certified=False", certificate_source)
        self.assertIn("approximation_error_certified=False", certificate_source)
        self.assertIn("paper_evidence_eligible=False", certificate_source)
        self.assertIn("live_site_rgba_refresh_allowed=True", certificate_source)
        self.assertIn(
            "def certify_and_bind_native_fixed_word_p0_training_topology(",
            certificate_source,
        )
        self.assertIn("certify_fixed_word_owner_identity(", certificate_source)
        self.assertIn("tensor.detach().cpu().clone().contiguous()", certificate_source)
        self.assertGreaterEqual(
            certificate_source.count("fit_derived_sample_to_node_weights("),
            2,
        )
        self.assertIn(
            '"verified_fit_derived_second_form_barycentric"',
            certificate_source,
        )
        self.assertNotIn("basis @ fit_matrix", certificate_source)
        training_certifier_start = certificate_source.index(
            "def certify_and_bind_native_fixed_word_p0_training_topology("
        )
        training_certifier_end = certificate_source.index(
            "\ndef _bind_native_training_topology(",
            training_certifier_start,
        )
        training_certifier_source = certificate_source[training_certifier_start:training_certifier_end]
        self.assertIn("certify_fixed_word_owner_identity(", training_certifier_source)
        self.assertNotIn("certify_prepared_adaptive_lie_world", training_certifier_source)
        self.assertNotIn("assert_native_fixed_word_p0_certificate_binding", training_certifier_source)

        topology_start = ops_source.index("def prepare_fixed_word_p0_topology_token(")
        world_start = ops_source.index("def refresh_fixed_word_p0_world_token(")
        chart_start = ops_source.index("def prepare_fixed_word_p0_chart_token(")
        state_start = ops_source.index("def prepare_fixed_word_p0_sample_state_token(")
        block_start = ops_source.index("def prepare_fixed_word_p0_sample_block_token(")
        sample_init_start = ops_source.index("def fixed_word_p0_lie_sample_state_init_launch_only(")
        sample_accumulate_start = ops_source.index("def fixed_word_p0_lie_sample_accumulate_launch_only(")
        topology_source = ops_source[topology_start:world_start]
        world_source = ops_source[world_start:chart_start]
        chart_source = ops_source[chart_start:state_start]
        state_source = ops_source[state_start:block_start]
        block_source = ops_source[block_start:sample_init_start]
        sample_accumulate_end = ops_source.index("\ndef ", sample_accumulate_start + 5)
        sample_accumulate_source = ops_source[sample_accumulate_start:sample_accumulate_end]

        self.assertIn("_validate_compact_referenced_boundaries_cpu(", topology_source)
        self.assertIn("assert_native_fixed_word_p0_runtime_binding(", topology_source)
        self.assertIn("certificate_binding.assert_native_topology(", topology_source)
        self.assertIn("tensor_signatures=_capture_tensor_signatures(topology_tensors)", topology_source)
        self.assertLess(
            world_source.index("sparse_power_boundary_from_sites_launch_only("),
            world_source.index("fixed_word_p0_sparse_mobius_lower_launch_only("),
        )
        self.assertNotIn("site_rgba_f32[:, 3]", world_source)
        self.assertNotIn(".cpu()", world_source)
        for required in (
            "certificate_binding.assert_native_chart(chart_index, compiler_node_t_f32)",
            'continuous_certificate_digest="" if training_mode else certificate_binding.canonical_digest',
            'training_binding_digest=certificate_binding.canonical_digest if training_mode else ""',
            "paper_evidence_eligible=certificate_binding.paper_evidence_eligible",
            "chart_generation_id=certified_chart.chart_digest",
            "runtime ``max(1, |C|+|D|)`` denominator guard remains active",
            "all-site owner/topology result",
        ):
            self.assertIn(required, chart_source)
        self.assertIn(
            "global_loss_scale=1.0 / float(global_loss_element_count)",
            state_source,
        )
        self.assertIn("expected_local_element_count", state_source)
        self.assertIn("global_track_count * global_sample_count * 3", state_source)
        self.assertIn("expected_block_count", state_source)
        self.assertIn("sample_block_size=sample_block_size", state_source)
        self.assertNotIn("expected_sample_blocks", state_source)
        self.assertNotIn("expected_block_ranges", state_source)
        self.assertNotIn("expected_block_ids", state_source)
        self.assertNotIn("global_sample_t_f64", state_source)
        self.assertNotIn("sample_t_f64", state_source)
        self.assertNotIn("sample_to_node_weights(", state_source)
        self.assertNotIn("global_loss_scale:", block_source)
        self.assertIn("loss_scale=sample_state.global_loss_scale", block_source)
        self.assertIn("sample_t_f64", block_source)
        self.assertIn("one finite CPU time per local sample", block_source)
        self.assertIn("sample_to_node_weight_result(", block_source)
        self.assertIn("sample_weight_linear_interactions", block_source)
        self.assertNotIn("sample_to_node_f32: Tensor", block_source)
        for forbidden in (
            ".cpu()",
            "_validate_fixed_word_incidence_csr_cpu",
            "_validate_track_boundary_incidence_csr_cpu",
            "boundary_site_pairs_i32",
            "site_rgba_f32[:, 3]",
        ):
            self.assertNotIn(forbidden, block_source)
        self.assertIn("sample_block.sample_state is not sample_state", sample_accumulate_source)
        self.assertNotIn(".cpu()", sample_accumulate_source)
        self.assertLess(topology_start, world_start)
        package_source = (variant / "torch_world_foam_lane2_fused_slab" / "__init__.py").read_text(encoding="utf-8")
        self.assertIn(op_name, package_source)

    def test_material_training_binding_is_separate_nonpaper_capability(self) -> None:
        variant = verify_mod.DEFAULT_VARIANT_ROOT / "world_foam_lane2_fused_slab_v0"
        sys.path.insert(0, str(variant))
        try:
            from native_track_adapter import _assert_binding_matches_prepared
            from staged_compiled_lie_adjoint import prepare_compact_staged_lie_world_snapshot
            from torch_world_foam_lane2_fused_slab import ops as native_ops
            from torch_world_foam_lane2_fused_slab.certificate_binding import (
                assert_native_fixed_word_p0_certificate_binding,
                assert_native_fixed_word_p0_runtime_binding,
                assert_native_fixed_word_p0_training_topology_binding,
                certify_and_bind_native_fixed_word_p0_training_topology,
                derive_native_fixed_word_p0_training_topology_binding,
            )

            strict, prepared = _native_continuous_binding_fixture()
        finally:
            sys.path.remove(str(variant))

        training = certify_and_bind_native_fixed_word_p0_training_topology(prepared)
        narrowed = derive_native_fixed_word_p0_training_topology_binding(strict)
        self.assertEqual(training.topology_snapshot_generation, narrowed.topology_snapshot_generation)
        self.assertEqual(training.training_snapshot_generation, narrowed.training_snapshot_generation)
        self.assertFalse(hasattr(training, "_prepared"))
        self.assertGreater(training.resident_immutable_bytes, 0)
        self.assertNotIn("site_density", training._bound_tensor_names)
        self.assertNotIn("site_color", training._bound_tensor_names)
        self.assertFalse(any(name.endswith(".coefficients") for name in training._bound_tensor_names))
        self.assertFalse(any("node_chart" in name for name in training._bound_tensor_names))
        original_by_name = dict(
            zip(
                training._bound_tensor_names,
                (
                    *(getattr(prepared.topology, name) for name in training._bound_tensor_names[:10]),
                    prepared.site_geometry,
                    prepared.world_snapshot.ray_coefficients,
                    prepared.world_snapshot.atlas.charts[0].transfer_atlas.node_times,
                    prepared.world_snapshot.atlas.charts[0].transfer_atlas.fit_matrix,
                ),
                strict=True,
            )
        )
        for name, bound_tensor in zip(
            training._bound_tensor_names,
            training._bound_tensors,
            strict=True,
        ):
            original = original_by_name[name]
            if bound_tensor.numel():
                self.assertNotEqual(
                    bound_tensor.untyped_storage().data_ptr(),
                    original.untyped_storage().data_ptr(),
                )
        self.assertEqual(training.binding_mode, "training_owner_topology_only")
        self.assertTrue(training.owner_identity_certified)
        self.assertTrue(training.geometry_rays_immutable)
        self.assertTrue(training.live_site_rgba_refresh_allowed)
        self.assertFalse(training.transfer_jacobian_certified)
        self.assertFalse(training.approximation_error_certified)
        self.assertFalse(training.paper_evidence_eligible)
        self.assertEqual(len(training.canonical_digest), 64)
        self.assertEqual(len(training.training_snapshot_generation), 64)
        self.assertEqual(
            training.sample_weight_evaluation,
            "verified_fit_derived_second_form_barycentric",
        )
        sample_times = torch.tensor([-0.4, 0.0, 0.4], dtype=torch.float64)
        strict_weight_result = strict.sample_to_node_weight_result(0, sample_times)
        training_weight_result = training.sample_to_node_weight_result(0, sample_times)
        self.assertEqual(
            training_weight_result.evaluation,
            "verified_fit_derived_second_form_barycentric",
        )
        self.assertEqual(training_weight_result.linear_weight_interactions, 3 * 4)
        self.assertEqual(training_weight_result.dense_fallback_interactions, 0)
        torch.testing.assert_close(
            training_weight_result.weights,
            strict_weight_result.weights,
            atol=0.0,
            rtol=0.0,
        )
        self.assertIs(assert_native_fixed_word_p0_runtime_binding(training), training)
        self.assertIs(
            assert_native_fixed_word_p0_training_topology_binding(training),
            training,
        )

        source = prepared.topology
        topology_tensors = tuple(
            tensor.clone()
            for tensor in (
                source.word_offsets_i32,
                source.word_owner_i32,
                source.word_left_incidence_i32,
                source.word_right_incidence_i32,
                source.track_incidence_offsets_i32,
                source.incidence_boundary_i32,
                source.boundary_site_pairs_i32,
            )
        )
        active_pairs = torch.cat(
            (
                torch.arange(source.boundary_count, dtype=torch.int32)[:, None],
                topology_tensors[-1],
            ),
            dim=1,
        )
        training_topology = native_ops.FixedWordP0TopologyToken(
            word_offsets_i32=topology_tensors[0],
            word_owner_i32=topology_tensors[1],
            word_left_incidence_i32=topology_tensors[2],
            word_right_incidence_i32=topology_tensors[3],
            track_incidence_offsets_i32=topology_tensors[4],
            incidence_boundary_i32=topology_tensors[5],
            boundary_site_pairs_i32=topology_tensors[6],
            active_boundary_site_pairs_i32=active_pairs,
            certificate_binding=training,
            continuous_certificate_digest="",
            topology_generation_id=training.topology_snapshot_generation,
            boundary_count=source.boundary_count,
            track_count=source.track_count,
            site_count=source.site_count,
            word_count=source.word_count,
            incidence_count=source.incidence_count,
            tensor_signatures=native_ops._capture_tensor_signatures((*topology_tensors, active_pairs)),
            training_binding_digest=training.canonical_digest,
            binding_mode=training.binding_mode,
            paper_evidence_eligible=False,
            transfer_jacobian_certified=False,
        )
        native_ops._assert_topology_token_current(training_topology)
        with self.assertRaisesRegex(ValueError, "cannot claim a continuous transfer certificate"):
            native_ops._assert_topology_token_current(
                replace(
                    training_topology,
                    continuous_certificate_digest=training.canonical_digest,
                    training_binding_digest="",
                )
            )
        with self.assertRaisesRegex(ValueError, "paper-evidence eligibility changed"):
            native_ops._assert_topology_token_current(replace(training_topology, paper_evidence_eligible=True))

        snapshot = prepared.world_snapshot
        sites = prepared.site_geometry.to(dtype=torch.float32).clone()
        rays = snapshot.ray_coefficients.to(dtype=torch.float32).clone()
        frozen_rgba = torch.cat(
            (snapshot.site_color, snapshot.site_density[:, None]),
            dim=1,
        ).to(dtype=torch.float32)
        live_rgba = frozen_rgba + torch.tensor([0.1, -0.05, 0.02, 0.2])
        training.assert_native_world(
            sites_f32=sites,
            site_rgba_f32=live_rgba,
            track_ray_coeff_f32=rays,
        )
        with self.assertRaisesRegex(ValueError, "certified compact snapshot"):
            strict.assert_native_world(
                sites_f32=sites,
                site_rgba_f32=live_rgba,
                track_ray_coeff_f32=rays,
            )

        refreshed_prepared = []
        for step, material_delta in ((2, 0.15), (3, 0.3)):
            next_density = prepared.source_tensors[2].detach().clone() + material_delta
            next_color = prepared.source_tensors[3].detach().clone() + material_delta / 2.0
            next_prepared = prepare_compact_staged_lie_world_snapshot(
                prepared.template,
                prepared.topology,
                site_geometry=prepared.source_tensors[0],
                ray_coefficients=prepared.source_tensors[1],
                site_density=next_density,
                site_color=next_color,
            )
            with self.subTest(material_step=step):
                training.assert_prepared_immutable(next_prepared)
                _assert_binding_matches_prepared(training, next_prepared)
                next_rgba = torch.cat(
                    (
                        next_prepared.world_snapshot.site_color,
                        next_prepared.world_snapshot.site_density[:, None],
                    ),
                    dim=1,
                ).to(dtype=torch.float32)
                training.assert_native_world(
                    sites_f32=next_prepared.site_geometry.to(dtype=torch.float32),
                    site_rgba_f32=next_rgba,
                    track_ray_coeff_f32=(next_prepared.world_snapshot.ray_coefficients.to(dtype=torch.float32)),
                )
            refreshed_prepared.append(next_prepared)
        with self.assertRaisesRegex(ValueError, "different compact prepared snapshot"):
            _assert_binding_matches_prepared(strict, refreshed_prepared[-1])

        changed_sites = sites.clone()
        changed_sites[0, 0] += 0.01
        with self.assertRaisesRegex(ValueError, "certified compact snapshot"):
            training.assert_native_world(
                sites_f32=changed_sites,
                site_rgba_f32=live_rgba,
                track_ray_coeff_f32=rays,
            )
        changed_rays = rays.clone()
        changed_rays[0, 0] += 0.01
        with self.assertRaisesRegex(ValueError, "certified compact snapshot"):
            training.assert_native_world(
                sites_f32=sites,
                site_rgba_f32=live_rgba,
                track_ray_coeff_f32=changed_rays,
            )

        with self.assertRaisesRegex(ValueError, "sealed native continuous certificate binding"):
            assert_native_fixed_word_p0_certificate_binding(training)
        with self.assertRaisesRegex(ValueError, "fabricated outside an owner certifier"):
            assert_native_fixed_word_p0_training_topology_binding(replace(training, _seal=object()))
        with self.assertRaisesRegex(ValueError, "must remain.*non-paper"):
            assert_native_fixed_word_p0_training_topology_binding(replace(training, paper_evidence_eligible=True))
        with self.assertRaisesRegex(ValueError, "cannot present itself as strict"):
            assert_native_fixed_word_p0_training_topology_binding(
                replace(training, binding_mode="strict_frozen_evaluation")
            )

        # Material state is deliberately outside this capability's immutable
        # tensor set; mutating it does not weaken the retained owner certificate.
        prepared.world_snapshot.site_density.add_(0.25)
        prepared.world_snapshot.site_color.add_(0.1)
        training.assert_current()
        _assert_binding_matches_prepared(training, prepared)

    def test_fixed_word_cpu_contract_rejects_unstable_or_nonlocal_rows(self) -> None:
        variant = verify_mod.DEFAULT_VARIANT_ROOT / "world_foam_lane2_fused_slab_v0"
        sys.path.insert(0, str(variant))
        try:
            from torch_world_foam_lane2_fused_slab.ops import _validate_fixed_word_incidence_csr_cpu
        finally:
            sys.path.remove(str(variant))

        valid = {
            "word_offsets_i32": torch.tensor([0, 3, 5], dtype=torch.int32),
            "word_owner_i32": torch.tensor([0, 1, 2, 2, 0], dtype=torch.int32),
            "word_left_incidence_i32": torch.tensor([-1, 0, 1, -1, 0], dtype=torch.int32),
            "word_right_incidence_i32": torch.tensor([0, 1, -2, 0, -2], dtype=torch.int32),
            "track_incidence_offsets_i32": torch.tensor([0, 2, 3], dtype=torch.int32),
            "track_count": 2,
            "site_count": 3,
        }
        _validate_fixed_word_incidence_csr_cpu(**valid)
        invalid_cases = (
            (
                "adjacent stable ordered word",
                {"word_left_incidence_i32": torch.tensor([-1, 1, 1, -1, 0], dtype=torch.int32)},
            ),
            (
                "row-local incidence id",
                {"word_right_incidence_i32": torch.tensor([0, 2, -2, 0, -2], dtype=torch.int32)},
            ),
            (
                "owner ids",
                {"word_owner_i32": torch.tensor([0, 1, 3, 2, 0], dtype=torch.int32)},
            ),
            (
                "nonempty",
                {
                    "word_offsets_i32": torch.tensor([0, 0, 5], dtype=torch.int32),
                    "word_left_incidence_i32": torch.tensor([-1, 0, 1, -1, 0], dtype=torch.int32),
                },
            ),
        )
        for message, replacement in invalid_cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                _validate_fixed_word_incidence_csr_cpu(**{**valid, **replacement})

    def test_compiled_lie_global_loss_scale_is_partition_invariant(self) -> None:
        variant = verify_mod.DEFAULT_VARIANT_ROOT / "world_foam_lane2_fused_slab_v0"
        sys.path.insert(0, str(variant))
        try:
            from torch_world_foam_lane2_fused_slab.ops import scaled_rgb_mse_reference
        finally:
            sys.path.remove(str(variant))

        generator = torch.Generator().manual_seed(41)
        prediction = torch.randn((5, 7, 3), generator=generator, dtype=torch.float64)
        target = torch.randn((5, 7, 3), generator=generator, dtype=torch.float64)
        global_scale = 1.0 / prediction.numel()
        full_loss, full_grad = scaled_rgb_mse_reference(
            prediction,
            target,
            loss_scale=global_scale,
        )
        block_loss = torch.zeros((), dtype=torch.float64)
        block_grad = torch.empty_like(full_grad)
        for track_slice in (slice(0, 2), slice(2, 5)):
            for sample_slice in (slice(0, 1), slice(1, 4), slice(4, 7)):
                loss, grad = scaled_rgb_mse_reference(
                    prediction[track_slice, sample_slice],
                    target[track_slice, sample_slice],
                    loss_scale=global_scale,
                )
                block_loss += loss
                block_grad[track_slice, sample_slice] = grad
        torch.testing.assert_close(block_loss, full_loss)
        torch.testing.assert_close(block_grad, full_grad)

        incorrectly_local_normalized = torch.zeros((), dtype=torch.float64)
        for track_slice in (slice(0, 2), slice(2, 5)):
            for sample_slice in (slice(0, 1), slice(1, 4), slice(4, 7)):
                local_prediction = prediction[track_slice, sample_slice]
                local_loss, _ = scaled_rgb_mse_reference(
                    local_prediction,
                    target[track_slice, sample_slice],
                    loss_scale=1.0 / local_prediction.numel(),
                )
                incorrectly_local_normalized += local_loss
        self.assertFalse(torch.isclose(incorrectly_local_normalized, full_loss))

    def test_native_streaming_tokens_fail_closed_on_stale_mixed_or_incomplete_state(self) -> None:
        variant = verify_mod.DEFAULT_VARIANT_ROOT / "world_foam_lane2_fused_slab_v0"
        sys.path.insert(0, str(variant))
        try:
            from torch_world_foam_lane2_fused_slab import ops as native_ops
            from torch_world_foam_lane2_fused_slab.certificate_binding import (
                assert_native_fixed_word_p0_certificate_binding,
            )

            binding, prepared = _native_continuous_binding_fixture()
        finally:
            sys.path.remove(str(variant))

        def make_topology():
            source = prepared.topology
            word_offsets = source.word_offsets_i32.clone()
            word_owner = source.word_owner_i32.clone()
            word_left = source.word_left_incidence_i32.clone()
            word_right = source.word_right_incidence_i32.clone()
            track_incidence_offsets = source.track_incidence_offsets_i32.clone()
            incidence_boundary = source.incidence_boundary_i32.clone()
            boundary_pairs = source.boundary_site_pairs_i32.clone()
            active_pairs = torch.cat(
                (
                    torch.arange(source.boundary_count, dtype=torch.int32)[:, None],
                    boundary_pairs,
                ),
                dim=1,
            )
            tensors = (
                word_offsets,
                word_owner,
                word_left,
                word_right,
                track_incidence_offsets,
                incidence_boundary,
                boundary_pairs,
                active_pairs,
            )
            return native_ops.FixedWordP0TopologyToken(
                word_offsets_i32=word_offsets,
                word_owner_i32=word_owner,
                word_left_incidence_i32=word_left,
                word_right_incidence_i32=word_right,
                track_incidence_offsets_i32=track_incidence_offsets,
                incidence_boundary_i32=incidence_boundary,
                boundary_site_pairs_i32=boundary_pairs,
                active_boundary_site_pairs_i32=active_pairs,
                certificate_binding=binding,
                continuous_certificate_digest=binding.canonical_digest,
                topology_generation_id=binding.topology_snapshot_generation,
                boundary_count=source.boundary_count,
                track_count=source.track_count,
                site_count=source.site_count,
                word_count=source.word_count,
                incidence_count=source.incidence_count,
                tensor_signatures=native_ops._capture_tensor_signatures(tensors),
            )

        def make_world(topology):
            snapshot = prepared.world_snapshot
            sites = prepared.site_geometry.to(dtype=torch.float32).clone()
            site_rgba = torch.cat(
                (snapshot.site_color, snapshot.site_density[:, None]),
                dim=1,
            ).to(dtype=torch.float32)
            rays = snapshot.ray_coefficients.to(dtype=torch.float32).clone()
            boundary = native_ops.sparse_power_boundary_from_sites_reference(
                topology.boundary_site_pairs_i32,
                sites,
            )
            mobius = torch.zeros((topology.incidence_count, 4), dtype=torch.float32)
            config_i32 = torch.tensor(
                [
                    topology.boundary_count,
                    topology.track_count,
                    1,
                    1,
                    topology.site_count,
                    topology.word_count,
                    topology.incidence_count,
                    topology.incidence_count,
                ],
                dtype=torch.int32,
            )
            tensors = (sites, site_rgba, rays, boundary, mobius, config_i32)
            return native_ops.FixedWordP0WorldRefreshToken(
                topology=topology,
                sites_f32=sites,
                site_rgba_f32=site_rgba,
                track_ray_coeff_f32=rays,
                boundary_f32=boundary,
                mobius_coeff_f32=mobius,
                config_i32=config_i32,
                config=native_ops.RealRayReplayConfig(near=0.1, far=0.9),
                physical_length_epsilon=1.0e-8,
                cone_tolerance=1.0e-6,
                world_generation_id=binding.world_snapshot_generation,
                tensor_signatures=native_ops._capture_tensor_signatures(tensors),
            )

        def make_chart(world):
            certified = binding.charts[0]
            compiler_nodes = prepared.world_snapshot.atlas.charts[certified.chart_index].transfer_atlas.node_times.to(
                dtype=torch.float32
            )
            node_chart = torch.zeros(
                (world.topology.track_count, certified.node_count, 4),
                dtype=torch.float32,
            )
            config_i32 = torch.tensor(
                [
                    world.topology.boundary_count,
                    world.topology.track_count,
                    certified.node_count,
                    1,
                    world.topology.site_count,
                    world.topology.word_count,
                    world.topology.incidence_count,
                    world.topology.incidence_count,
                ],
                dtype=torch.int32,
            )
            config_f32 = torch.tensor(
                [0.1, 0.9, 1.0e-7, 1.0e-8, 1.0e-6, 1.0],
                dtype=torch.float32,
            )
            tensors = (compiler_nodes, node_chart, config_i32, config_f32)
            return native_ops.FixedWordP0ChartToken(
                world=world,
                compiler_node_t_f32=compiler_nodes,
                node_chart_f32=node_chart,
                config_i32=config_i32,
                config_f32=config_f32,
                continuous_certificate_digest=binding.canonical_digest,
                chart_index=certified.chart_index,
                chart_generation_id=certified.chart_digest,
                node_count=certified.node_count,
                tensor_signatures=native_ops._capture_tensor_signatures(tensors),
            )

        def make_state(
            chart,
            *,
            start: int,
            end: int,
            block_id: str,
            global_track_count: int = 2,
            global_sample_count: int = 5,
            normalization: str = "normalization-a",
            partition: str = "partition-a",
            sample_block_size: int | None = None,
        ):
            loss = torch.zeros(1, dtype=torch.float32)
            grad_node = torch.zeros(
                (chart.world.topology.track_count, chart.node_count, 4),
                dtype=torch.float32,
            )
            diagnostic = torch.zeros(3, dtype=torch.int32)
            normalized_block_size = end - start if sample_block_size is None else sample_block_size
            ledger = native_ops._FixedWordP0SampleLedger(
                sample_block_size=normalized_block_size,
                expected_block_count=(end - start + normalized_block_size - 1) // normalized_block_size,
                next_global_sample_start=start,
            )
            ledger.state_tensor_signatures = native_ops._capture_tensor_signatures((loss, grad_node, diagnostic))
            return native_ops.FixedWordP0SampleStateToken(
                chart=chart,
                loss_f32=loss,
                grad_node_chart_f32=grad_node,
                cone_diagnostic_i32=diagnostic,
                loss_normalization_id=normalization,
                sample_partition_generation_id=partition,
                global_track_count=global_track_count,
                global_sample_count=global_sample_count,
                global_sample_start=start,
                global_sample_end=end,
                global_loss_element_count=global_track_count * global_sample_count * 3,
                expected_local_element_count=(end - start) * 3,
                global_loss_scale=1.0 / float(global_track_count * global_sample_count * 3),
                ledger=ledger,
            )

        def make_block(state, block_id: str, start: int, end: int):
            block_times = torch.linspace(
                -0.4,
                0.4,
                state.global_sample_count,
                dtype=torch.float64,
            )[start:end]
            sample_weight_result = binding.sample_to_node_weight_result(
                state.chart.chart_index,
                block_times,
            )
            sample_to_node = sample_weight_result.weights.to(dtype=torch.float32)
            target = torch.zeros(
                (state.chart.world.topology.track_count, end - start, 3),
                dtype=torch.float32,
            )
            background = torch.zeros(3, dtype=torch.float32)
            topology = state.chart.world.topology
            config_i32 = torch.tensor(
                [
                    topology.boundary_count,
                    topology.track_count,
                    state.chart.node_count,
                    end - start,
                    topology.site_count,
                    topology.word_count,
                    topology.incidence_count,
                    topology.incidence_count,
                ],
                dtype=torch.int32,
            )
            config_f32 = torch.tensor(
                [0.1, 0.9, 1.0e-7, 1.0e-8, 1.0e-6, state.global_loss_scale],
                dtype=torch.float32,
            )
            tensors = (sample_to_node, target, background, config_i32, config_f32)
            return native_ops.FixedWordP0SampleBlockToken(
                sample_state=state,
                sample_to_node_f32=sample_to_node,
                target_rgb_f32=target,
                background_rgb_f32=background,
                config_i32=config_i32,
                config_f32=config_f32,
                sample_block_id=block_id,
                global_sample_start=start,
                global_sample_end=end,
                sample_count=end - start,
                element_count=(end - start) * 3,
                sample_weight_evaluation=sample_weight_result.evaluation,
                sample_weight_linear_interactions=(sample_weight_result.linear_weight_interactions),
                sample_weight_dense_fallback_interactions=(sample_weight_result.dense_fallback_interactions),
                sample_weight_exact_node_rows=sample_weight_result.exact_node_row_count,
                sample_weight_dense_fallback_rows=(sample_weight_result.dense_fallback_row_count),
                tensor_signatures=native_ops._capture_tensor_signatures(tensors),
            )

        def make_world_grad(
            token_world,
            *,
            ranges=None,
            normalization: str = "normalization-a",
            partition: str = "partition-a",
        ):
            if ranges is None:
                ranges = ((binding.charts[0].chart_digest, 0, 5),)
            grad_rgba = torch.zeros((token_world.topology.site_count, 4), dtype=torch.float32)
            grad_mobius = torch.zeros(
                (token_world.topology.incidence_count, 4),
                dtype=torch.float32,
            )
            grad_boundary = torch.zeros(
                (token_world.topology.boundary_count, 5),
                dtype=torch.float32,
            )
            ledger = native_ops._FixedWordP0WorldGradLedger(
                expected_chart_generation_ids=frozenset(chart_id for chart_id, _, _ in ranges),
                expected_chart_ranges=ranges,
            )
            ledger.grad_tensor_signatures = native_ops._capture_tensor_signatures(
                (grad_rgba, grad_mobius, grad_boundary)
            )
            return native_ops.FixedWordP0WorldGradToken(
                world=token_world,
                grad_site_rgba_f32=grad_rgba,
                grad_mobius_coeff_f32=grad_mobius,
                grad_boundary_f32=grad_boundary,
                loss_normalization_id=normalization,
                sample_partition_generation_id=partition,
                global_track_count=2,
                global_sample_count=5,
                global_loss_element_count=30,
                ledger=ledger,
            )

        topology = make_topology()
        world = make_world(topology)
        chart_a = make_chart(world)
        self.assertTrue(binding.passed)
        self.assertEqual(len(binding.canonical_digest), 64)
        self.assertEqual(binding.topology_snapshot_generation, topology.topology_generation_id)
        self.assertEqual(binding.world_snapshot_generation, world.world_generation_id)
        self.assertTrue(binding.owner_identity_certified)
        self.assertEqual(
            binding.owner_identity_scope,
            "all_competitor_sites_continuously_certified",
        )
        self.assertGreater(binding.total_owner_certificate_leaves, 0)
        self.assertLessEqual(
            binding.maximum_owner_difference_upper_bound,
            binding.owner_identity_tolerance,
        )
        self.assertEqual(len(binding.charts[0].owner_identity_certificate_digest), 64)
        self.assertFalse(binding.runtime_floating_point_roundoff_certified)
        self.assertEqual(binding.charts[0].node_count, chart_a.node_count)
        self.assertGreater(binding.charts[0].estimated_interval_jet_work_units, 0)
        self.assertEqual(
            binding.sample_weight_evaluation,
            "verified_fit_derived_second_form_barycentric",
        )

        for rejected, message in (
            (replace(binding, passed=False), "did not pass"),
            (replace(binding, canonical_digest="0" * 64), "canonical digest is invalid"),
            (replace(binding, _seal=object()), "fabricated outside the certifier"),
            (replace(binding, owner_identity_certified=False), "does not certify owner identity"),
            (
                replace(binding, topology_snapshot_generation="0" * 64),
                "exposed facts do not match",
            ),
        ):
            with self.subTest(binding_rejection=message), self.assertRaisesRegex(ValueError, message):
                assert_native_fixed_word_p0_certificate_binding(rejected)

        topology_kwargs = {
            "word_offsets_i32": topology.word_offsets_i32,
            "word_owner_i32": topology.word_owner_i32,
            "word_left_incidence_i32": topology.word_left_incidence_i32,
            "word_right_incidence_i32": topology.word_right_incidence_i32,
            "track_incidence_offsets_i32": topology.track_incidence_offsets_i32,
            "incidence_boundary_i32": topology.incidence_boundary_i32,
            "boundary_site_pairs_i32": topology.boundary_site_pairs_i32,
        }
        wrong_pairs = topology.boundary_site_pairs_i32.flip(1)
        with self.assertRaisesRegex(ValueError, "certified compact snapshot"):
            binding.assert_native_topology(**{**topology_kwargs, "boundary_site_pairs_i32": wrong_pairs})
        wrong_sites = world.sites_f32.clone()
        wrong_sites[0, 0] += 1.0
        with self.assertRaisesRegex(ValueError, "certified compact snapshot"):
            binding.assert_native_world(
                sites_f32=wrong_sites,
                site_rgba_f32=world.site_rgba_f32,
                track_ray_coeff_f32=world.track_ray_coeff_f32,
            )
        wrong_nodes = chart_a.compiler_node_t_f32.clone()
        wrong_nodes[0] += 0.01
        with self.assertRaisesRegex(ValueError, "certified compact snapshot"):
            binding.assert_native_chart(0, wrong_nodes)
        with self.assertRaisesRegex(ValueError, "leave certified chart interval"):
            binding.sample_to_node_weights(0, torch.tensor([0.6], dtype=torch.float64))

        sample_times = torch.tensor([-0.4, 0.0, 0.4], dtype=torch.float64)
        weights = binding.sample_to_node_weights(0, sample_times)
        weight_result = binding.sample_to_node_weight_result(0, sample_times)
        self.assertEqual(
            weight_result.evaluation,
            "verified_fit_derived_second_form_barycentric",
        )
        self.assertEqual(weight_result.linear_weight_interactions, 3 * 4)
        self.assertEqual(weight_result.dense_fallback_interactions, 0)
        torch.testing.assert_close(weights, weight_result.weights.to(dtype=torch.float32))
        chart_certificate = binding.charts[0]
        x = (2.0 * sample_times - (chart_certificate.t_max + chart_certificate.t_min)) / (
            chart_certificate.t_max - chart_certificate.t_min
        )
        basis = torch.stack(
            (torch.ones_like(x), x, 2.0 * x.square() - 1.0, 4.0 * x.pow(3) - 3.0 * x),
            dim=1,
        )
        expected_weights = basis @ prepared.world_snapshot.atlas.charts[0].transfer_atlas.fit_matrix
        torch.testing.assert_close(weights, expected_weights.to(dtype=torch.float32))

        stale_topology = make_topology()
        stale_topology.boundary_site_pairs_i32[0, 0] = 1
        with self.assertRaisesRegex(ValueError, "topology token tensor storage or mutation version changed"):
            native_ops._assert_topology_token_current(stale_topology)
        stale_world = make_world(make_topology())
        stale_world.sites_f32[0, 0] = 1.0
        with self.assertRaisesRegex(ValueError, "world refresh token tensor storage or mutation version changed"):
            native_ops._assert_world_token_current(stale_world)
        stale_chart = make_chart(make_world(make_topology()))
        stale_chart.compiler_node_t_f32[0] = 0.25
        with self.assertRaisesRegex(ValueError, "chart token tensor storage or mutation version changed"):
            native_ops._assert_chart_token_current(stale_chart)

        state_a = make_state(chart_a, start=0, end=2, block_id="a")
        block_a = make_block(state_a, "a", 0, 2)
        state_b = make_state(chart_a, start=0, end=2, block_id="a")
        with self.assertRaisesRegex(ValueError, "different chart/normalization state"):
            native_ops.fixed_word_p0_lie_sample_accumulate_launch_only(block_a, state_b)
        changed_scale_state = replace(state_a, global_loss_scale=0.5)
        with self.assertRaisesRegex(ValueError, "different chart/normalization state"):
            native_ops.fixed_word_p0_lie_sample_accumulate_launch_only(block_a, changed_scale_state)
        block_a.config_f32[5] = 0.5
        with self.assertRaisesRegex(ValueError, "sample block token tensor storage or mutation version changed"):
            native_ops._assert_sample_block_current(block_a)

        missing = make_state(chart_a, start=0, end=2, block_id="missing")
        with self.assertRaisesRegex(ValueError, "missing K blocks"):
            native_ops._require_sample_state_ready_for_reverse(missing)
        complete_state_a = make_state(chart_a, start=0, end=5, block_id="chart-a")
        complete_state_a.ledger.consumed_block_count = complete_state_a.ledger.expected_block_count
        complete_state_a.ledger.next_global_sample_start = complete_state_a.global_sample_end
        complete_state_a.ledger.accumulated_element_count = complete_state_a.expected_local_element_count
        native_ops._require_sample_state_ready_for_reverse(complete_state_a)
        self.assertEqual(complete_state_a.global_loss_scale, 1.0 / 30.0)

        sequential = make_state(
            chart_a,
            start=0,
            end=5,
            block_id="unused-by-constant-ledger",
            sample_block_size=2,
        )
        self.assertEqual(sequential.ledger.expected_block_count, 3)
        for forbidden in ("expected_block_ids", "expected_block_ranges", "consumed_block_ids"):
            self.assertFalse(hasattr(sequential.ledger, forbidden))
        native_ops._assert_next_sample_block_range(
            sequential,
            global_sample_start=0,
            global_sample_end=2,
        )
        sequential.ledger.consumed_block_count = 1
        sequential.ledger.next_global_sample_start = 2
        with self.assertRaisesRegex(ValueError, "next deterministic K partition"):
            native_ops._assert_next_sample_block_range(
                sequential,
                global_sample_start=0,
                global_sample_end=2,
            )
        with self.assertRaisesRegex(ValueError, "next deterministic K partition"):
            native_ops._assert_next_sample_block_range(
                sequential,
                global_sample_start=3,
                global_sample_end=5,
            )
        native_ops._assert_next_sample_block_range(
            sequential,
            global_sample_start=2,
            global_sample_end=4,
        )
        sequential.ledger.consumed_block_count = 2
        sequential.ledger.next_global_sample_start = 4
        native_ops._assert_next_sample_block_range(
            sequential,
            global_sample_start=4,
            global_sample_end=5,
        )

        with self.assertRaisesRegex(ValueError, "different loss normalization id"):
            native_ops.fixed_word_p0_lie_node_vjp_accumulate_launch_only(
                chart_a,
                complete_state_a,
                make_world_grad(world, normalization="wrong-normalization"),
            )
        with self.assertRaisesRegex(ValueError, "different sample partition generation"):
            native_ops.fixed_word_p0_lie_node_vjp_accumulate_launch_only(
                chart_a,
                complete_state_a,
                make_world_grad(world, partition="wrong-partition"),
            )
        with self.assertRaisesRegex(ValueError, "registered global partition"):
            native_ops.fixed_word_p0_lie_node_vjp_accumulate_launch_only(
                chart_a,
                complete_state_a,
                make_world_grad(
                    world,
                    ranges=((binding.charts[0].chart_digest, 0, 3), ("other-chart", 3, 5)),
                ),
            )
        with self.assertRaisesRegex(ValueError, "different world refresh"):
            native_ops.fixed_word_p0_lie_node_vjp_accumulate_launch_only(
                chart_a,
                complete_state_a,
                make_world_grad(make_world(make_topology())),
            )
        incomplete_world_grad = make_world_grad(world)
        with self.assertRaisesRegex(ValueError, "missing chart reversals"):
            native_ops.fixed_word_p0_sparse_mobius_boundary_finalize_launch_only(incomplete_world_grad)
        with self.assertRaisesRegex(ValueError, "before the boundary adjoint"):
            native_ops.fixed_word_p0_site_geometry_finalize_launch_only(incomplete_world_grad)

        for records in (
            (("chart-a", 0, 2), ("chart-b", 3, 5)),
            (("chart-a", 0, 3), ("chart-b", 2, 5)),
        ):
            with self.assertRaisesRegex(ValueError, "without gaps or overlaps"):
                native_ops._validate_half_open_partition(
                    records,
                    expected_start=0,
                    expected_end=5,
                    name="chart partition",
                )
        self.assertEqual(
            native_ops._validate_half_open_partition(
                (("chart-b", 2, 5), ("chart-a", 0, 2)),
                expected_start=0,
                expected_end=5,
                name="chart partition",
            ),
            (("chart-a", 0, 2), ("chart-b", 2, 5)),
        )
        native_ops._validate_compact_referenced_boundaries_cpu(
            torch.tensor([0, 1], dtype=torch.int32),
            boundary_count=2,
        )
        with self.assertRaisesRegex(ValueError, "exact compact referenced boundary table"):
            native_ops._validate_compact_referenced_boundaries_cpu(
                torch.tensor([0], dtype=torch.int32),
                boundary_count=2,
            )

        prepared.site_geometry[0, 0] += 1.0
        with self.assertRaisesRegex(ValueError, "prepared|snapshot tensors changed"):
            assert_native_fixed_word_p0_certificate_binding(binding)

    def test_sparse_mobius_denominator_guard_matches_cpu_certificate(self) -> None:
        from compiled_lie_world_adjoint import _cut_depth_and_jacobian

        # The old |C|+|D*t| scale accepted this at t=0 because its threshold
        # collapsed with C. The certified max(1, |C|+|D|) scale rejects it.
        near_parallel = torch.tensor([[0.0, 0.0, 5.0e-10, 1.0]], dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "denominator is unsafe"):
            _cut_depth_and_jacobian(
                {0: 0},
                near_parallel,
                0,
                torch.tensor(0.0, dtype=torch.float64),
                near=-1.0,
                far=1.0,
            )

        cancellation = torch.tensor([[0.5, 0.0, 1.0, -1.0]], dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "denominator is unsafe"):
            _cut_depth_and_jacobian(
                {0: 0},
                cancellation,
                0,
                torch.tensor(1.0, dtype=torch.float64),
                near=-2.0,
                far=2.0,
            )

        safe = torch.tensor([[0.5, 0.1, 2.0, 0.25]], dtype=torch.float64)
        depth, jacobian = _cut_depth_and_jacobian(
            {0: 0},
            safe,
            0,
            torch.tensor(0.25, dtype=torch.float64),
            near=-2.0,
            far=2.0,
        )
        self.assertTrue(bool(torch.isfinite(depth).item()))
        self.assertTrue(bool(torch.isfinite(jacobian).all().item()))

    def test_compiled_lie_density_contract_is_physical_and_one_sided_at_zero(self) -> None:
        variant = verify_mod.DEFAULT_VARIANT_ROOT / "world_foam_lane2_fused_slab_v0"
        metal_source = (variant / "csrc" / "metal" / "world_foam_lane2_shared_replay_tensor.metal").read_text(
            encoding="utf-8"
        )

        def kernel_source(name: str) -> str:
            start = metal_source.index(f"kernel void {name}(")
            end = metal_source.find("\nkernel void ", start + 1)
            return metal_source[start : end if end >= 0 else None]

        forward = kernel_source("wf2_fixed_word_p0_lie_node_forward_tensor")
        reverse_helper_start = metal_source.index("static inline void wf2_fixed_word_p0_lie_node_vjp_impl(")
        reverse_helper_end = metal_source.index(
            "\nkernel void wf2_fixed_word_p0_lie_node_vjp_tensor(",
            reverse_helper_start,
        )
        reverse = metal_source[reverse_helper_start:reverse_helper_end] + kernel_source(
            "wf2_fixed_word_p0_lie_node_vjp_tensor"
        )
        for source in (forward, reverse):
            self.assertIn("!isfinite(raw_density) || raw_density < 0.0f", source)
            self.assertIn("const float density = raw_density;", source)
            self.assertNotIn("max(raw_density", source)
            self.assertNotIn("raw_density > 0.0f", source)
        self.assertIn("grad_rgba.w = physical_length * tau_bar;", reverse)

        trainer_source = (verify_mod.DYNAWORLD / "src" / "train" / "powerfoam_metal_trainer.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "densities = F.softplus(self.raw_densities, beta=POWERFOAM_SOFTPLUS_BETA)",
            trainer_source,
        )

    def test_compiled_lie_training_sample_path_has_no_prediction_allocation(self) -> None:
        variant = verify_mod.DEFAULT_VARIANT_ROOT / "world_foam_lane2_fused_slab_v0"
        metal_source = (variant / "csrc" / "metal" / "world_foam_lane2_shared_replay_tensor.metal").read_text(
            encoding="utf-8"
        )
        host_source = (variant / "csrc" / "metal" / "world_foam_lane2_metal.mm").read_text(encoding="utf-8")
        ops_source = (variant / "torch_world_foam_lane2_fused_slab" / "ops.py").read_text(encoding="utf-8")

        helper_start = metal_source.index("inline bool wf2_fixed_word_p0_lie_sample_mse_vjp(")
        rgb_kernel_start = metal_source.index("kernel void wf2_fixed_word_p0_lie_sample_mse_vjp_tensor(")
        loss_kernel_start = metal_source.index(
            "kernel void wf2_fixed_word_p0_lie_sample_mse_vjp_accumulate_only_tensor("
        )
        next_kernel_start = metal_source.index("\nkernel void ", loss_kernel_start + 1)
        helper_source = metal_source[helper_start:rgb_kernel_start]
        rgb_kernel_source = metal_source[rgb_kernel_start:loss_kernel_start]
        loss_kernel_source = metal_source[loss_kernel_start:next_kernel_start]
        self.assertIn("thread float3& prediction", helper_source)
        self.assertIn("wf2_fixed_word_p0_lie_sample_mse_vjp(", rgb_kernel_source)
        self.assertIn("wf2_fixed_word_p0_lie_sample_mse_vjp(", loss_kernel_source)
        self.assertNotIn("prediction_rgb_f32", loss_kernel_source)

        host_start = host_source.index("metal_fixed_word_p0_lie_sample_accumulate_loss_only_launch_only(")
        host_end = host_source.index(
            "\nstd::tuple<torch::Tensor, torch::Tensor, torch::Tensor>\n"
            "metal_fixed_word_p0_lie_world_grad_init_launch_only(",
            host_start,
        )
        loss_host_source = host_source[host_start:host_end]
        self.assertIn(
            "launch(k.fixed_word_p0_lie_sample_mse_vjp_accumulate_only",
            loss_host_source,
        )
        self.assertNotIn("torch::empty", loss_host_source)
        self.assertNotIn("prediction_rgb", loss_host_source)

        ops_start = ops_source.index("def fixed_word_p0_lie_sample_accumulate_loss_only_launch_only(")
        ops_end = ops_source.index("\ndef ", ops_start + 5)
        loss_ops_source = ops_source[ops_start:ops_end]
        self.assertIn(
            "fixed_word_p0_lie_sample_accumulate_loss_only_launch_only(",
            loss_ops_source,
        )
        self.assertNotIn("prediction_rgb", loss_ops_source)

        for adapter_name in (
            "native_track_adapter.py",
            "native_piecewise_topology_adapter.py",
        ):
            adapter_source = (
                verify_mod.DYNAWORLD / "research_experiments" / "world_foam_lane2" / adapter_name
            ).read_text(encoding="utf-8")
            self.assertIn(
                "fixed_word_p0_lie_sample_accumulate_loss_only_launch_only(",
                adapter_source,
            )
            self.assertNotIn(
                "prediction = native.fixed_word_p0_lie_sample_accumulate_launch_only(",
                adapter_source,
            )

    def test_material_reverse_abi_omits_every_geometry_side_buffer_and_finalize(self) -> None:
        variant = verify_mod.DEFAULT_VARIANT_ROOT / "world_foam_lane2_fused_slab_v0"
        metal_source = (variant / "csrc" / "metal" / "world_foam_lane2_shared_replay_tensor.metal").read_text(
            encoding="utf-8"
        )
        host_source = (variant / "csrc" / "metal" / "world_foam_lane2_metal.mm").read_text(encoding="utf-8")
        bindings_source = (variant / "csrc" / "bindings.cpp").read_text(encoding="utf-8")
        ops_source = (variant / "torch_world_foam_lane2_fused_slab" / "ops.py").read_text(encoding="utf-8")
        package_source = (variant / "torch_world_foam_lane2_fused_slab" / "__init__.py").read_text(encoding="utf-8")

        schemas = {
            schema.split("(", 1)[0]: schema
            for schema in re.findall(
                r'm\.def\(\s*"([^"]+)"',
                bindings_source,
                flags=re.DOTALL,
            )
        }
        init_name = "fixed_word_p0_lie_material_world_grad_init_launch_only"
        node_name = "fixed_word_p0_lie_material_node_vjp_accumulate_launch_only"
        for name in (init_name, node_name):
            torch._C.parse_schema(schemas[name])
            self.assertNotIn("grad_mobius", schemas[name])
            self.assertNotIn("grad_boundary", schemas[name])

        init_host_start = host_source.index(f"metal_{init_name}(")
        init_host_end = host_source.index(
            "\nstd::tuple<torch::Tensor, torch::Tensor>\nmetal_fixed_word_p0_lie_node_vjp_accumulate_launch_only(",
            init_host_start,
        )
        init_host = host_source[init_host_start:init_host_end]
        self.assertIn("{site_count, 4}", init_host)
        for omitted in (
            "incidence_count",
            "boundary_count",
            "grad_mobius",
            "grad_boundary",
        ):
            self.assertNotIn(omitted, init_host)

        node_host_start = host_source.index(f"metal_{node_name}(")
        node_host_end = host_source.index(
            "\ntorch::Tensor metal_fixed_word_p0_sparse_mobius_boundary_finalize_launch_only(",
            node_host_start,
        )
        node_host = host_source[node_host_start:node_host_end]
        self.assertIn("launch(k.fixed_word_p0_lie_material_node_vjp", node_host)
        self.assertIn("return grad_site_rgba_f32;", node_host)
        self.assertNotIn("grad_mobius", node_host)
        self.assertNotIn("grad_boundary", node_host)

        kernel_name = "wf2_fixed_word_p0_lie_material_node_vjp_tensor"
        kernel_start = metal_source.index(f"kernel void {kernel_name}(")
        kernel_end = metal_source.index("\nkernel void ", kernel_start + 1)
        material_kernel = metal_source[kernel_start:kernel_end]
        self.assertIn(
            "device atomic_float* grad_site_rgba_f32 [[buffer(11)]]",
            material_kernel,
        )
        self.assertIn("false,\n      gid);", material_kernel)
        self.assertNotIn("grad_mobius_coeff_f32 [[buffer(", material_kernel)
        self.assertNotIn("grad_boundary_f32 [[buffer(", material_kernel)

        def python_function(name: str) -> str:
            start = ops_source.index(f"def {name}(")
            end = ops_source.index("\ndef ", start + 5)
            return ops_source[start:end]

        material_python = "\n".join(
            python_function(name)
            for name in (
                init_name,
                node_name,
                "fixed_word_p0_lie_material_world_grad_finalize_launch_only",
            )
        )
        self.assertIn("FixedWordP0MaterialWorldGradToken(", material_python)
        self.assertNotIn("grad_mobius", material_python)
        self.assertNotIn("grad_boundary", material_python)
        self.assertNotIn("fixed_word_p0_site_geometry_finalize_launch_only(", material_python)
        for exported in (
            "FixedWordP0MaterialWorldGradToken",
            init_name,
            node_name,
            "fixed_word_p0_lie_material_world_grad_finalize_launch_only",
        ):
            self.assertIn(exported, package_source)

        adapter_source = (
            verify_mod.DYNAWORLD / "research_experiments" / "world_foam_lane2" / "native_track_adapter.py"
        ).read_text(encoding="utf-8")
        for required in (
            "native.fixed_word_p0_lie_material_world_grad_init_launch_only",
            "native.fixed_word_p0_lie_material_node_vjp_accumulate_launch_only",
            "native.fixed_word_p0_lie_material_world_grad_finalize_launch_only",
            "geometry_vjp_executed=geometry_vjp_executed",
            "if result.geometry_vjp_executed:",
            "_assert_frozen_geometry_bars_zero(ledger)",
        ):
            self.assertIn(required, adapter_source)

        trainer_source = (
            verify_mod.DYNAWORLD / "research_experiments" / "world_foam_lane2" / "material_training_step.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"geometry_vjp_executed": False', trainer_source)
        self.assertIn(
            '"material_only_reverse_tensor_bytes_omitted": sum(',
            trainer_source,
        )

    def test_sparse_power_boundary_site_vjp_is_native_sparse_and_preserves_compiled_abi(self) -> None:
        variant = verify_mod.DEFAULT_VARIANT_ROOT / "world_foam_lane2_fused_slab_v0"
        metal_source = (variant / "csrc" / "metal" / "world_foam_lane2_shared_replay_tensor.metal").read_text(
            encoding="utf-8"
        )
        derive_kernel_name = "wf2_sparse_power_boundary_from_sites_launch_only_tensor"
        derive_kernel_start = metal_source.index(f"kernel void {derive_kernel_name}(")
        derive_kernel_end = metal_source.find("\nkernel void ", derive_kernel_start + 1)
        derive_kernel_source = metal_source[derive_kernel_start : derive_kernel_end if derive_kernel_end >= 0 else None]
        for required in (
            "device const int* boundary_site_pairs_i32 [[buffer(0)]]",
            "const uint pair_base = boundary_id * 2u;",
            "const float4 normal = 2.0f * (right_q - left_q);",
            "dot(left_q, left_q) - dot(right_q, right_q) -",
            "left_weight + right_weight",
            "boundary_f32[boundary_base + 4u] = bias;",
        ):
            self.assertIn(required, derive_kernel_source)
        kernel_name = "wf2_sparse_power_boundary_site_vjp_launch_only_tensor"
        kernel_start = metal_source.index(f"kernel void {kernel_name}(")
        kernel_end = metal_source.find("\nkernel void ", kernel_start + 1)
        kernel_source = metal_source[kernel_start : kernel_end if kernel_end >= 0 else None]
        for required in (
            "device const int* active_boundary_site_pairs_i32 [[buffer(0)]]",
            "dispatch width equals U",
            "const float3 grad_normal = float3(",
            "-2.0f * grad_normal + 2.0f * grad_bias * left_xyz",
            "2.0f * grad_normal - 2.0f * grad_bias * right_xyz",
            "-grad_bias",
            "grad_bias",
        ):
            self.assertIn(required, kernel_source)
        self.assertEqual(kernel_source.count("wf2_atomic_add5("), 2)
        self.assertNotIn("for (uint boundary_id", kernel_source)

        host_source = (variant / "csrc" / "metal" / "world_foam_lane2_metal.mm").read_text(encoding="utf-8")
        derive_host_start = host_source.index("metal_sparse_power_boundary_from_sites_launch_only(")
        derive_host_end = host_source.index(
            "\ntorch::Tensor metal_fixed_word_p0_sparse_mobius_lower_launch_only(",
            derive_host_start,
        )
        derive_host_function = host_source[derive_host_start:derive_host_end]
        for required in (
            "auto boundary_f32 = torch::empty(",
            "launch(k.sparse_power_boundary_from_sites_launch_only",
            "fn.dispatch((uint64_t)boundary_count, threads);",
        ):
            self.assertIn(required, derive_host_function)
        for forbidden in (".cpu()", "TORCH_CHECK", "check_i32", "check_float"):
            self.assertNotIn(forbidden, derive_host_function)
        host_start = host_source.index("metal_sparse_power_boundary_vjp_to_sites_launch_only(")
        host_end = host_source.index("\nstd::tuple<torch::Tensor, torch::Tensor>\n", host_start)
        host_function = host_source[host_start:host_end]
        for required in (
            "auto grad_sites = torch::zeros_like(sites_f32);",
            "launch(k.sparse_power_boundary_site_vjp_launch_only",
            "fn.dispatch((uint64_t)active_count, threads);",
        ):
            self.assertIn(required, host_function)
        for forbidden in (".cpu()", "TORCH_CHECK", "check_i32", "check_float"):
            self.assertNotIn(forbidden, host_function)

        bindings_source = (variant / "csrc" / "bindings.cpp").read_text(encoding="utf-8")
        self.assertIn(
            '"sparse_power_boundary_from_sites_launch_only(Tensor boundary_site_pairs_i32, '
            'Tensor sites_f32, int boundary_count) -> Tensor"',
            bindings_source,
        )
        self.assertIn(
            '"sparse_power_boundary_vjp_to_sites_launch_only(Tensor active_boundary_site_pairs_i32, '
            'Tensor sites_f32, Tensor grad_boundary_f32) -> Tensor"',
            bindings_source,
        )
        compiled_schema = bindings_source.split(
            '"fixed_word_p0_compiled_lie_transfer_mse_vjp_sparse_mobius_boundary(', 1
        )[1].split('");', 1)[0]
        self.assertIn(
            "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)",
            compiled_schema,
        )

        ops_source = (variant / "torch_world_foam_lane2_fused_slab" / "ops.py").read_text(encoding="utf-8")
        for required in (
            "def recode_sparse_power_boundary_site_pairs(",
            "def sparse_power_boundary_from_sites_reference(",
            "def sparse_power_boundary_vjp_to_sites(",
            "def sparse_power_boundary_vjp_to_sites_launch_only(",
            "def prepare_sparse_power_boundary_vjp_to_sites(",
            "def launch_prepared_sparse_power_boundary_vjp_to_sites(",
            "def fixed_word_p0_compiled_lie_transfer_mse_vjp_sparse_mobius_site_geometry(",
        ):
            self.assertIn(required, ops_source)
        world_start = ops_source.index("def refresh_fixed_word_p0_world_token(")
        world_end = ops_source.index("\ndef prepare_fixed_word_p0_chart_token(", world_start)
        world_source = ops_source[world_start:world_end]
        self.assertLess(
            world_source.index("sparse_power_boundary_from_sites_launch_only("),
            world_source.index("fixed_word_p0_sparse_mobius_lower_launch_only("),
        )
        finalize_start = ops_source.index("def fixed_word_p0_site_geometry_finalize_launch_only(")
        finalize_end = ops_source.index("\ndef ", finalize_start + 5)
        finalize_source = ops_source[finalize_start:finalize_end]
        self.assertIn("world.topology.active_boundary_site_pairs_i32", finalize_source)
        self.assertIn("world.sites_f32", finalize_source)

        convenience_start = ops_source.index(
            "def fixed_word_p0_compiled_lie_transfer_mse_vjp_sparse_mobius_site_geometry("
        )
        convenience_end = ops_source.index("\ndef ", convenience_start + 5)
        convenience_source = ops_source[convenience_start:convenience_end]
        self.assertNotIn("boundary_f32: Tensor", convenience_source)
        for required in (
            "prepare_fixed_word_p0_topology_token(",
            "refresh_fixed_word_p0_world_token(",
            "prepare_fixed_word_p0_chart_token(",
            "prepare_fixed_word_p0_sample_state_token(",
            "prepare_fixed_word_p0_sample_block_token(",
            "fixed_word_p0_lie_node_vjp_accumulate_launch_only(",
            "fixed_word_p0_sparse_mobius_boundary_finalize_launch_only(",
            "fixed_word_p0_site_geometry_finalize_launch_only(world_grad)",
        ):
            self.assertIn(required, convenience_source)
        self.assertNotIn("sparse_power_boundary_vjp_to_sites(", convenience_source)

    def test_sparse_power_boundary_site_recode_and_vjp_match_central_difference(self) -> None:
        variant = verify_mod.DEFAULT_VARIANT_ROOT / "world_foam_lane2_fused_slab_v0"
        sys.path.insert(0, str(variant))
        try:
            from torch_world_foam_lane2_fused_slab.ops import (
                recode_sparse_power_boundary_site_pairs,
                sparse_power_boundary_vjp_to_sites_reference,
            )
        finally:
            sys.path.remove(str(variant))

        boundary_pairs = torch.tensor(
            [[0, 1], [1, 2], [0, 2], [2, 3], [0, 3]],
            dtype=torch.int32,
        )
        # Boundary 3 is repeated across tracks and sites 0/2 participate in
        # several faces, exercising both reductions and shared-site atomics.
        incidence_boundary = torch.tensor([3, 0, 3, 2, 0, 2], dtype=torch.int32)
        active = recode_sparse_power_boundary_site_pairs(
            incidence_boundary,
            boundary_pairs,
            boundary_count=5,
            site_count=4,
        )
        torch.testing.assert_close(
            active,
            torch.tensor([[0, 0, 1], [2, 0, 2], [3, 2, 3]], dtype=torch.int32),
        )
        sites = torch.tensor(
            [
                [-0.3, 0.2, 1.1, 0.05, 0.2],
                [0.7, -0.4, 2.0, 0.25, -0.1],
                [1.2, 0.6, 2.7, 0.65, 0.35],
                [-0.8, 0.1, 3.4, 0.9, -0.25],
            ],
            dtype=torch.float64,
        )
        grad_boundary = torch.tensor(
            [
                [0.11, -0.07, 0.19, -0.13, 0.23],
                [9.0, 8.0, 7.0, 6.0, 5.0],
                [-0.17, 0.29, 0.03, 0.31, -0.37],
                [0.41, -0.43, 0.47, -0.53, 0.59],
                [4.0, 3.0, 2.0, 1.0, -8.0],
            ],
            dtype=torch.float64,
        )
        actual = sparse_power_boundary_vjp_to_sites_reference(active, sites, grad_boundary)

        def objective(site_values: torch.Tensor) -> torch.Tensor:
            table = active.to(dtype=torch.long)
            boundary_ids, left_ids, right_ids = table.unbind(dim=1)
            left = site_values.index_select(0, left_ids)
            right = site_values.index_select(0, right_ids)
            normal = 2.0 * (right[:, :4] - left[:, :4])
            bias = left[:, :4].square().sum(dim=1) - right[:, :4].square().sum(dim=1) - left[:, 4] + right[:, 4]
            boundaries = torch.cat((normal, bias[:, None]), dim=1)
            return (boundaries * grad_boundary.index_select(0, boundary_ids)).sum()

        epsilon = 1.0e-6
        finite_difference = torch.empty_like(sites)
        for site_id in range(sites.shape[0]):
            for component in range(sites.shape[1]):
                delta = torch.zeros_like(sites)
                delta[site_id, component] = epsilon
                finite_difference[site_id, component] = (objective(sites + delta) - objective(sites - delta)) / (
                    2.0 * epsilon
                )
        torch.testing.assert_close(actual, finite_difference, atol=2.0e-9, rtol=2.0e-9)

        duplicate = torch.cat((active, active[:1]), dim=0)
        with self.assertRaisesRegex(ValueError, "strictly increasing and unique"):
            sparse_power_boundary_vjp_to_sites_reference(duplicate, sites, grad_boundary)

    def test_derived_power_sites_through_compiled_lie_reverse_match_end_to_end_central_difference(self) -> None:
        from compiled_lie_world_adjoint import DTYPE, compiled_lie_world_mse_vjp
        from compiled_transfer_adjoint import make_stable_cell_word

        variant = verify_mod.DEFAULT_VARIANT_ROOT / "world_foam_lane2_fused_slab_v0"
        sys.path.insert(0, str(variant))
        try:
            from torch_world_foam_lane2_fused_slab.ops import (
                sparse_power_boundary_from_sites_reference,
                sparse_power_boundary_vjp_to_sites_reference,
            )
        finally:
            sys.path.remove(str(variant))

        boundary_pairs = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32)
        active_pairs = torch.tensor([[0, 0, 1], [1, 1, 2]], dtype=torch.int32)
        sites = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.00],
                [0.0, 0.0, 1.0, 0.0, 0.08],
                [0.0, 0.0, 2.0, 0.0, -0.05],
            ],
            dtype=DTYPE,
        )
        ray_coefficients = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
            dtype=DTYPE,
        )
        words = (make_stable_cell_word([0, 1, 2], [-1, 0, 1], [0, 1, -2]),)
        density = torch.tensor([0.35, 0.62, 0.48], dtype=DTYPE)
        color = torch.tensor(
            [[0.85, 0.12, 0.08], [0.11, 0.72, 0.91], [0.42, 0.81, 0.24]],
            dtype=DTYPE,
        )
        times = torch.linspace(-0.5, 0.5, 5, dtype=DTYPE)
        targets = torch.full((1, times.numel(), 3), 0.27, dtype=DTYPE)

        def run(site_values: torch.Tensor):
            return compiled_lie_world_mse_vjp(
                boundary=sparse_power_boundary_from_sites_reference(boundary_pairs, site_values),
                ray_coefficients=ray_coefficients,
                words=words,
                site_density=density,
                site_color=color,
                times=times,
                targets=targets,
                background=torch.tensor([0.03, 0.04, 0.05], dtype=DTYPE),
                t_min=-0.5,
                t_max=0.5,
                near=0.1,
                far=2.5,
                node_count=2,
                frame_block_size=2,
                track_block_size=1,
                validation_count=0,
            )

        result = run(sites)
        actual = sparse_power_boundary_vjp_to_sites_reference(
            active_pairs,
            sites,
            result.grad_boundary,
        )
        epsilon = 1.0e-6
        finite_difference = torch.empty_like(sites)
        for site_id in range(sites.shape[0]):
            for component in range(sites.shape[1]):
                delta = torch.zeros_like(sites)
                delta[site_id, component] = epsilon
                finite_difference[site_id, component] = (run(sites + delta).loss - run(sites - delta).loss) / (
                    2.0 * epsilon
                )
        torch.testing.assert_close(actual, finite_difference, atol=2.0e-8, rtol=2.0e-6)

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

    def test_kinetic_precompiled_length_node_abi_is_frame_free_and_reverses_lengths(self) -> None:
        variant = verify_mod.DEFAULT_VARIANT_ROOT / "world_foam_lane2_fused_slab_v0"
        metal_source = (variant / "csrc" / "metal" / "world_foam_lane2_shared_replay_tensor.metal").read_text(
            encoding="utf-8"
        )
        host_source = (variant / "csrc" / "metal" / "world_foam_lane2_metal.mm").read_text(encoding="utf-8")
        bindings_source = (variant / "csrc" / "bindings.cpp").read_text(encoding="utf-8")
        ops_source = (variant / "torch_world_foam_lane2_fused_slab" / "ops.py").read_text(encoding="utf-8")

        def kernel_source(name: str) -> str:
            start = metal_source.index(f"kernel void {name}(")
            cursor = metal_source.index("{", start)
            depth = 0
            while cursor < len(metal_source):
                if metal_source[cursor] == "{":
                    depth += 1
                elif metal_source[cursor] == "}":
                    depth -= 1
                    if depth == 0:
                        return metal_source[start : cursor + 1]
                cursor += 1
            raise AssertionError(f"unterminated Metal kernel: {name}")

        forward_name = "kinetic_precompiled_length_p0_lie_node_forward_launch_only"
        forward_into_name = (
            "kinetic_precompiled_length_p0_lie_node_forward_into_launch_only_v1"
        )
        reverse_name = "kinetic_precompiled_length_p0_lie_node_vjp_accumulate_launch_only"
        forward = kernel_source(f"wf2_{forward_name.removesuffix('_launch_only')}_tensor")
        reverse_kernel = kernel_source(f"wf2_{reverse_name.removesuffix('_accumulate_launch_only')}_tensor")
        reverse_helper_start = metal_source.index(
            "static inline void wf2_kinetic_precompiled_length_p0_lie_node_vjp_impl("
        )
        reverse_helper_end = metal_source.index(
            "\nkernel void wf2_kinetic_precompiled_length_p0_lie_node_vjp_tensor(",
            reverse_helper_start,
        )
        reverse = metal_source[reverse_helper_start:reverse_helper_end] + reverse_kernel
        for source in (forward, reverse):
            for forbidden in (
                "frame_count",
                "sample_count",
                "sample_to_node",
                "mobius_coeff",
                "track_ray_coeff",
                "boundary_f32",
                "uint owners[",
                "float lengths[",
                "float trans_before[",
            ):
                self.assertNotIn(forbidden, source)
            for required in (
                "node_physical_length_f32",
                "const uint track_id = gid / node_count;",
                "const uint node_id = gid - track_id * node_count;",
                "const uint chart_base = gid * 4u;",
                "const uint node_length_base = node_id * word_count;",
                "const float segment_alpha = -expm1(-optical_depth);",
                "word_offsets_i32",
                "word_owner_i32",
            ):
                self.assertIn(required, source)
        for required in (
            "total_m += total_beta * segment_alpha * rgb;",
            "total_kappa += optical_depth;",
            "node_physical_length_f32[node_length_base + uint(cursor)]",
            "wf2_lie_inverse_phi_and_derivative(total_kappa, inverse_phi, inverse_phi_prime);",
            "node_chart_f32[chart_base + 0u] = total_kappa;",
        ):
            self.assertIn(required, forward)
        for required in (
            "const float3 bar_m = inverse_phi * grad_chart.yzw;",
            "dot(bar_m, prefix_m + prefix_beta * rgb - total_m) + bar_kappa_word;",
            "prefix_beta * segment_alpha * bar_m",
            "const uint length_index = node_length_base + uint(cursor);",
            "grad_node_physical_length_f32[length_index] = density * tau_bar;",
            "device atomic_float* grad_site_rgba_f32",
            "device float* grad_node_physical_length_f32",
        ):
            self.assertIn(required, reverse)

        schemas = {
            schema.split("(", 1)[0]: schema
            for schema in re.findall(r'm\.def\(\s*"([^"]+)"', bindings_source, flags=re.DOTALL)
        }
        for name in (forward_name, forward_into_name, reverse_name):
            torch._C.parse_schema(schemas[name])
            self.assertIn(f'"{name}"', bindings_source)
            self.assertIn(f"metal_{name}(", host_source)
            self.assertIn(f"def {name}(", ops_source)
        self.assertIn("Tensor(a!) grad_site_rgba_f32", schemas[reverse_name])
        self.assertIn("-> (Tensor(a!), Tensor)", schemas[reverse_name])
        self.assertIn("Tensor(a!) node_chart_out_f32", schemas[forward_into_name])
        self.assertIn("-> Tensor(a!)", schemas[forward_into_name])

        forward_host_start = host_source.index(f"metal_{forward_name}(")
        forward_host_end = host_source.index(
            "\nstd::tuple<torch::Tensor, torch::Tensor, torch::Tensor>\n"
            "metal_fixed_word_p0_lie_sample_state_init_launch_only(",
            forward_host_start,
        )
        reverse_host_start = host_source.index(f"metal_{reverse_name}(")
        reverse_host_end = host_source.index(
            "\ntorch::Tensor metal_fixed_word_p0_lie_material_node_vjp_accumulate_launch_only(",
            reverse_host_start,
        )
        launch_host = (
            host_source[forward_host_start:forward_host_end] + host_source[reverse_host_start:reverse_host_end]
        )
        self.assertNotIn(".cpu()", launch_host)
        self.assertNotIn("frame_count", launch_host)
        self.assertIn("{track_count, node_count, 4}", launch_host)
        self.assertGreaterEqual(
            launch_host.count("fn.dispatch((uint64_t)track_count * (uint64_t)node_count, threads);"),
            2,
        )
        self.assertIn("torch::zeros_like(node_physical_length_f32)", launch_host)

        forward_into_host_start = host_source.index(
            f"metal_{forward_into_name}("
        )
        forward_into_host_end = host_source.index(
            "\nstd::tuple<torch::Tensor, torch::Tensor, torch::Tensor>\n"
            "metal_fixed_word_p0_lie_sample_state_init_launch_only(",
            forward_into_host_start,
        )
        forward_into_host = host_source[
            forward_into_host_start:forward_into_host_end
        ]
        self.assertNotIn("torch::empty", forward_into_host)
        self.assertNotIn("torch::zeros", forward_into_host)
        self.assertIn("fn.setArg(4, node_chart_out_f32);", forward_into_host)
        self.assertIn("return node_chart_out_f32;", forward_into_host)

        for name in (forward_name, forward_into_name, reverse_name):
            start = ops_source.index(f"def {name}(")
            end = ops_source.index("\ndef ", start + 5)
            wrapper = ops_source[start:end]
            self.assertNotIn(".cpu()", wrapper)
            self.assertNotIn(".contiguous()", wrapper)
            self.assertNotIn("frame_count", wrapper)

    def test_kinetic_material_reverse_allocates_no_node_length_bar(self) -> None:
        variant = verify_mod.DEFAULT_VARIANT_ROOT / "world_foam_lane2_fused_slab_v0"
        metal_source = (variant / "csrc" / "metal" / "world_foam_lane2_shared_replay_tensor.metal").read_text(
            encoding="utf-8"
        )
        host_source = (variant / "csrc" / "metal" / "world_foam_lane2_metal.mm").read_text(encoding="utf-8")
        bindings_source = (variant / "csrc" / "bindings.cpp").read_text(encoding="utf-8")
        ops_source = (variant / "torch_world_foam_lane2_fused_slab" / "ops.py").read_text(encoding="utf-8")
        package_source = (variant / "torch_world_foam_lane2_fused_slab" / "__init__.py").read_text(
            encoding="utf-8"
        )
        name = "kinetic_precompiled_length_p0_lie_material_node_vjp_accumulate_launch_only"
        kernel_name = "wf2_kinetic_precompiled_length_p0_lie_material_node_vjp_tensor"
        kernel_start = metal_source.index(f"kernel void {kernel_name}(")
        kernel_end = metal_source.index("\nkernel void ", kernel_start + 1)
        kernel = metal_source[kernel_start:kernel_end]
        self.assertIn("unused_length_bar_f32", kernel)
        self.assertIn("grad_site_rgba_f32,\n      unused_length_bar_f32", kernel)
        self.assertIn("false,\n      gid", kernel)
        self.assertNotIn("grad_node_physical_length_f32[length_index]", kernel)

        host = _braced_definition(host_source, f"metal_{name}(")
        self.assertNotIn("torch::empty", host)
        self.assertNotIn("torch::zeros", host)
        self.assertNotIn("grad_node_physical_length", host)
        self.assertGreaterEqual(host.count("fn.setArg(6, grad_site_rgba_f32);"), 1)
        self.assertGreaterEqual(host.count("fn.setArg(7, grad_site_rgba_f32);"), 1)
        self.assertIn("return grad_site_rgba_f32;", host)

        schemas = {
            schema.split("(", 1)[0]: schema
            for schema in re.findall(r'm\.def\(\s*"([^"]+)"', bindings_source, flags=re.DOTALL)
        }
        torch._C.parse_schema(schemas[name])
        self.assertIn("Tensor(a!) grad_site_rgba_f32", schemas[name])
        self.assertIn("-> Tensor(a!)", schemas[name])
        self.assertIn(f'"{name}"', bindings_source)
        self.assertIn(f"def {name}(", ops_source)
        self.assertIn(name, package_source)


if __name__ == "__main__":
    unittest.main()

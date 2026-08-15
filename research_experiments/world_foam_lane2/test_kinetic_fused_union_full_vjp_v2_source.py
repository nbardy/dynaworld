from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VARIANT = (
    ROOT
    / "third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0"
)
OPS = VARIANT / "torch_world_foam_lane2_fused_slab/ops.py"
BINDINGS = VARIANT / "csrc/bindings.cpp"
HOST = VARIANT / "csrc/metal/world_foam_lane2_metal.mm"
METAL = VARIANT / "csrc/metal/world_foam_lane2_shared_replay_tensor.metal"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _between(text: str, begin: str, end: str) -> str:
    return text.split(begin, 1)[1].split(end, 1)[0]


def test_union_v2_keeps_v1_oracle_and_three_index_identities_distinct() -> None:
    ops = _text(OPS)
    prepared = _between(
        ops,
        "def prepare_kinetic_fused_union_full_vjp_v2(",
        "def _kinetic_fused_union_full_vjp_v2_tensors(",
    )
    assert "prepare_kinetic_fused_direct_full_vjp_v1(" in prepared
    assert "source_site_ids_i64" in prepared
    assert "compact_to_geometry_output_i64" in prepared
    assert "geometry_output_source_site_ids_i64" in prepared
    assert (
        "tuple(output_source_ids[index] for index in compact_to_output) != source_ids"
        in prepared
    )
    assert "must be storage-distinct" in prepared
    assert "transfer_predecessors" in prepared
    assert "transfer_predecessor_release_requires_proven_fence" in prepared
    assert 'geometry_output_index_space: str = "request_union"' in ops
    assert 'factorization_identity: str = "P_b=P_U*Q_b"' in ops


def test_union_v2_metal_validates_factorization_before_union_writes() -> None:
    metal = _text(METAL)
    validate = _between(
        metal,
        "kernel void wf2_kinetic_fused_union_full_vjp_validate_v2_tensor(",
        "kernel void wf2_kinetic_fused_union_full_vjp_finalize_v2_tensor(",
    )
    accumulate = _between(
        metal,
        "kernel void wf2_kinetic_fused_union_full_vjp_v2_tensor(",
        "kernel void wf2_clear_affine_loss_site_rgba_grad_tensor(",
    )
    assert (
        "geometry_output_source_site_ids_i64[uint(union_raw)] != global_raw"
        in validate
    )
    assert "geometry_output_source_site_ids_i64[union_index - 1u] >= global_raw" in validate
    assert "WF2_KINETIC_FUSED_V2_REASON_INDEX_SPACE" in validate
    assert "grad_union_positions0_f32" in accumulate
    assert "grad_union_velocities_f32" in accumulate
    assert "grad_union_weight_coefficients_f32" in accumulate
    assert "left_union * 3u" in accumulate
    assert "right_union * 3u" in accumulate
    assert "left_global * 3u" not in accumulate
    assert "right_global * 3u" not in accumulate
    assert "grad_global_positions0_f32" not in accumulate


def test_union_v2_native_boundary_exposes_only_split_fail_atomic_phases() -> None:
    bindings = _text(BINDINGS)
    host = _text(HOST)
    names = (
        "kinetic_fused_union_full_vjp_validate_shared_status_launch_only_v2",
        "kinetic_fused_union_full_vjp_accumulate_shared_status_launch_only_v2",
        "kinetic_fused_union_full_vjp_finalize_shared_status_launch_only_v2",
    )
    for name in names:
        assert f'"{name}(' in bindings
        assert f'"{name}"' in bindings
        assert f"metal_{name}(" in host
    assert "kinetic_fused_union_full_vjp_accumulate_launch_only_v2(" not in bindings
    core = _between(
        host,
        "metal_kinetic_fused_union_full_vjp_phase_core_v2(",
        "metal_kinetic_fused_union_full_vjp_validate_shared_status_launch_only_v2(",
    )
    validate_launch = core.index("launch(k.kinetic_fused_union_full_vjp_validate_v2")
    accumulate_launch = core.index("launch(k.kinetic_fused_union_full_vjp_v2")
    assert validate_launch < accumulate_launch
    assert "grad_union_positions0_f32.size(0) == union_site_count" in core
    assert 'check_i32_mps_1d(config_i32, "config_i32", 7)' in core


def test_union_v2_python_wrapper_has_no_combined_or_persistent_commit_route() -> None:
    ops = _text(OPS)
    result = _between(
        ops,
        "class KineticFusedUnionFullVjpResultV2:",
        "def _tensor_mutation_signature(",
    )
    launch = _between(
        ops,
        "def kinetic_fused_union_full_vjp_accumulate_launch_only_v2(",
        "def fixed_word_p0_lie_material_node_vjp_accumulate_launch_only(",
    )
    assert 'launch_phase not in {"validate", "accumulate", "finalize"}' in launch
    assert 'launch_phase == "combined"' not in launch
    assert "optimizer" not in launch
    assert "persistent" in launch
    assert "accepted_bars" not in launch
    assert "prepared.union_site_count" in launch
    assert "prepared.global_site_count" in launch
    assert "observed_predecessor_bytes" in launch
    assert (
        "if not self.accumulation_enqueued or not self.finalization_enqueued:"
        in result
    )

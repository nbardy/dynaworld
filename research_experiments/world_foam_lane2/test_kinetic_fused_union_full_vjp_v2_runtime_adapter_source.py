from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ADAPTER = (
    ROOT
    / "research_experiments/world_foam_lane2/kinetic_native_equal_rank_runtime_adapter.py"
)


def _text() -> str:
    return ADAPTER.read_text(encoding="utf-8")


def _between(text: str, begin: str, end: str) -> str:
    return text.split(begin, 1)[1].split(end, 1)[0]


def test_union_v2_adapter_binds_exact_bundle_manifest_and_factorization() -> None:
    source = _text()
    prepare = _between(
        source,
        "def prepare_kinetic_native_equal_rank_fused_union_full_vjp_construction_lifetime_v2(",
        "def _settle_failed_fused_union_construction(",
    )
    assert "PaperKineticUnionLocalSpatialBundle" in prepare
    assert "_certify_union_local_spatial_bundle_cold_current(spatial_bundle)" in prepare
    assert "binding.compact_to_union_i64.tolist()" not in prepare
    assert "spatial_bundle.compact_to_union_by_block" in prepare
    assert source.count("_certify_union_local_spatial_bundle_cold_current(") == 4
    assert "compact_to_union_i64.tolist()" not in source
    assert "active_block_manifest_generation_id != spatial_bundle.generation_digest" in prepare
    assert "prepared blocks are not the exact canonical active bundle order" in prepare
    assert "tuple(spatial_bundle.union_source_site_ids[index] for index in compact_map)" in prepare
    assert "union-v2 block does not prove P_b=P_U Q_b" in prepare
    assert "raw.geometry_output_source_site_ids_i64 is not union_ids" in source
    assert "raw.compact_to_geometry_output_i64 is not binding.compact_to_union_i64" in source
    assert "mapping_tensor_owned_by_preparer" in source
    assert "(False, False)" in source
    assert "required_transaction_bytes = required_output_bar_bytes + 4" in prepare
    assert "raw_prepare(" not in prepare


def test_union_v2_construction_publishes_every_return_before_next_allocation() -> None:
    source = _text()
    materialize = _between(
        source,
        "def materialize_kinetic_native_equal_rank_fused_union_full_vjp_transaction_v2(",
        "def prepare_kinetic_native_equal_rank_fused_union_full_vjp_transaction_v2(",
    )
    assert "lifetime.raw_union_blocks[block_index] = raw" in materialize
    assert "raw_union_blocks=[None] * len(blocks)" in source
    assert "output_tensors=[None] * (len(blocks) + 3)" in source
    assert materialize.index("lifetime.raw_union_blocks[block_index] = raw") < materialize.index(
        "_fused_union_raw_prepared_tensors(raw)"
    )
    assert "lifetime.output_tensors[block_index] = compact_bar" in materialize
    assert "lifetime.output_tensors[len(blocks) + union_bar_index] = union_bar" in materialize
    assert ".append(raw)" not in materialize
    assert ".append(compact_bar)" not in materialize
    assert "_settle_failed_fused_union_construction(" in materialize
    assert 'lifetime.phase = "transferred"' in materialize


def test_union_v2_adapter_has_one_sticky_status_and_global_phase_order() -> None:
    source = _text()
    execute = _between(
        source,
        "def execute_kinetic_native_equal_rank_fused_union_full_vjp_transaction_v2(",
        "def execute_kinetic_native_equal_rank_node_vjp(",
    )
    assert 'for phase in ("validate", "accumulate", "finalize"):' in execute
    assert "validation_status_i32=validation_status" in execute
    assert "validate_shared_union_ledgers" in execute
    assert "finalize_shared_union_ledgers" in execute
    assert "state.compact_ledger_validation_count += 1" in execute
    assert "state.shared_union_ledger_validation_count += int(block_index == 0)" in execute
    assert "state.compact_ledger_finalization_count += 1" in execute
    assert "state.shared_union_ledger_finalization_count += int(block_index == 0)" in execute
    assert execute.count("returned = device_completion_fence()") == 1
    assert "state.completion_fence_call_count += 1" in execute
    assert "reason_mask = int(validation_status.item())" in execute
    assert "result.assert_current()" in execute
    assert execute.index("returned = device_completion_fence()") < execute.index(
        "reason_mask = int(validation_status.item())"
    )


def test_union_v2_adapter_releases_roots_only_after_acceptance() -> None:
    source = _text()
    execute = _between(
        source,
        "def execute_kinetic_native_equal_rank_fused_union_full_vjp_transaction_v2(",
        "def execute_kinetic_native_equal_rank_node_vjp(",
    )
    accepted = execute.index("state.accepted = True")
    assert accepted > execute.index("returned = device_completion_fence()")
    for release in (
        "state.prepared_blocks = ()",
        "state.raw_union_blocks = ()",
        "state.resident_union_source_site_ids_i64 = None",
        "state.spatial_bundle = None",
    ):
        assert execute.index(release) > accepted
    failure = _between(
        source,
        "def _settle_failed_fused_union_transaction(",
        "def execute_kinetic_native_equal_rank_fused_union_full_vjp_transaction_v2(",
    )
    assert "state.quarantined = True" in failure
    assert "_retain_fused_union_rejected_roots(state)" in failure
    assert "_FUSED_VJP_RESTART_REQUIRED_QUARANTINE.append(state)" in failure
    assert "prepared_blocks = ()" not in failure
    assert "raw_union_blocks = ()" not in failure
    assert "_FUSED_UNION_REJECTED_ROOT_QUARANTINE: Any | None = None" in source
    assert "_FUSED_UNION_REJECTED_ROOT_QUARANTINE: list" not in source
    assert "or _FUSED_UNION_REJECTED_ROOT_QUARANTINE is not None" in source
    construction_failure = _between(
        source,
        "def _settle_failed_fused_union_construction(",
        "def materialize_kinetic_native_equal_rank_fused_union_full_vjp_transaction_v2(",
    )
    assert construction_failure.count("returned = fence()") == 1
    assert "lifetime.construction_completion_fence_call_count += 1" in construction_failure
    assert "_retain_fused_union_rejected_roots(quarantine_roots)" in construction_failure
    assert "if after != before:" in construction_failure


def test_union_v2_success_breaks_construction_cycle_after_execution_fence() -> None:
    source = _text()
    execute = _between(
        source,
        "def execute_kinetic_native_equal_rank_fused_union_full_vjp_transaction_v2(",
        "def execute_kinetic_native_equal_rank_node_vjp(",
    )
    release = execute.index("construction_lifetime.transaction = None")
    assert release > execute.index("returned = device_completion_fence()")
    assert release > execute.index("result.assert_current()")
    for statement in (
        "construction_lifetime.raw_union_blocks.clear()",
        "construction_lifetime.output_tensors.clear()",
        "construction_lifetime.prepared_blocks = ()",
        "construction_lifetime.grad_node_chart_f32_by_block = ()",
        "construction_lifetime.spatial_bundle = None",
        "construction_lifetime.compact_to_geometry_output_by_block = ()",
        "construction_lifetime.thresholds_f32_by_block = ()",
        "construction_lifetime.union_abi_identity = ()",
        "construction_lifetime.node_bar_signatures = ()",
        "construction_lifetime.union_identity_signature = ()",
        "construction_lifetime.compact_map_signatures = ()",
        "construction_lifetime.compact_map_generation_digests = ()",
        'construction_lifetime.phase = "released"',
        "state.construction_lifetime = None",
    ):
        assert execute.index(statement) > release
    assert execute.index("state.construction_lifetime = None") > execute.index(
        'construction_lifetime.phase = "released"'
    )


def test_union_v2_transaction_budget_includes_the_sticky_status() -> None:
    source = _text()
    ready = _between(
        source,
        "def _assert_fused_union_transaction_ready(",
        "def prepare_kinetic_native_equal_rank_fused_union_full_vjp_construction_lifetime_v2(",
    )
    assert "transaction.validation_status_tensor_bytes_during_execution != 4" in ready
    assert "!= transaction.output_bar_scratch_tensor_bytes + 4" in ready
    assert "transaction.transaction_scratch_tensor_byte_budget" in ready
    result = _between(
        source,
        "class KineticNativeEqualRankFusedUnionFullVjpV2TransactionResult:",
        "class KineticNativeEqualRankVJPResult:",
    )
    assert "validation_status_tensor_bytes_during_transaction: int" in result
    assert "retained_validation_status_tensor_bytes: int" in result


def test_union_v2_result_is_union_local_single_use_and_cannot_commit() -> None:
    source = _text()
    result = _between(
        source,
        "class KineticNativeEqualRankFusedUnionFullVjpV2TransactionResult:",
        "class KineticNativeEqualRankVJPResult:",
    )
    assert "def consume_bars_once(" in result
    assert "self.assert_current()" in result
    assert "state.consumed = True" in result
    assert "state.union_position_bar = None" in result
    assert 'output_geometry_index_space: str = "request_union"' in result
    assert 'material_output_index_space: str = "block_compact"' in result
    assert "union_material_finiteness_certified: bool = False" in result
    assert "persistent_or_global_write_performed: bool = False" in result
    assert "optimizer_write_performed: bool = False" in result
    assert "index_add_" not in result
    assert "optimizer.step" not in result


def test_union_v2_adapter_explicitly_excludes_bounded_q1() -> None:
    source = _text()
    transaction = _between(
        source,
        "class KineticNativeEqualRankFusedUnionFullVjpV2Transaction:",
        "class _KineticNativeEqualRankFusedUnionFullVjpV2ResultState:",
    )
    ready = _between(
        source,
        "def _assert_fused_union_transaction_ready(",
        "def prepare_kinetic_native_equal_rank_fused_union_full_vjp_construction_lifetime_v2(",
    )
    assert "bounded_batch_q: int | None = None" in transaction
    assert "transaction.bounded_batch_q is not None" in ready
    assert "exact_active_block_manifest_certified: bool = True" in transaction

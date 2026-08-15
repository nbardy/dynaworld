from __future__ import annotations

import pytest
from host_memory_contract import WorldFoamHostMemoryDimensions


def _dimensions(**updates: object) -> WorldFoamHostMemoryDimensions:
    values: dict[str, object] = {
        "global_track_count": 32,
        "global_sample_count": 300,
        "global_site_count": 11,
        "global_word_count": 70,
        "global_incidence_count": 50,
        "block_track_count": 8,
        "sample_block_size": 4,
        "block_site_count": 7,
        "block_boundary_count": 6,
        "block_word_count": 18,
        "block_incidence_count": 12,
        "chart_node_counts": (4, 8),
        "plan_pixel_count": 32,
        "plan_sample_count": 300,
        "image_height": 128,
        "image_width": 192,
        "compiled_cpu_artifact_store_max_resident_accounted_bytes": 20_000_000,
        "spatial_block_count": 4,
    }
    values.update(updates)
    return WorldFoamHostMemoryDimensions(**values)


def test_exact_tensor_payload_formulas_match_current_layouts() -> None:
    dims = _dimensions()

    assert dims.continuous_certificate_parameter_count == 5 * 6 + 12 * 8 + 4 * 12 + 4 * 7
    assert dims.global_kinetic_geometry_parameter_tensor_bytes == 8 * 11 * (6 + 3)
    assert dims.global_source_parameter_tensor_bytes == 8 * 11 * (8 + 3)
    assert dims.global_material_step_accumulator_tensor_bytes == 16 * 11 + 4
    assert dims.global_fixed_camera_geometry_gradient_tensor_bytes == (
        16 * 11 + 4 + 8 * 11 * (6 + 3)
    )
    assert dims.global_gradient_tensor_bytes == 16 * 11 + 4 + 8 * 11 * (6 + 3)
    assert dims.global_decoded_material_and_raw_chain_gradient_tensor_bytes == 48 * 11
    assert dims.global_material_training_state_plus_geometry_tensor_bytes == 120 * 11
    assert dims.global_material_training_step_peak_base_tensor_bytes == 136 * 11 + 4
    assert dims.global_template_tensor_bytes == (
        64 * 32 * (4 + 8) + 8 * ((4 + 4 * 4) + (8 + 8 * 8)) + 48 * 2 * 50 + 24 * 70
    )
    assert dims.compact_topology_tensor_bytes == (16 * 8 + 8 + 12 * 18 + 4 * 12 + 16 * 6 + 8 * 7)
    assert dims.compact_world_and_atlas_tensor_bytes == (
        72 * 7 + 96 * 8 + 40 * 6 + 64 * 8 * (4 + 8) + 8 * ((4 + 4 * 4) + (8 + 8 * 8)) + 48 * 2 * 12 + 24 * 18
    )
    assert dims.staged_target_and_ray_tensor_bytes == 36 * 8 * 4
    assert dims.staged_target_tensor_bytes == 12 * 8 * 4
    assert dims.native_node_state_and_bar_tensor_bytes == 32 * 8 * 8
    assert dims.native_geometry_length_bar_tensor_bytes == 4 * 8 * 18
    assert dims.native_material_geometry_length_bar_tensor_bytes == 0
    assert dims.native_sample_payload_peak_tensor_bytes == 36 * 8 * 4
    assert dims.native_material_sample_payload_peak_tensor_bytes == 12 * 8 * 4
    assert dims.native_optional_prediction_tensor_bytes == 12 * 8 * 4
    assert dims.maximum_materialized_sample_count == 8 * 4
    assert dims.native_sample_weight_tensor_bytes == 4 * 8 * 4 * 8
    assert dims.native_sample_identity_tensor_bytes == 12 * 8 * 4
    assert dims.native_materialized_sample_block_tensor_bytes == (
        4 * 8 * 4 * 8 + 24 * 8 * 4
    )
    assert dims.native_prepared_sample_public_scratch_tensor_bytes == 4 * 8 * 4 + 20
    assert dims.native_synchronized_material_sample_launch_tensor_bytes == (
        4 * 8 * 4 * 8 + 28 * 8 * 4 + 20
    )
    assert dims.native_sample_time_tensor_bytes == 8 * 4
    assert dims.native_compact_live_world_and_bar_tensor_bytes == 72 * 7 + 48 * 8
    assert dims.native_topology_cache_tensor_bytes_per_block == (dims.compact_topology_tensor_bytes - 8 * 6)
    assert dims.native_topology_cache_tensor_bytes == dims.compact_topology_tensor_bytes - 8 * 6
    assert dims.training_binding_tensor_bytes_per_block == 1908
    assert dims.unique_schedule_tensor_bytes == 832
    assert dims.native_adapter_sample_peak_dominant_tensor_bytes == 7224
    assert dims.native_adapter_finalize_peak_dominant_tensor_bytes == 2192
    assert dims.native_audited_dominant_peak_tensor_payload_lower_bound_bytes == 7224
    assert dims.native_material_adapter_sample_peak_dominant_tensor_bytes == 6144
    assert dims.native_material_adapter_finalize_peak_dominant_tensor_bytes == 1740
    assert dims.native_material_audited_dominant_peak_tensor_payload_lower_bound_bytes == 6144
    assert dims.staging_plan_tensor_bytes == 8 * 32 + 12 * 300
    assert dims.ordered_plan_peak_additional_tensor_bytes == 20 * 300
    assert dims.decoded_rgb_frame_tensor_bytes == 12 * 128 * 192
    assert dims.cpu_piecewise_reference_target_bytes == 24 * 32 * 300


def test_expensive_replay_state_is_frame_independent_but_current_metadata_is_linear() -> None:
    short = _dimensions(global_sample_count=300)
    long = short.with_global_sample_count(30_000)

    assert long.expensive_block_tensor_bytes == short.expensive_block_tensor_bytes
    assert long.staged_target_and_ray_tensor_bytes == short.staged_target_and_ray_tensor_bytes
    assert (
        long.native_audited_dominant_peak_tensor_payload_lower_bound_bytes
        == short.native_audited_dominant_peak_tensor_payload_lower_bound_bytes
    )
    assert (
        long.native_material_audited_dominant_peak_tensor_payload_lower_bound_bytes
        == short.native_material_audited_dominant_peak_tensor_payload_lower_bound_bytes
    )
    fixed_metadata = 8 * short.plan_pixel_count
    assert (
        long.cheap_temporal_tensor_bytes - fixed_metadata == (short.cheap_temporal_tensor_bytes - fixed_metadata) * 100
    )
    assert long.cpu_piecewise_reference_target_bytes == short.cpu_piecewise_reference_target_bytes * 100


def test_camera_calibration_bars_are_explicit_and_fixed_camera_is_the_default() -> None:
    fixed = _dimensions(global_track_count=32)
    calibrated = _dimensions(
        global_track_count=32,
        trainable_camera_ray_count=32,
    )

    assert fixed.trainable_camera_ray_count == 0
    assert fixed.global_gradient_tensor_bytes == (
        fixed.global_fixed_camera_geometry_gradient_tensor_bytes
    )
    assert calibrated.global_source_parameter_tensor_bytes == (
        fixed.global_source_parameter_tensor_bytes
    )
    assert calibrated.global_gradient_tensor_bytes == (
        fixed.global_gradient_tensor_bytes + 96 * 32
    )


def test_kinetic_weight_rank_changes_only_site_geometry_storage() -> None:
    constant = _dimensions(weight_coefficient_count=1)
    quadratic = _dimensions(weight_coefficient_count=3)

    assert quadratic.global_source_parameter_tensor_bytes - constant.global_source_parameter_tensor_bytes == (
        8 * quadratic.global_site_count * 2
    )
    assert quadratic.global_gradient_tensor_bytes - constant.global_gradient_tensor_bytes == (
        8 * quadratic.global_site_count * 2
    )


def test_current_global_template_still_scales_with_global_tracks() -> None:
    small = _dimensions(global_track_count=32)
    large = _dimensions(global_track_count=320)

    assert large.expensive_block_tensor_bytes == small.expensive_block_tensor_bytes
    assert large.global_template_tensor_bytes > small.global_template_tensor_bytes
    table = {row.name: row for row in large.allocation_table()}
    assert table["global_cpu_atlas_template"].status == "unbounded"


def test_strict_certificate_dense_ad_is_not_a_publication_scale_memory_path() -> None:
    dims = _dimensions(
        global_track_count=8192,
        plan_pixel_count=8192,
        block_track_count=8192,
        block_site_count=1,
        block_boundary_count=0,
        block_word_count=8192,
        block_incidence_count=0,
        chart_node_counts=(16,),
    )

    # Even the impossible best case with no active faces and one site has
    # D >= 12*B_p. Pointer slots alone exceed 768 GiB; Fraction/Interval/Dual
    # objects and arithmetic temporaries are deliberately not counted.
    assert dims.continuous_certificate_parameter_count == 98_308
    assert dims.strict_certificate_dense_pointer_lower_bound_bytes > 768 * 2**30


def test_full_video_residency_is_explicit_not_hidden_in_the_bounded_claim() -> None:
    streamed = _dimensions(source_target_resident_bytes=0)
    resident = _dimensions(source_target_resident_bytes=7_500_000_000)
    streamed_row = {row.name: row for row in streamed.allocation_table()}["target_source_residency"]
    resident_row = {row.name: row for row in resident.allocation_table()}["target_source_residency"]

    assert streamed_row.status == "optional"
    assert resident_row.status == "unbounded"
    assert resident_row.bytes == 7_500_000_000


def test_native_runtime_table_excludes_dense_oracles_and_full_target_tensors() -> None:
    dims = _dimensions(source_target_resident_bytes=0)
    table = {row.name: row for row in dims.native_runtime_allocation_table()}

    assert "global_cpu_atlas_template" not in table
    assert "continuous_certificate_dense_pointer_floor" not in table
    assert "cpu_piecewise_reference_targets" not in table
    assert all(row.status != "unbounded" for row in table.values())
    assert table["global_kinetic_geometry_parameters"].bytes == 8 * 11 * (6 + 3)
    assert table["global_material_step_accumulator"].bytes == 16 * 11 + 4
    assert table["native_node_state_and_bar"].bytes == 32 * 8 * 8
    assert table["native_synchronized_material_sample_launch"].bytes == (
        4 * 8 * 4 * 8 + 28 * 8 * 4 + 20
    )
    assert table["native_sample_times"].bytes == 8 * 4
    assert table["native_material_geometry_length_bar"].bytes == 0
    assert table["sample_identity_and_time_metadata"].status == "cheap_linear"
    assert table["native_material_audited_dominant_peak_tensor_payload_lower_bound"].status == ("partial_lower_bound")
    assert table["native_cached_material_topology_tokens"].bytes == (
        dims.compact_topology_tensor_bytes - 8 * dims.block_boundary_count
    )
    assert table["bounded_compiled_cpu_artifact_store"].bytes == 20_000_000


def test_legacy_all_block_binding_residency_is_not_a_bounded_runtime_row() -> None:
    dims = _dimensions(spatial_block_count=4)
    table = {row.name: row for row in dims.allocation_table()}
    legacy = table[
        "legacy_compact_material_training_binding_private_tensors_per_block"
    ]

    assert legacy.bytes == dims.training_binding_tensor_bytes_per_block
    assert legacy.status == "unbounded"
    assert "multiplies by spatial block count" in legacy.dependence


def test_native_topology_cache_residency_is_independent_of_total_spatial_blocks() -> None:
    small = _dimensions(spatial_block_count=4, native_topology_cache_max_entries=1)
    large = _dimensions(spatial_block_count=400, native_topology_cache_max_entries=1)

    assert large.native_topology_cache_tensor_bytes == small.native_topology_cache_tensor_bytes
    assert large.native_topology_cache_tensor_bytes == small.native_topology_cache_tensor_bytes_per_block


def test_rectangular_multiview_staging_counts_original_view_time_metadata() -> None:
    single = _dimensions(global_sample_count=300, plan_sample_count=300)
    three_views = _dimensions(
        global_track_count=3 * 32,
        global_sample_count=300,
        plan_sample_count=900,
    )

    assert three_views.staging_plan_tensor_bytes - single.staging_plan_tensor_bytes == 12 * 600
    assert (
        three_views.native_audited_dominant_peak_tensor_payload_lower_bound_bytes
        == single.native_audited_dominant_peak_tensor_payload_lower_bound_bytes
    )


def test_native_runtime_table_fails_the_memory_light_contract_for_resident_video() -> None:
    dims = _dimensions(source_target_resident_bytes=1)
    table = {row.name: row for row in dims.native_runtime_allocation_table()}

    assert table["target_source_residency"].status == "unbounded"


@pytest.mark.parametrize("field", ["global_sample_count", "sample_block_size", "image_height"])
def test_positive_dimensions_fail_closed(field: str) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _dimensions(**{field: 0})


def test_direct_kinetic_and_camera_bar_dimensions_fail_closed() -> None:
    with pytest.raises(ValueError, match="degree-2 ABI"):
        _dimensions(weight_coefficient_count=4)
    with pytest.raises(ValueError, match="cannot exceed global .* track count"):
        _dimensions(global_track_count=8, trainable_camera_ray_count=9)

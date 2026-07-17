from __future__ import annotations

import unittest

import verify_native_packed_extension as native_verify


class VerifyNativePackedExtensionTests(unittest.TestCase):
    def test_native_sorted_packed_extension_fixture_passes_when_built(self) -> None:
        if not any(native_verify.VARIANT_ROOT.glob("torch_world_foam_lane2_fused_slab/_C*.so")):
            self.skipTest("world_foam_lane2_fused_slab_v0 extension has not been built")

        result = native_verify.verify()

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["base_offsets_i32"], [0, 2])
        self.assertEqual(result["base_offsets_i32_dtype"], "int32")
        self.assertEqual(result["base_record_i32"], [2097152, 1049089])
        self.assertEqual(result["base_record_i32_dtype"], "int32")
        self.assertEqual(result["change_record_i32"], [])
        self.assertEqual(result["change_record_i32_dtype"], "int32")
        self.assertEqual(result["track_change_offsets_i32"], [0, 0])
        self.assertEqual(result["track_change_offsets_i32_dtype"], "int32")
        self.assertEqual(result["cut_array_cut_ids_i64"], [-1, 0, -2, -1, 0, -2])
        self.assertEqual(result["cut_array_cut_ids_i64_dtype"], "int64")
        self.assertEqual(result["cut_array_cut_offsets_i64"], [0, 3, 6])
        self.assertEqual(result["cut_array_start_segments_i64"], [0, 0])
        self.assertEqual(result["cut_array_initial_owner_i64"], [0, 0])
        self.assertEqual(result["cut_base_offsets_i32"], [0, 2])
        self.assertEqual(result["cut_base_offsets_i32_dtype"], "int32")
        self.assertEqual(result["cut_base_record_i32"], [2097152, 1049089])
        self.assertEqual(result["cut_base_record_i32_dtype"], "int32")
        self.assertEqual(result["cut_change_record_i32"], [])
        self.assertEqual(result["cut_change_record_i32_dtype"], "int32")
        self.assertEqual(result["cut_track_change_offsets_i32"], [0, 0])
        self.assertEqual(result["cut_track_change_offsets_i32_dtype"], "int32")
        self.assertEqual(result["changing_sorted_track_change_offsets_i32"], [0, 1])
        self.assertEqual(result["changing_sorted_change_frame_i32"], [1])
        self.assertEqual(result["changing_sorted_change_offsets_i32"], [0, 2])
        self.assertEqual(result["changing_sorted_change_record_i32"], [2097153, 1049088])
        self.assertEqual(result["changing_sorted_change_record_i32_dtype"], "int32")
        self.assertEqual(result["direct_csr_base_offsets_i32"], [0, 2])
        self.assertEqual(result["direct_csr_base_offsets_i32_dtype"], "int32")
        self.assertEqual(result["direct_csr_base_record_i32"], [2097152, 1049089])
        self.assertEqual(result["direct_csr_base_record_i32_dtype"], "int32")
        self.assertEqual(result["direct_csr_change_record_i32"], [])
        self.assertEqual(result["direct_csr_change_record_i32_dtype"], "int32")
        self.assertEqual(result["direct_csr_track_change_offsets_i32"], [0, 0])
        self.assertEqual(result["direct_csr_track_change_offsets_i32_dtype"], "int32")
        self.assertEqual(result["changing_direct_csr_change_frame_i32"], [1])
        self.assertEqual(result["changing_direct_csr_change_offsets_i32"], [0, 2])
        self.assertEqual(result["changing_direct_csr_change_record_i32"], [2097153, 1049088])
        self.assertEqual(result["changing_direct_csr_change_record_i32_dtype"], "int32")
        self.assertEqual(result["changing_cut_track_change_offsets_i32"], [0, 1])
        self.assertEqual(result["changing_cut_change_frame_i32"], [1])
        self.assertEqual(result["changing_cut_change_offsets_i32"], [0, 2])
        self.assertEqual(result["changing_cut_change_record_i32"], [2097153, 1049088])
        self.assertEqual(result["changing_cut_change_record_i32_dtype"], "int32")
        self.assertTrue(result["has_launch_only_packed_framegroup16_op"])
        self.assertTrue(result["has_launch_only_packed_framegroup16_unchecked_op"])
        self.assertTrue(result["has_launch_only_packed_framegroup16_reduce32_op"])
        self.assertTrue(result["has_launch_only_packed_framegroup16_reduce32_unchecked_op"])
        self.assertTrue(result["has_launch_only_packed_framegroup16_rowselect32_op"])
        self.assertTrue(result["has_launch_only_packed_framegroup16_rowselect32_unchecked_op"])
        self.assertTrue(result["has_launch_only_packed_framegroup16_rowdesc_op"])
        self.assertTrue(result["has_launch_only_packed_framegroup16_rowdesc_unchecked_op"])
        self.assertTrue(result["has_launch_only_packed_framegroup16_rowdesc32_op"])
        self.assertTrue(result["has_launch_only_packed_framegroup16_rowdesc32_unchecked_op"])
        self.assertTrue(result["has_launch_only_packed_framegroup16_recompute_op"])
        self.assertTrue(result["has_launch_only_packed_framegroup16_smallrun16_op"])
        self.assertTrue(result["has_launch_only_packed_framegroup16_materialized_op"])
        self.assertTrue(result["has_affine_candidate_num32_den16_fused_mse_op"])
        self.assertTrue(result["has_affine_candidate_num32_den16_track_fused_mse_op"])
        self.assertTrue(result["pack_endpoint_records_i32_rejects_rank2"])
        self.assertIn("rank-1", result["pack_endpoint_records_i32_rejects_rank2_message"])
        self.assertTrue(result["pack_endpoint_records_i32_rejects_owner_out_of_range"])
        self.assertIn("owner ids", result["pack_endpoint_records_i32_rejects_owner_out_of_range_message"])
        self.assertTrue(result["pack_endpoint_records_i32_rejects_cut_out_of_range"])
        self.assertIn("cut codes", result["pack_endpoint_records_i32_rejects_cut_out_of_range_message"])
        self.assertTrue(result["gate4_delta_replace_from_cuts_rejects_start_segment_oob"])
        self.assertIn(
            "start_segment out of bounds",
            result["gate4_delta_replace_from_cuts_rejects_start_segment_oob_message"],
        )
        self.assertTrue(result["gate4_delta_replace_packed_from_cuts_rejects_start_segment_oob"])
        self.assertIn(
            "start_segment out of bounds",
            result["gate4_delta_replace_packed_from_cuts_rejects_start_segment_oob_message"],
        )
        self.assertTrue(result["gate4_delta_replace_from_cuts_rejects_active_mismatch"])
        self.assertIn(
            "both active or both inactive",
            result["gate4_delta_replace_from_cuts_rejects_active_mismatch_message"],
        )
        self.assertTrue(result["gate4_delta_replace_packed_from_cuts_rejects_active_mismatch"])
        self.assertIn(
            "both active or both inactive",
            result["gate4_delta_replace_packed_from_cuts_rejects_active_mismatch_message"],
        )
        self.assertTrue(result["gate4_delta_replace_from_cuts_rejects_boundary_other_oob"])
        self.assertIn(
            "boundary_other_by_owner_i64 values must be -1 or valid site ids",
            result["gate4_delta_replace_from_cuts_rejects_boundary_other_oob_message"],
        )
        self.assertTrue(result["gate4_delta_replace_packed_from_cuts_rejects_boundary_other_oob"])
        self.assertIn(
            "boundary_other_by_owner_i64 values must be -1 or valid site ids",
            result["gate4_delta_replace_packed_from_cuts_rejects_boundary_other_oob_message"],
        )
        for prefix, message in (
            ("nan_depth", "cut_depths_f64 values must be finite"),
            ("decreasing_depth", "cut_depths_f64 values must be nondecreasing within each row"),
            ("bad_first_sentinel", "cut row first id must be -1 near sentinel"),
            ("bad_last_sentinel", "cut row last id must be -2 far sentinel"),
            ("internal_boundary_id_oob", "cut row internal ids must be valid boundary ids"),
            ("single_cut_row", "cut row with cuts requires at least near/far sentinels"),
        ):
            unpacked_key = f"gate4_delta_replace_from_cuts_rejects_{prefix}"
            packed_key = f"gate4_delta_replace_packed_from_cuts_rejects_{prefix}"
            self.assertTrue(result[unpacked_key])
            self.assertIn(message, result[f"{unpacked_key}_message"])
            self.assertTrue(result[packed_key])
            self.assertIn(message, result[f"{packed_key}_message"])
        for prefix, message in (
            ("row_active_bad_value", "row_active_i64 values must be 0 or 1"),
            ("valid_count_oob", "valid_counts_i64 value out of bounds"),
            ("negative_boundary_id", "sorted_ids_i64 boundary id out of bounds"),
            ("boundary_id_oob", "sorted_ids_i64 boundary id out of bounds"),
            ("boundary_other_oob", "boundary_other_by_owner_i64 values must be -1 or valid site ids"),
            ("nan_depth", "sorted_depths_f64 values must be finite"),
            ("below_near_depth", "sorted_depths_f64 values must be within [near, far]"),
            ("above_far_depth", "sorted_depths_f64 values must be within [near, far]"),
            ("decreasing_depth", "sorted_depths_f64 valid depths must be nondecreasing"),
        ):
            unpacked_key = f"gate4_delta_replace_from_sorted_rejects_{prefix}"
            packed_key = f"gate4_delta_replace_packed_from_sorted_rejects_{prefix}"
            self.assertTrue(result[unpacked_key])
            self.assertIn(message, result[f"{unpacked_key}_message"])
            self.assertTrue(result[packed_key])
            self.assertIn(message, result[f"{packed_key}_message"])
        for prefix, message in (
            ("row_active_bad_value", "row_active_i64 values must be 0 or 1"),
            ("valid_count_oob", "valid_counts_i64 value out of bounds"),
            ("negative_boundary_id", "sorted_ids_i64 values must be nonnegative boundary ids"),
            ("nan_depth", "sorted_depths_f64 values must be finite"),
            ("below_near_depth", "sorted_depths_f64 values must be within [near, far]"),
            ("above_far_depth", "sorted_depths_f64 values must be within [near, far]"),
            ("decreasing_depth", "sorted_depths_f64 valid depths must be nondecreasing"),
        ):
            key = f"gate4_cut_arrays_from_sorted_rejects_{prefix}"
            self.assertTrue(result[key])
            self.assertIn(message, result[f"{key}_message"])
        for key in (
            "base_offsets_i32",
            "base_record_i32",
            "change_record_i32",
            "track_change_offsets_i32",
            "cut_array_cut_ids_i64",
            "cut_array_cut_offsets_i64",
            "cut_array_start_segments_i64",
            "cut_array_initial_owner_i64",
            "cut_base_offsets_i32",
            "cut_base_record_i32",
            "cut_change_record_i32",
            "cut_track_change_offsets_i32",
            "changing_sorted_change_frame_i32",
            "changing_sorted_change_offsets_i32",
            "changing_sorted_track_change_offsets_i32",
            "changing_sorted_change_record_i32",
            "direct_csr_base_offsets_i32",
            "direct_csr_base_record_i32",
            "direct_csr_change_record_i32",
            "direct_csr_track_change_offsets_i32",
            "changing_direct_csr_change_frame_i32",
            "changing_direct_csr_change_offsets_i32",
            "changing_direct_csr_track_change_offsets_i32",
            "changing_direct_csr_change_record_i32",
            "changing_cut_change_frame_i32",
            "changing_cut_change_offsets_i32",
            "changing_cut_track_change_offsets_i32",
            "changing_cut_change_record_i32",
        ):
            self.assertEqual(result[f"{key}_device"], "cpu")
            self.assertEqual(result[f"{key}_shape"], [len(result[key])])
            self.assertTrue(result[f"{key}_contiguous"])


if __name__ == "__main__":
    unittest.main()

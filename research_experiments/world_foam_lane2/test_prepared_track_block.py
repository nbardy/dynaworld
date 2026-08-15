from __future__ import annotations

import unittest

import torch
from compiled_transfer_adjoint import direct_word_render, make_stable_cell_word
from prepared_track_block import (
    accumulate_prepared_rows_,
    gather_prepared_rows,
    prepare_worldfoam_track_block,
    scatter_prepared_rows,
)


class PreparedWorldFoamTrackBlockTest(unittest.TestCase):
    def setUp(self) -> None:
        self.words = (
            make_stable_cell_word([0, 1], [-1, 0], [0, -2]),
            make_stable_cell_word([3, 1, 2], [-1, 2, 0], [2, 0, -2]),
            make_stable_cell_word([2, 4], [-1, 2], [2, -2]),
        )
        self.boundary_site_pairs = torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 4]],
            dtype=torch.int64,
        )

    def test_compacts_tracks_boundaries_sites_and_row_local_incidence(self) -> None:
        block = prepare_worldfoam_track_block(
            self.words,
            self.boundary_site_pairs,
            site_count=5,
            track_start=1,
            track_end=3,
        )
        torch.testing.assert_close(block.source_track_ids, torch.tensor([1, 2]))
        torch.testing.assert_close(block.source_boundary_ids, torch.tensor([0, 2]))
        torch.testing.assert_close(block.source_site_ids, torch.tensor([0, 1, 2, 3, 4]))
        torch.testing.assert_close(block.word_offsets_i32, torch.tensor([0, 3, 5], dtype=torch.int32))
        torch.testing.assert_close(block.word_owner_i32, torch.tensor([3, 1, 2, 2, 4], dtype=torch.int32))
        torch.testing.assert_close(
            block.track_incidence_offsets_i32,
            torch.tensor([0, 2, 3], dtype=torch.int32),
        )
        torch.testing.assert_close(
            block.incidence_boundary_i32,
            torch.tensor([0, 1, 1], dtype=torch.int32),
        )
        torch.testing.assert_close(
            block.word_left_incidence_i32,
            torch.tensor([-1, 1, 0, -1, 0], dtype=torch.int32),
        )
        torch.testing.assert_close(
            block.word_right_incidence_i32,
            torch.tensor([1, 0, -2, 0, -2], dtype=torch.int32),
        )
        torch.testing.assert_close(
            block.boundary_site_pairs_i32,
            torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        )
        self.assertGreater(block.resident_bytes, 0)

    def test_repeated_global_boundary_gets_one_incidence_per_track(self) -> None:
        repeated = (
            make_stable_cell_word(
                [0, 1, 2, 0],
                [-1, 0, 1, 0],
                [0, 1, 0, -2],
            ),
        )
        block = prepare_worldfoam_track_block(
            repeated,
            self.boundary_site_pairs,
            site_count=5,
            track_start=0,
            track_end=1,
        )
        self.assertEqual(block.boundary_count, 2)
        self.assertEqual(block.incidence_count, 2)
        torch.testing.assert_close(
            block.word_left_incidence_i32,
            torch.tensor([-1, 0, 1, 0], dtype=torch.int32),
        )
        torch.testing.assert_close(
            block.word_right_incidence_i32,
            torch.tensor([0, 1, 0, -2], dtype=torch.int32),
        )

    def test_single_segment_block_has_no_boundary_or_incidence_rows(self) -> None:
        block = prepare_worldfoam_track_block(
            (make_stable_cell_word([4], [-1], [-2]),),
            self.boundary_site_pairs,
            site_count=5,
            track_start=0,
            track_end=1,
        )
        self.assertEqual(block.boundary_count, 0)
        self.assertEqual(block.incidence_count, 0)
        self.assertEqual(tuple(block.boundary_site_pairs_i32.shape), (0, 2))
        torch.testing.assert_close(block.source_site_ids, torch.tensor([4]))

    def test_gather_and_scatter_preserve_global_row_identity(self) -> None:
        block = prepare_worldfoam_track_block(
            self.words,
            self.boundary_site_pairs,
            site_count=5,
            track_start=1,
            track_end=2,
        )
        values = torch.arange(15, dtype=torch.float64).reshape(5, 3)
        compact = gather_prepared_rows(values, block.source_site_ids)
        scattered = scatter_prepared_rows(
            compact,
            block.source_site_ids,
            output_rows=5,
        )
        expected = torch.zeros_like(values)
        expected[block.source_site_ids] = values[block.source_site_ids]
        torch.testing.assert_close(scattered, expected)

    def test_caller_owned_scatter_sums_shared_rows_without_reallocation(self) -> None:
        output = torch.zeros((5, 2), dtype=torch.float64)
        storage_pointer = output.untyped_storage().data_ptr()
        first = accumulate_prepared_rows_(
            output,
            torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64),
            torch.tensor([1, 3]),
        )
        second = accumulate_prepared_rows_(
            output,
            torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float64),
            torch.tensor([3, 4]),
        )
        self.assertIs(first, output)
        self.assertIs(second, output)
        self.assertEqual(output.untyped_storage().data_ptr(), storage_pointer)
        torch.testing.assert_close(
            output,
            torch.tensor(
                [
                    [0.0, 0.0],
                    [1.0, 2.0],
                    [0.0, 0.0],
                    [8.0, 10.0],
                    [7.0, 8.0],
                ],
                dtype=torch.float64,
            ),
        )
        with self.assertRaisesRegex(ValueError, "trailing dimensions"):
            accumulate_prepared_rows_(output, torch.ones((1, 3)), torch.tensor([0]))
        with self.assertRaisesRegex(ValueError, "dtype and device"):
            accumulate_prepared_rows_(output, torch.ones((1, 2)), torch.tensor([0]))

    def test_compact_csr_reconstructs_identical_ordered_transfer(self) -> None:
        block = prepare_worldfoam_track_block(
            self.words,
            self.boundary_site_pairs,
            site_count=5,
            track_start=1,
            track_end=3,
        )
        boundary = torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0, -1.3],
                [0.0, 0.0, 1.0, 0.0, -1.0],
                [0.0, 0.0, 1.0, 0.0, -0.7],
                [0.0, 0.0, 1.0, 0.0, -1.6],
            ],
            dtype=torch.float64,
        )
        rays = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        )
        density = torch.tensor([0.3, 0.5, 0.7, 0.9, 1.1], dtype=torch.float64)
        color = torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [0.3, 0.2, 0.1],
                [0.8, 0.1, 0.2],
                [0.1, 0.7, 0.3],
                [0.2, 0.3, 0.9],
            ],
            dtype=torch.float64,
        )
        compact_words = []
        for track_id in range(block.track_count):
            start = int(block.word_offsets_i32[track_id].item())
            end = int(block.word_offsets_i32[track_id + 1].item())
            incidence_start = int(block.track_incidence_offsets_i32[track_id].item())

            def compact_cut(cut_id: int, row_start: int = incidence_start) -> int:
                return cut_id if cut_id < 0 else int(block.incidence_boundary_i32[row_start + cut_id].item())

            compact_words.append(
                make_stable_cell_word(
                    block.word_owner_i32[start:end],
                    [compact_cut(int(value)) for value in block.word_left_incidence_i32[start:end]],
                    [compact_cut(int(value)) for value in block.word_right_incidence_i32[start:end]],
                )
            )
        common = {
            "times": torch.linspace(-0.5, 0.5, 7, dtype=torch.float64),
            "background": torch.tensor([0.02, 0.03, 0.04], dtype=torch.float64),
            "near": 0.1,
            "far": 2.0,
        }
        expected = direct_word_render(
            boundary=boundary,
            ray_coefficients=rays[1:3],
            words=self.words[1:3],
            site_density=density,
            site_color=color,
            **common,
        )
        actual = direct_word_render(
            boundary=gather_prepared_rows(boundary, block.source_boundary_ids),
            ray_coefficients=gather_prepared_rows(rays, block.source_track_ids),
            words=compact_words,
            site_density=gather_prepared_rows(density, block.source_site_ids),
            site_color=gather_prepared_rows(color, block.source_site_ids),
            **common,
        )
        torch.testing.assert_close(actual, expected, atol=2.0e-15, rtol=2.0e-14)

    def test_storage_has_no_sample_axis_and_is_frame_count_invariant(self) -> None:
        first = prepare_worldfoam_track_block(
            self.words,
            self.boundary_site_pairs,
            site_count=5,
            track_start=0,
            track_end=2,
        )
        second = prepare_worldfoam_track_block(
            self.words,
            self.boundary_site_pairs,
            site_count=5,
            track_start=0,
            track_end=2,
        )
        self.assertEqual(first.resident_bytes, second.resident_bytes)
        for field in first.__dataclass_fields__:
            self.assertNotIn("frame", field)
            self.assertNotIn("sample", field)

    def test_fails_closed_on_invalid_words_or_boundary_pairs(self) -> None:
        with self.assertRaisesRegex(ValueError, "out-of-range boundary"):
            prepare_worldfoam_track_block(
                (make_stable_cell_word([0, 1], [-1, 9], [9, -2]),),
                self.boundary_site_pairs,
                site_count=5,
                track_start=0,
                track_end=1,
            )
        with self.assertRaisesRegex(ValueError, "out-of-range site"):
            prepare_worldfoam_track_block(
                self.words,
                torch.tensor([[0, 7]], dtype=torch.int64),
                site_count=5,
                track_start=0,
                track_end=1,
            )
        invalid_words = (
            (
                "distinct left and right",
                make_stable_cell_word([0, 1, 2], [-1, 0, 0], [0, 0, -2]),
            ),
            (
                "only the first segment may use the near cut",
                make_stable_cell_word([0, 1], [-1, -1], [-1, -2]),
            ),
            (
                "only the final segment may use the far cut",
                make_stable_cell_word([0, 1], [-1, -2], [-2, -2]),
            ),
        )
        for message, word in invalid_words:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                prepare_worldfoam_track_block(
                    (word,),
                    self.boundary_site_pairs,
                    site_count=5,
                    track_start=0,
                    track_end=1,
                )


if __name__ == "__main__":
    unittest.main()

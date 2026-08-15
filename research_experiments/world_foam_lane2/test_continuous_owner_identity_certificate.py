from __future__ import annotations

import inspect
import unittest

import torch
from compiled_transfer_adjoint import FAR_CUT_ID, NEAR_CUT_ID, StableCellWord, power_boundary_parameters
from continuous_owner_identity_certificate import (
    ContinuousOwnerIdentityError,
    certify_fixed_word_owner_identity,
)


class ContinuousOwnerIdentityCertificateTest(unittest.TestCase):
    def setUp(self) -> None:
        # Three unweighted sites on the ray. Pair rows are (0,1), (0,2),
        # (1,2), so the true word transitions use boundary ids 0 and 2.
        self.sites = torch.tensor(
            [
                [0.0, 0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        )
        self.pairs = torch.tensor([[0, 1], [0, 2], [1, 2]], dtype=torch.int64)
        self.boundary = power_boundary_parameters(self.sites, self.pairs)
        self.rays = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        self.correct_word = StableCellWord(
            owners=torch.tensor([0, 1, 2], dtype=torch.int64),
            left_cut_ids=torch.tensor([NEAR_CUT_ID, 0, 2], dtype=torch.int64),
            right_cut_ids=torch.tensor([0, 2, FAR_CUT_ID], dtype=torch.int64),
        )

    def certify(self, word: StableCellWord, **kwargs: object):
        return certify_fixed_word_owner_identity(
            sites=self.sites,
            boundary=self.boundary,
            ray_coefficients=self.rays,
            words=(word,),
            t_min=0.0,
            t_max=1.0,
            near=-1.5,
            far=1.5,
            **kwargs,
        )

    def test_true_word_checks_every_third_site_continuously(self) -> None:
        result = self.certify(self.correct_word, ownership_tolerance=1.0e-12)

        self.assertTrue(result.passed)
        self.assertTrue(result.owner_identity_certified)
        self.assertTrue(result.all_competitor_sites_checked)
        self.assertTrue(result.continuous_time_coverage)
        self.assertEqual(result.track_count, 1)
        self.assertEqual(result.run_count, 3)
        self.assertEqual(result.site_count, 3)
        self.assertGreaterEqual(result.checked_endpoint_inequality_count, 12)
        self.assertLessEqual(result.maximum_owner_difference_upper_bound, 1.0e-12)

    def test_pairwise_ordered_word_fails_when_third_site_undercuts(self) -> None:
        omitted_middle = StableCellWord(
            owners=torch.tensor([0, 2], dtype=torch.int64),
            left_cut_ids=torch.tensor([NEAR_CUT_ID, 1], dtype=torch.int64),
            right_cut_ids=torch.tensor([1, FAR_CUT_ID], dtype=torch.int64),
        )

        with self.assertRaisesRegex(ContinuousOwnerIdentityError, "third-cell undercut witness"):
            self.certify(omitted_middle, ownership_tolerance=1.0e-12)

    def test_owner_certificate_has_no_requested_frame_count(self) -> None:
        parameters = inspect.signature(certify_fixed_word_owner_identity).parameters
        self.assertNotIn("frame_count", parameters)
        self.assertNotIn("sample_count", parameters)

    def test_mutated_site_invalidates_previous_word(self) -> None:
        self.sites[1, 4] = 4.0

        with self.assertRaisesRegex(ContinuousOwnerIdentityError, "third-cell undercut witness"):
            self.certify(self.correct_word, ownership_tolerance=1.0e-12)

    def test_work_budget_fails_closed(self) -> None:
        with self.assertRaisesRegex(ContinuousOwnerIdentityError, "work budget exceeded"):
            self.certify(self.correct_word, max_work_units=1)


if __name__ == "__main__":
    unittest.main()

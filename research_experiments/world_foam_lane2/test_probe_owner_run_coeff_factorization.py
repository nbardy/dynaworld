from __future__ import annotations

import unittest

import torch

from probe_owner_run_coeff_factorization import (
    _factorized_boundary_coefficients,
    _track_ray_linear_coefficients,
)


class OwnerRunCoeffFactorizationProbeTest(unittest.TestCase):
    def test_factorized_boundary_coefficients_match_dense_formula(self) -> None:
        boundary_f32 = torch.tensor(
            [
                [2.0, -1.0, 0.5, 0.25, -0.75],
                [-0.5, 1.5, 1.0, -0.1, 0.4],
            ],
            dtype=torch.float32,
        )
        track_ray_coeff_f32 = torch.tensor(
            [
                [0.1, 0.2, 0.3, 0.01, 0.02, 0.03, 0.4, 0.5, 0.6, 0.04, 0.05, 0.06],
                [-0.2, 0.0, 0.7, 0.03, -0.02, 0.01, 0.8, -0.1, 0.2, -0.05, 0.02, 0.01],
            ],
            dtype=torch.float32,
        )

        coeff = _factorized_boundary_coefficients(
            boundary_f32=boundary_f32,
            track_ray_coeff_f32=track_ray_coeff_f32,
        )

        expected_rows = []
        for track in track_ray_coeff_f32.tolist():
            origin_base = track[0:3]
            origin_slope = track[3:6]
            direction_base = track[6:9]
            direction_slope = track[9:12]
            for boundary in boundary_f32.tolist():
                nx, ny, nz, nt, b = boundary
                expected_rows.append(
                    [
                        -(nx * origin_base[0] + ny * origin_base[1] + nz * origin_base[2] + b),
                        -(nx * origin_slope[0] + ny * origin_slope[1] + nz * origin_slope[2] + nt),
                        nx * direction_base[0] + ny * direction_base[1] + nz * direction_base[2],
                        nx * direction_slope[0] + ny * direction_slope[1] + nz * direction_slope[2],
                    ]
                )
        torch.testing.assert_close(coeff, torch.tensor(expected_rows, dtype=torch.float32))

    def test_track_ray_linear_coefficients_exact_for_affine_rays(self) -> None:
        frame_t = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
        base = torch.tensor([1.0, -2.0, 0.5, 0.25, 0.5, 1.5], dtype=torch.float32)
        slope = torch.tensor([0.1, 0.2, -0.3, 0.05, -0.02, 0.04], dtype=torch.float32)
        track = torch.stack([base + slope * t for t in frame_t], dim=0)
        coeff = _track_ray_linear_coefficients(track_rays=track.reshape(1, 3, 6), frame_t=frame_t)

        expected = torch.cat((base[:3], slope[:3], base[3:], slope[3:])).reshape(1, 12)
        torch.testing.assert_close(coeff, expected, atol=1.0e-6, rtol=0.0)


if __name__ == "__main__":
    unittest.main()

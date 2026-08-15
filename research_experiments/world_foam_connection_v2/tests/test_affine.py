from __future__ import annotations

import torch

from world_foam_connection_v2.affine import (
    AffineGenerator,
    AffineTransfer,
    compose,
    generator_exponential,
    generator_sandwich,
    physical_cone_report,
    segment_time_derivative,
)


def test_repo_order_front_red_attenuates_back_blue() -> None:
    front = AffineTransfer(
        beta=torch.tensor(0.5, dtype=torch.float64),
        moment=torch.tensor((0.5, 0.0, 0.0), dtype=torch.float64),
    )
    back = AffineTransfer(
        beta=torch.tensor(0.25, dtype=torch.float64),
        moment=torch.tensor((0.0, 0.0, 0.75), dtype=torch.float64),
    )

    result = compose(front, back)

    torch.testing.assert_close(result.beta, torch.tensor(0.125, dtype=torch.float64))
    torch.testing.assert_close(
        result.moment,
        torch.tensor((0.5, 0.0, 0.375), dtype=torch.float64),
    )
    assert not torch.allclose(compose(back, front).moment, result.moment)


def test_physical_p0_exponential_stays_in_cone() -> None:
    generator = AffineGenerator(
        scalar=torch.tensor(-1.3, dtype=torch.float64),
        source=torch.tensor((1.04, 0.26, 0.13), dtype=torch.float64),
    )

    transfer = generator_exponential(
        generator,
        torch.tensor(2.1, dtype=torch.float64),
    )

    assert physical_cone_report(transfer, tolerance=1.0e-12).passed


def test_exact_segment_rate_matches_central_difference() -> None:
    generator = AffineGenerator(
        scalar=torch.tensor(-0.9, dtype=torch.float64),
        source=torch.tensor((0.6, 0.2, 0.1), dtype=torch.float64),
    )
    generator_rate = AffineGenerator(
        scalar=torch.tensor(-0.07, dtype=torch.float64),
        source=torch.tensor((0.02, -0.01, 0.03), dtype=torch.float64),
    )
    length = torch.tensor(1.4, dtype=torch.float64)
    length_rate = torch.tensor(0.2, dtype=torch.float64)
    derivative = segment_time_derivative(
        generator,
        generator_rate,
        length,
        length_rate,
    ).as_vector()
    epsilon = 1.0e-6

    def shifted(sign: float) -> torch.Tensor:
        return generator_exponential(
            AffineGenerator(
                generator.scalar + sign * epsilon * generator_rate.scalar,
                generator.source + sign * epsilon * generator_rate.source,
            ),
            length + sign * epsilon * length_rate,
        ).as_vector()

    finite_difference = (shifted(1.0) - shifted(-1.0)) / (2.0 * epsilon)
    torch.testing.assert_close(
        derivative,
        finite_difference,
        rtol=2.0e-8,
        atol=2.0e-9,
    )


def test_generator_sandwich_retains_suffix_moment() -> None:
    prefix = AffineTransfer(
        torch.tensor(0.8, dtype=torch.float64),
        torch.tensor((0.1, 0.0, 0.0), dtype=torch.float64),
    )
    generator = AffineGenerator(
        torch.tensor(-1.2, dtype=torch.float64),
        torch.tensor((0.0, 0.5, 0.0), dtype=torch.float64),
    )
    suffix = AffineTransfer(
        torch.tensor(0.7, dtype=torch.float64),
        torch.tensor((0.0, 0.0, 0.4), dtype=torch.float64),
    )

    tangent = generator_sandwich(prefix, generator, suffix)

    torch.testing.assert_close(
        tangent.moment,
        torch.tensor((0.0, 0.4, -0.384), dtype=torch.float64),
    )

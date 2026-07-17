from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


DTYPE = torch.float64
SEED = 0
RENDER_MAX_ABS_ERROR = 1.0e-12
ELEMENT_MAX_ABS_ERROR = 1.0e-12
GRAD_MAX_ABS_ERROR = 1.0e-6
FINITE_DIFFERENCE_EPSILON = 1.0e-5
DEFAULT_OUT = Path("outputs/benchmarks/2026-07-08_worldfoam_cell_path_optical_transfer_summary.json")


@dataclass(frozen=True)
class TransferElement:
    beta: torch.Tensor
    m: torch.Tensor


def _scalar(value: float | torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, dtype=DTYPE).reshape(())


def _vec(value: list[float] | tuple[float, ...] | torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, dtype=DTYPE).reshape(-1)


def _to_dtype(tensor: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(tensor, dtype=DTYPE).contiguous()


def compose(front: TransferElement, back: TransferElement) -> TransferElement:
    """Compose optical-transfer elements so F_comp(g) = F_front(F_back(g))."""

    return TransferElement(
        beta=_scalar(front.beta * back.beta),
        m=_to_dtype(front.m + front.beta * back.m),
    )


def scan(elements: list[TransferElement] | tuple[TransferElement, ...]) -> TransferElement:
    result = TransferElement(beta=_scalar(1.0), m=torch.zeros(3, dtype=DTYPE))
    for element in elements:
        result = compose(result, element)
    return result


def decode(element: TransferElement, background: torch.Tensor) -> torch.Tensor:
    return _to_dtype(element.m + element.beta * _vec(background))


def constant_run_element(sigma: float | torch.Tensor, length: float | torch.Tensor, color: torch.Tensor) -> TransferElement:
    tau = _scalar(sigma) * _scalar(length)
    beta = torch.exp(-tau)
    return TransferElement(beta=beta, m=(1.0 - beta) * _vec(color))


def render_word(
    sigmas: torch.Tensor,
    lengths: torch.Tensor,
    colors: torch.Tensor,
    background: torch.Tensor,
) -> torch.Tensor:
    elements = [
        constant_run_element(sigma, length, color)
        for sigma, length, color in zip(_to_dtype(sigmas), _to_dtype(lengths), _to_dtype(colors), strict=True)
    ]
    return render_word_from_elements(elements, background)


def render_word_from_elements(
    elements: list[TransferElement] | tuple[TransferElement, ...],
    background: torch.Tensor,
) -> torch.Tensor:
    return decode(scan(elements), _vec(background))


def make_two_run_fixture() -> dict[str, torch.Tensor]:
    return {
        "sigmas": torch.tensor([0.35, 0.90], dtype=DTYPE),
        "lengths": torch.tensor([0.80, 0.45], dtype=DTYPE),
        "colors": torch.tensor([[0.90, 0.20, 0.08], [0.08, 0.62, 0.95]], dtype=DTYPE),
        "background": torch.tensor([0.02, 0.03, 0.05], dtype=DTYPE),
        "target": torch.tensor([0.34, 0.22, 0.28], dtype=DTYPE),
    }


def make_three_run_fixture() -> dict[str, torch.Tensor]:
    return {
        "sigmas": torch.tensor([0.25, 0.75, 0.42], dtype=DTYPE),
        "lengths": torch.tensor([0.60, 0.35, 1.10], dtype=DTYPE),
        "colors": torch.tensor(
            [[0.86, 0.12, 0.05], [0.12, 0.55, 0.92], [0.38, 0.82, 0.28]],
            dtype=DTYPE,
        ),
        "background": torch.tensor([0.03, 0.04, 0.06], dtype=DTYPE),
        "target": torch.tensor([0.30, 0.36, 0.24], dtype=DTYPE),
    }


def _elements_from_runs(sigmas: torch.Tensor, lengths: torch.Tensor, colors: torch.Tensor) -> list[TransferElement]:
    return [
        constant_run_element(sigma, length, color)
        for sigma, length, color in zip(_to_dtype(sigmas), _to_dtype(lengths), _to_dtype(colors), strict=True)
    ]


def same_representation_replay_fixture() -> dict[str, Any]:
    fixture = make_three_run_fixture()
    compiled_elements = _elements_from_runs(fixture["sigmas"], fixture["lengths"], fixture["colors"])
    compiled_element = scan(compiled_elements)
    compiled_color = decode(compiled_element, fixture["background"])
    replay_color = render_word(fixture["sigmas"], fixture["lengths"], fixture["colors"], fixture["background"])
    replay_element = scan(_elements_from_runs(fixture["sigmas"], fixture["lengths"], fixture["colors"]))
    return {
        "compiled_color": compiled_color,
        "replay_color": replay_color,
        "compiled_element": compiled_element,
        "replay_element": replay_element,
        "render_max_abs_error": float((compiled_color - replay_color).abs().max().item()),
        "element_beta_max_abs_error": float((compiled_element.beta - replay_element.beta).abs().item()),
        "element_m_max_abs_error": float((compiled_element.m - replay_element.m).abs().max().item()),
    }


def _loss_from_color(color: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    residual = _vec(color) - _vec(target)
    return 0.5 * residual.square().sum()


def _loss_from_elements(
    elements: list[TransferElement],
    background: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return _loss_from_color(render_word_from_elements(elements, background), target)


def _loss_from_runs(
    sigmas: torch.Tensor,
    lengths: torch.Tensor,
    colors: torch.Tensor,
    background: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return _loss_from_color(render_word(sigmas, lengths, colors, background), target)


def analytic_prefix_suffix_vjp(
    sigmas: torch.Tensor,
    lengths: torch.Tensor,
    colors: torch.Tensor,
    background: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    sigmas = _to_dtype(sigmas)
    lengths = _to_dtype(lengths)
    colors = _to_dtype(colors)
    background = _vec(background)
    target = _vec(target)
    elements = _elements_from_runs(sigmas, lengths, colors)
    color = render_word_from_elements(elements, background)
    grad_color_out = color - target
    count = int(sigmas.numel())

    grad_beta = torch.zeros(count, dtype=DTYPE)
    grad_m = torch.zeros((count, 3), dtype=DTYPE)
    grad_delta_tau = torch.zeros(count, dtype=DTYPE)
    grad_sigma = torch.zeros(count, dtype=DTYPE)
    grad_length = torch.zeros(count, dtype=DTYPE)
    grad_color = torch.zeros((count, 3), dtype=DTYPE)

    prefix_beta = _scalar(1.0)
    suffix_colors = [
        render_word_from_elements(elements[index + 1 :], background)
        for index in range(count)
    ]
    for index, element in enumerate(elements):
        c_after = suffix_colors[index]
        grad_m[index] = prefix_beta * grad_color_out
        grad_beta[index] = torch.dot(grad_color_out, prefix_beta * c_after)
        grad_delta_tau[index] = torch.dot(
            grad_color_out,
            prefix_beta * element.beta * (colors[index] - c_after),
        )
        grad_sigma[index] = lengths[index] * grad_delta_tau[index]
        grad_length[index] = sigmas[index] * grad_delta_tau[index]
        grad_color[index] = prefix_beta * (1.0 - element.beta) * grad_color_out
        prefix_beta = prefix_beta * element.beta

    return {
        "color": color,
        "loss": _loss_from_color(color, target),
        "beta": grad_beta,
        "m": grad_m,
        "delta_tau": grad_delta_tau,
        "sigma": grad_sigma,
        "length": grad_length,
        "color_grad": grad_color,
    }


def _central_difference_scalar(fn, base: torch.Tensor, index: int, epsilon: float) -> torch.Tensor:
    plus = base.clone()
    minus = base.clone()
    plus[index] += epsilon
    minus[index] -= epsilon
    return (fn(plus) - fn(minus)) / (2.0 * epsilon)


def _central_difference_matrix(fn, base: torch.Tensor, row: int, col: int, epsilon: float) -> torch.Tensor:
    plus = base.clone()
    minus = base.clone()
    plus[row, col] += epsilon
    minus[row, col] -= epsilon
    return (fn(plus) - fn(minus)) / (2.0 * epsilon)


def finite_difference_vjp(
    sigmas: torch.Tensor,
    lengths: torch.Tensor,
    colors: torch.Tensor,
    background: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = FINITE_DIFFERENCE_EPSILON,
) -> dict[str, torch.Tensor]:
    sigmas = _to_dtype(sigmas)
    lengths = _to_dtype(lengths)
    colors = _to_dtype(colors)
    background = _vec(background)
    target = _vec(target)
    count = int(sigmas.numel())
    base_elements = _elements_from_runs(sigmas, lengths, colors)
    base_beta = torch.stack([element.beta for element in base_elements])
    base_m = torch.stack([element.m for element in base_elements])
    base_tau = sigmas * lengths

    def loss_beta(beta_values: torch.Tensor) -> torch.Tensor:
        elements = [
            TransferElement(beta=_scalar(beta_values[index]), m=base_m[index].clone())
            for index in range(count)
        ]
        return _loss_from_elements(elements, background, target)

    def loss_m(m_values: torch.Tensor) -> torch.Tensor:
        elements = [
            TransferElement(beta=base_beta[index].clone(), m=m_values[index].clone())
            for index in range(count)
        ]
        return _loss_from_elements(elements, background, target)

    def loss_tau(tau_values: torch.Tensor) -> torch.Tensor:
        beta = torch.exp(-tau_values)
        elements = [
            TransferElement(beta=_scalar(beta[index]), m=(1.0 - beta[index]) * colors[index])
            for index in range(count)
        ]
        return _loss_from_elements(elements, background, target)

    def loss_sigma(sigma_values: torch.Tensor) -> torch.Tensor:
        return _loss_from_runs(sigma_values, lengths, colors, background, target)

    def loss_length(length_values: torch.Tensor) -> torch.Tensor:
        return _loss_from_runs(sigmas, length_values, colors, background, target)

    def loss_color(color_values: torch.Tensor) -> torch.Tensor:
        return _loss_from_runs(sigmas, lengths, color_values, background, target)

    grad_beta = torch.zeros(count, dtype=DTYPE)
    grad_m = torch.zeros((count, 3), dtype=DTYPE)
    grad_tau = torch.zeros(count, dtype=DTYPE)
    grad_sigma = torch.zeros(count, dtype=DTYPE)
    grad_length = torch.zeros(count, dtype=DTYPE)
    grad_color = torch.zeros((count, 3), dtype=DTYPE)
    for index in range(count):
        grad_beta[index] = _central_difference_scalar(loss_beta, base_beta, index, epsilon)
        grad_tau[index] = _central_difference_scalar(loss_tau, base_tau, index, epsilon)
        grad_sigma[index] = _central_difference_scalar(loss_sigma, sigmas, index, epsilon)
        grad_length[index] = _central_difference_scalar(loss_length, lengths, index, epsilon)
        for channel in range(3):
            grad_m[index, channel] = _central_difference_matrix(loss_m, base_m, index, channel, epsilon)
            grad_color[index, channel] = _central_difference_matrix(loss_color, colors, index, channel, epsilon)
    return {
        "beta": grad_beta,
        "m": grad_m,
        "delta_tau": grad_tau,
        "sigma": grad_sigma,
        "length": grad_length,
        "color_grad": grad_color,
    }


def _max_grad_error(analytic: dict[str, torch.Tensor], finite: dict[str, torch.Tensor]) -> dict[str, float]:
    keys = ("beta", "m", "delta_tau", "sigma", "length", "color_grad")
    return {
        key: float((analytic[key] - finite[key]).abs().max().item())
        for key in keys
    }


def commutator_swap_probe() -> dict[str, Any]:
    fixture = make_two_run_fixture()
    elements = _elements_from_runs(fixture["sigmas"], fixture["lengths"], fixture["colors"])
    word_ab = render_word_from_elements([elements[0], elements[1]], fixture["background"])
    word_ba = render_word_from_elements([elements[1], elements[0]], fixture["background"])
    alpha_a = 1.0 - elements[0].beta
    alpha_b = 1.0 - elements[1].beta
    expected = alpha_a * alpha_b * (fixture["colors"][0] - fixture["colors"][1])
    measured = word_ab - word_ba
    return {
        "measured": measured,
        "expected": expected,
        "max_abs_error": float((measured - expected).abs().max().item()),
        "measured_norm": float(measured.norm().item()),
        "expected_norm": float(expected.norm().item()),
    }


def _associativity_check() -> dict[str, float]:
    fixture = make_three_run_fixture()
    elements = _elements_from_runs(fixture["sigmas"], fixture["lengths"], fixture["colors"])
    left = compose(compose(elements[0], elements[1]), elements[2])
    right = compose(elements[0], compose(elements[1], elements[2]))
    return {
        "beta": float((left.beta - right.beta).abs().item()),
        "m": float((left.m - right.m).abs().max().item()),
    }


def _alpha_equivalence_check() -> dict[str, float]:
    fixture = make_two_run_fixture()
    sigma = fixture["sigmas"][0]
    length = fixture["lengths"][0]
    color = fixture["colors"][0]
    background = fixture["background"]
    element = constant_run_element(sigma, length, color)
    rendered = decode(element, background)
    alpha = 1.0 - torch.exp(-(sigma * length))
    expected = alpha * color + (1.0 - alpha) * background
    return {
        "render": float((rendered - expected).abs().max().item()),
        "beta": float((element.beta - (1.0 - alpha)).abs().item()),
        "m": float((element.m - alpha * color).abs().max().item()),
    }


def run_all_checks() -> dict[str, Any]:
    torch.manual_seed(SEED)
    fixture = make_three_run_fixture()
    replay = same_representation_replay_fixture()
    analytic = analytic_prefix_suffix_vjp(
        fixture["sigmas"],
        fixture["lengths"],
        fixture["colors"],
        fixture["background"],
        fixture["target"],
    )
    finite = finite_difference_vjp(
        fixture["sigmas"],
        fixture["lengths"],
        fixture["colors"],
        fixture["background"],
        fixture["target"],
        FINITE_DIFFERENCE_EPSILON,
    )
    grad_errors = _max_grad_error(analytic, finite)
    associativity = _associativity_check()
    alpha = _alpha_equivalence_check()
    commutator = commutator_swap_probe()
    max_errors = {
        "render": float(replay["render_max_abs_error"]),
        "element": max(float(replay["element_beta_max_abs_error"]), float(replay["element_m_max_abs_error"])),
        "grad": max(grad_errors.values()),
        "commutator": float(commutator["max_abs_error"]),
        "associativity": max(associativity.values()),
        "alpha_equivalence": max(alpha.values()),
    }
    checks = {
        "monoid_associative": "ok" if max_errors["associativity"] <= ELEMENT_MAX_ABS_ERROR else "failed",
        "alpha_equivalence": "ok" if max_errors["alpha_equivalence"] <= RENDER_MAX_ABS_ERROR else "failed",
        "replay_equivalence": (
            "ok"
            if max_errors["render"] <= RENDER_MAX_ABS_ERROR and max_errors["element"] <= ELEMENT_MAX_ABS_ERROR
            else "failed"
        ),
        "vjp_finite_difference": "ok" if max_errors["grad"] <= GRAD_MAX_ABS_ERROR else "failed",
        "commutator_swap": "ok" if max_errors["commutator"] <= RENDER_MAX_ABS_ERROR else "failed",
    }
    status = "ok" if all(value == "ok" for value in checks.values()) else "failed"
    return {
        "status": status,
        "dtype": "float64",
        "seed": SEED,
        "fixture": "constant_density_owner_run_word",
        "thresholds": {
            "render_max_abs_error": RENDER_MAX_ABS_ERROR,
            "element_max_abs_error": ELEMENT_MAX_ABS_ERROR,
            "grad_max_abs_error": GRAD_MAX_ABS_ERROR,
            "finite_difference_epsilon": FINITE_DIFFERENCE_EPSILON,
        },
        "checks": checks,
        "max_errors": max_errors,
        "grad_errors": grad_errors,
        "replay_equivalence": {
            "render_max_abs_error": replay["render_max_abs_error"],
            "element_beta_max_abs_error": replay["element_beta_max_abs_error"],
            "element_m_max_abs_error": replay["element_m_max_abs_error"],
        },
        "commutator_swap": {
            "max_abs_error": commutator["max_abs_error"],
            "measured_norm": commutator["measured_norm"],
            "expected_norm": commutator["expected_norm"],
        },
    }


def verify_summary(result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if result.get("status") != "ok":
        errors.append("status must be ok")
    if result.get("dtype") != "float64":
        errors.append("dtype must be float64")
    if result.get("fixture") != "constant_density_owner_run_word":
        errors.append("fixture must be constant_density_owner_run_word")
    checks = result.get("checks")
    if not isinstance(checks, dict):
        errors.append("checks must be present")
        checks = {}
    for key in (
        "monoid_associative",
        "alpha_equivalence",
        "replay_equivalence",
        "vjp_finite_difference",
        "commutator_swap",
    ):
        if checks.get(key) != "ok":
            errors.append(f"check {key} must be ok")
    thresholds = result.get("thresholds")
    max_errors = result.get("max_errors")
    if not isinstance(thresholds, dict) or not isinstance(max_errors, dict):
        errors.append("thresholds and max_errors must be present")
        return errors
    if float(max_errors.get("render", math.inf)) > float(thresholds.get("render_max_abs_error", 0.0)):
        errors.append("render error exceeds threshold")
    if float(max_errors.get("element", math.inf)) > float(thresholds.get("element_max_abs_error", 0.0)):
        errors.append("element error exceeds threshold")
    if float(max_errors.get("grad", math.inf)) > float(thresholds.get("grad_max_abs_error", 0.0)):
        errors.append("grad error exceeds threshold")
    if float(max_errors.get("commutator", math.inf)) > float(thresholds.get("render_max_abs_error", 0.0)):
        errors.append("commutator error exceeds render threshold")
    return errors


def assert_summary(result: dict[str, Any]) -> None:
    errors = verify_summary(result)
    if errors:
        raise AssertionError("cell-path optical-transfer fixture failed:\n- " + "\n- ".join(errors))


def write_summary_json(path: str | Path, result: dict[str, Any]) -> Path:
    assert_summary(result)
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--verify-report", type=Path, help="validate an existing summary JSON without rerunning")
    args = parser.parse_args()

    if args.verify_report is not None:
        result = json.loads(args.verify_report.read_text(encoding="utf-8"))
        assert_summary(result)
        print(f"verified {args.verify_report}")
        return

    result = run_all_checks()
    path = write_summary_json(args.out, result)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

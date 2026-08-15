from __future__ import annotations

import re
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
VARIANT = ROOT / "third_party" / "fast-mac-gsplat" / "variants" / "world_foam_lane2_fused_slab_v0"
METAL = VARIANT / "csrc" / "metal" / "world_foam_lane2_shared_replay_tensor.metal"
HOST = VARIANT / "csrc" / "metal" / "world_foam_lane2_metal.mm"
BINDINGS = VARIANT / "csrc" / "bindings.cpp"
OPS = VARIANT / "torch_world_foam_lane2_fused_slab" / "ops.py"
PACKAGE = VARIANT / "torch_world_foam_lane2_fused_slab" / "__init__.py"


def _braced_source(source: str, marker: str) -> str:
    start = source.index(marker)
    cursor = source.index("{", start)
    depth = 0
    while cursor < len(source):
        if source[cursor] == "{":
            depth += 1
        elif source[cursor] == "}":
            depth -= 1
            if depth == 0:
                return source[start : cursor + 1]
        cursor += 1
    raise AssertionError(f"unterminated source block: {marker}")


def _python_function(source: str, name: str) -> str:
    start = source.index(f"def {name}(")
    end = source.find("\ndef ", start + 5)
    return source[start : end if end >= 0 else None]


def _phi(kappa: torch.Tensor) -> torch.Tensor:
    return -torch.expm1(-kappa) / kappa


def _phi_prime(kappa: torch.Tensor) -> torch.Tensor:
    return ((kappa + 1.0) * torch.exp(-kappa) - 1.0) / kappa.square()


def test_cpu_oracle_uses_row_local_nodes_and_explicit_loss_scale() -> None:
    node_chart = torch.tensor(
        [
            [[0.4, 0.10, 0.08, 0.06], [0.7, 0.18, 0.12, 0.09]],
            [[0.5, 0.14, 0.07, 0.11], [0.9, 0.21, 0.19, 0.15]],
            [[0.6, 0.08, 0.16, 0.10], [0.8, 0.17, 0.22, 0.13]],
            [[0.3, 0.04, 0.05, 0.03], [0.4, 0.06, 0.07, 0.05]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    sample_rows = torch.tensor([2, 0, 2, 1], dtype=torch.int64)
    weights = torch.tensor(
        [[0.25, 0.75], [0.6, 0.4], [0.8, 0.2], [0.35, 0.65]],
        dtype=torch.float64,
    )
    targets = torch.tensor(
        [[0.24, 0.31, 0.18], [0.32, 0.19, 0.28], [0.14, 0.27, 0.33], [0.21, 0.23, 0.16]],
        dtype=torch.float64,
    )
    background = torch.tensor([0.05, 0.1, 0.2], dtype=torch.float64)
    loss_scale = 0.037

    selected_nodes = node_chart.index_select(0, sample_rows)
    chart = (selected_nodes * weights[:, :, None]).sum(dim=1)
    kappa = chart[:, 0]
    velocity = chart[:, 1:]
    beta = torch.exp(-kappa)
    phi = _phi(kappa)
    prediction = phi[:, None] * velocity + beta[:, None] * background
    residual = prediction - targets
    loss = residual.square().sum() * loss_scale
    loss.backward()

    grad_prediction = 2.0 * loss_scale * residual.detach()
    grad_beta = (grad_prediction * background).sum(dim=1)
    grad_chart = torch.cat(
        (
            (
                -beta.detach() * grad_beta
                + _phi_prime(kappa.detach()) * (velocity.detach() * grad_prediction).sum(dim=1)
            )[:, None],
            phi.detach()[:, None] * grad_prediction,
        ),
        dim=1,
    )
    expected_grad = torch.zeros_like(node_chart)
    for sample_id, row_id in enumerate(sample_rows.tolist()):
        for node_id in range(weights.shape[1]):
            expected_grad[row_id, node_id] += weights[sample_id, node_id] * grad_chart[sample_id]

    assert prediction.shape == (sample_rows.numel(), 3)
    torch.testing.assert_close(loss.detach(), residual.detach().square().sum() * loss_scale)
    torch.testing.assert_close(node_chart.grad, expected_grad, rtol=2.0e-12, atol=2.0e-12)
    torch.testing.assert_close(node_chart.grad[3], torch.zeros_like(node_chart.grad[3]))
    assert node_chart.grad[2].abs().sum() > node_chart.grad[0].abs().sum()


def test_ragged_metal_kernels_share_one_row_selected_arithmetic_helper() -> None:
    source = METAL.read_text(encoding="utf-8")
    helper = _braced_source(source, "inline bool wf2_kinetic_ragged_p0_lie_sample_mse_vjp(")
    prediction_kernel = _braced_source(source, "kernel void wf2_kinetic_ragged_p0_lie_sample_mse_vjp_tensor(")
    loss_only_kernel = _braced_source(
        source,
        "kernel void wf2_kinetic_ragged_p0_lie_sample_mse_vjp_accumulate_only_tensor(",
    )

    for kernel in (prediction_kernel, loss_only_kernel):
        assert kernel.count("wf2_kinetic_ragged_p0_lie_sample_mse_vjp(") == 1
    for required in (
        "const uint row_count = uint(config_i32[0]);",
        "const uint node_count = uint(config_i32[1]);",
        "const uint sample_count = uint(config_i32[2]);",
        "const int row_raw = sample_row_i32[gid];",
        "const uint weight_base = gid * node_count;",
        "const uint node_base = row_id * node_count;",
        "const float cone_tolerance = config_f32[0];",
        "const float loss_scale = config_f32[1];",
        "(node_base + node_id) * 4u",
        "sample_to_node_f32[weight_base + node_id] * grad_chart",
    ):
        assert required in helper
    for forbidden in (
        "track_id",
        "frame_count",
        "row_count * sample_count",
        "gid / sample_count",
        "common_refinement",
    ):
        assert forbidden not in helper
    assert "device float* prediction_rgb_f32" in prediction_kernel
    assert "prediction_rgb_f32" not in loss_only_kernel


def test_ragged_launch_wiring_is_bounded_and_schemas_parse() -> None:
    host_source = HOST.read_text(encoding="utf-8")
    bindings_source = BINDINGS.read_text(encoding="utf-8")
    ops_source = OPS.read_text(encoding="utf-8")
    package_source = PACKAGE.read_text(encoding="utf-8")
    prediction_name = "kinetic_ragged_p0_lie_sample_accumulate_launch_only"
    loss_only_name = "kinetic_ragged_p0_lie_sample_accumulate_loss_only_launch_only"

    schemas = {
        schema.split("(", 1)[0]: schema
        for schema in re.findall(r'm\.def\(\s*"([^"]+)"', bindings_source, flags=re.DOTALL)
    }
    for name in (prediction_name, loss_only_name):
        torch._C.parse_schema(schemas[name])
        assert f'"{name}"' in bindings_source
        assert f"metal_{name}(" in host_source
        assert f"def {name}(" in ops_source
        assert name in package_source
    assert "Tensor sample_row_i32" in schemas[prediction_name]
    assert "Tensor(a!) loss_f32" in schemas[loss_only_name]
    assert "Tensor(b!) grad_node_chart_f32" in schemas[loss_only_name]
    assert "Tensor(c!) cone_diagnostic_i32" in schemas[loss_only_name]

    prediction_host = _braced_source(host_source, f"torch::Tensor metal_{prediction_name}(")
    loss_only_host = _braced_source(
        host_source,
        f"std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>\nmetal_{loss_only_name}(",
    )
    assert prediction_host.count("torch::empty(") == 1
    assert "{sample_count, 3}" in prediction_host
    for forbidden in ("torch::empty(", "torch::zeros(", ".cpu()", ".contiguous()"):
        assert forbidden not in loss_only_host
    assert "fn.dispatch((uint64_t)sample_count, threads);" in prediction_host
    assert "fn.dispatch((uint64_t)sample_count, threads);" in loss_only_host

    preparation = _python_function(ops_source, "prepare_kinetic_ragged_p0_lie_sample_block")
    for required in (
        'sample_row_i32.device.type != "cpu"',
        "sample_row_i32.is_contiguous()",
        "sample_row_i32.min().item()",
        "sample_row_i32.max().item()",
        "[row_count, node_count, sample_count]",
        "[normalized_cone_tolerance, normalized_loss_scale]",
        "tensor_signatures=_capture_tensor_signatures(tensors)",
    ):
        assert required in preparation
    assert '.to(device="cpu", dtype=torch.int64)' not in preparation
    for name in (prediction_name, loss_only_name):
        launch = _python_function(ops_source, name)
        for forbidden in (".cpu()", ".contiguous()", "torch.empty", "torch.zeros", "torch.tensor"):
            assert forbidden not in launch

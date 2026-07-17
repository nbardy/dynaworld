from __future__ import annotations

import torch

from benchmark_gradients import module_parameter_grads, named_module_parameter_grads, sequence_leaf_grads
from runtime_types import GaussianSequence


def _sequence() -> GaussianSequence:
    return GaussianSequence(
        xyz=torch.ones(1, 2, 3, requires_grad=True),
        scales=torch.ones(1, 2, 3, requires_grad=True),
        quats=torch.ones(1, 2, 4, requires_grad=True),
        opacities=torch.ones(1, 2, 1, requires_grad=True),
        rgbs=torch.ones(1, 2, 5, requires_grad=True),
    )


def test_sequence_leaf_grads_supports_none_and_zero_missing_policies() -> None:
    sequence = _sequence()
    none_grads = sequence_leaf_grads(sequence)
    assert none_grads["xyz"] is None

    zero_grads = sequence_leaf_grads(sequence, missing="zero")
    assert torch.equal(zero_grads["xyz"], torch.zeros_like(sequence.xyz))
    assert torch.equal(zero_grads["rgbs"], torch.zeros_like(sequence.rgbs))


def test_sequence_leaf_grads_rejects_unknown_missing_policy() -> None:
    sequence = _sequence()
    try:
        sequence_leaf_grads(sequence, missing="bad")  # type: ignore[arg-type]
    except ValueError as exc:
        assert "Unsupported missing gradient policy" in str(exc)
    else:
        raise AssertionError("sequence_leaf_grads accepted an unknown missing-gradient policy")


def test_sequence_leaf_grads_can_clone_detached_gradients() -> None:
    sequence = _sequence()
    loss = sequence.xyz.sum() + sequence.rgbs.sum()
    loss.backward()
    grads = sequence_leaf_grads(sequence, clone=True)
    assert grads["xyz"] is not sequence.xyz.grad
    assert torch.equal(grads["xyz"], sequence.xyz.grad)
    assert grads["scales"] is None


def test_module_grad_helpers_collect_flat_named_parameter_grads() -> None:
    module = torch.nn.Linear(2, 1)
    module(torch.ones(1, 2)).sum().backward()

    grads = module_parameter_grads(module)
    assert set(grads) == {"weight", "bias"}
    assert grads["weight"] is not module.weight.grad
    assert torch.equal(grads["weight"], module.weight.grad)

    named = named_module_parameter_grads({"head": module, "missing": None})
    assert set(named) == {"head.weight", "head.bias"}

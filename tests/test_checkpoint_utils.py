from __future__ import annotations

from pathlib import Path

import pytest
import torch

from checkpoint_utils import load_checkpoint_mapping, load_torch_checkpoint, model_state_dict_from_checkpoint


def test_load_torch_checkpoint_supports_weights_only_payloads(tmp_path: Path) -> None:
    path = tmp_path / "weights.pt"
    torch.save({"weight": torch.ones(2, 3)}, path)

    payload = load_torch_checkpoint(path, map_location="cpu", weights_only=True)

    assert isinstance(payload, dict)
    torch.testing.assert_close(payload["weight"], torch.ones(2, 3))


def test_load_checkpoint_mapping_rejects_non_mapping_payload(tmp_path: Path) -> None:
    path = tmp_path / "bad.pt"
    torch.save(["not", "a", "mapping"], path)

    with pytest.raises(ValueError, match="Fixture checkpoint .* must contain a mapping payload"):
        load_checkpoint_mapping(path, map_location="cpu", label="Fixture checkpoint")


def test_model_state_dict_from_checkpoint_accepts_wrapped_and_raw_state_dicts() -> None:
    state = {"weight": torch.ones(2, 3)}

    assert model_state_dict_from_checkpoint({"model": state}) is state
    assert model_state_dict_from_checkpoint(state) is state

    with pytest.raises(ValueError, match="Expected a checkpoint dict"):
        model_state_dict_from_checkpoint({"model": "not a state dict"})

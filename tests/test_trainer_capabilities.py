from __future__ import annotations

from trainer_capabilities import trainer_has_capabilities, trainer_uses_multicam_phase


class _DummyTrainer:
    pass


def test_trainer_uses_multicam_phase_requires_all_multicam_capabilities() -> None:
    trainer = _DummyTrainer()
    trainer.sample_multicam_clip = object()
    trainer.multicam_bundle = object()
    trainer.camera_rig = object()
    assert not trainer_uses_multicam_phase(trainer)

    trainer.rig_regularization_loss = object()
    assert trainer_uses_multicam_phase(trainer)


def test_trainer_has_capabilities_checks_named_attributes() -> None:
    trainer = _DummyTrainer()
    trainer.alpha = object()
    trainer.beta = object()
    assert trainer_has_capabilities(trainer, ("alpha", "beta"))
    assert not trainer_has_capabilities(trainer, ("alpha", "gamma"))

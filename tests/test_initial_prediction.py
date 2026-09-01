import numpy as np
import pytest

from candidate_runtime.initial_prediction import (
    InitialPredictionConfig,
    _load_foreground_probability,
)


def test_initial_prediction_defaults_match_release_policy(monkeypatch):
    for name in (
        "AUTOPET_FDG_PROBABILITY_THRESHOLD",
        "AUTOPET_FDG_RELAX_BURDEN_COMPONENTS",
        "AUTOPET_PSMA_PRUNE_THRESHOLD",
    ):
        monkeypatch.delenv(name, raising=False)
    config = InitialPredictionConfig.from_environment()
    assert config.fdg_probability_threshold == pytest.approx(0.47)
    assert config.fdg_relax_burden_components == 25
    assert config.psma_prune_threshold == pytest.approx(0.86)


def test_initial_prediction_configuration_accepts_environment_overrides(monkeypatch):
    monkeypatch.setenv("AUTOPET_FDG_PROBABILITY_THRESHOLD", "0.45")
    monkeypatch.setenv("AUTOPET_FDG_RELAX_BURDEN_COMPONENTS", "30")
    monkeypatch.setenv("AUTOPET_PSMA_PRUNE_THRESHOLD", "0.91")
    config = InitialPredictionConfig.from_environment()
    assert config == InitialPredictionConfig(0.45, 30, 0.91)


def test_probability_loader_normalizes_zyx_archive(tmp_path):
    path = tmp_path / "prediction.npz"
    probabilities = np.zeros((2, 3, 4, 5), dtype=np.float32)
    probabilities[1, 1, 2, 3] = 0.75
    np.savez(path, probabilities=probabilities)
    foreground = _load_foreground_probability(path, (3, 4, 5))
    assert foreground.shape == (5, 4, 3)
    assert foreground[3, 2, 1] == pytest.approx(0.75)


def test_probability_loader_rejects_unresolvable_geometry(tmp_path):
    path = tmp_path / "prediction.npz"
    np.savez(path, probabilities=np.zeros((2, 2, 2, 2), dtype=np.float32))
    with pytest.raises(RuntimeError, match="grid mismatch"):
        _load_foreground_probability(path, (3, 4, 5))

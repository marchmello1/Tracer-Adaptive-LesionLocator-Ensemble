from pathlib import Path

import numpy as np
import pytest

import candidate_runtime.interactive_update as interactive_update
from candidate_runtime.interactive_update import (
    InteractionPolicy,
    InteractiveBackend,
    refine_prediction,
)


def test_interaction_policy_defaults_are_conservative(monkeypatch):
    for name in (
        "AUTOPET_PSMA_INTERACTION_MAX_COMPONENTS",
        "AUTOPET_BACKGROUND_CONSENSUS",
        "AUTOPET_BACKGROUND_MAX_RETAINED_FRACTION",
    ):
        monkeypatch.delenv(name, raising=False)
    assert InteractionPolicy.from_environment() == InteractionPolicy(128, False, 0.5)


def test_empty_interaction_returns_exact_binary_initial_mask(tmp_path):
    initial = np.zeros((3, 4, 5), dtype=np.uint8)
    initial[1, 2, 3] = 1
    actual = refine_prediction(
        initial,
        None,
        None,
        {"tumor": [], "background": []},
        "fdg",
        tmp_path,
    )
    np.testing.assert_array_equal(actual, initial)


def test_interactive_donor_geometry_is_checked_before_fusion(monkeypatch, tmp_path):
    monkeypatch.setattr(
        interactive_update,
        "_infer_donor",
        lambda *args, **kwargs: np.zeros((2, 2, 2), dtype=bool),
    )
    with pytest.raises(RuntimeError, match="donor grid mismatch"):
        refine_prediction(
            np.zeros((3, 4, 5), dtype=np.uint8),
            None,
            None,
            {"tumor": [[1, 1, 1]], "background": []},
            "fdg",
            tmp_path,
            backend=InteractiveBackend(Path("source"), Path("weights"), Path("runner")),
            policy=InteractionPolicy(128, False, 0.5),
        )


def test_refinement_adds_safe_disconnected_prompted_lesion(monkeypatch, tmp_path):
    donor_xyz = np.zeros((5, 4, 3), dtype=bool)
    donor_xyz[4, 3, 2] = True
    monkeypatch.setattr(
        interactive_update, "_infer_donor", lambda *args, **kwargs: donor_xyz
    )
    actual = refine_prediction(
        np.zeros((3, 4, 5), dtype=np.uint8),
        None,
        None,
        {"tumor": [[4, 3, 2]], "background": []},
        "fdg",
        tmp_path,
        backend=InteractiveBackend(Path("source"), Path("weights"), Path("runner")),
        policy=InteractionPolicy(128, False, 0.5),
    )
    assert actual[2, 3, 4] == 1

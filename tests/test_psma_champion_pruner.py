import numpy as np

from candidate_runtime.psma_champion_pruner import prune_psma_components


def test_pruner_removes_only_components_above_false_probability_threshold():
    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[1, 1, 1] = True
    mask[5:7, 5:7, 5:7] = True
    probability = mask.astype(np.float32)
    pet = np.zeros_like(probability)

    pruned, audit = prune_psma_components(
        mask,
        probability,
        pet,
        spacing=(2.0, 2.0, 2.0),
        false_threshold=0.9,
        false_probability_override={1: 0.95, 2: 0.2},
    )

    assert not pruned[1, 1, 1]
    assert pruned[5:7, 5:7, 5:7].all()
    assert audit["removed_components"] == 1


def test_pruner_fails_closed_on_shape_mismatch():
    mask = np.zeros((4, 4, 4), dtype=bool)
    try:
        prune_psma_components(
            mask,
            np.zeros((3, 3, 3), dtype=np.float32),
            np.zeros_like(mask, dtype=np.float32),
            spacing=(1.0, 1.0, 1.0),
        )
    except ValueError as error:
        assert "shape mismatch" in str(error)
    else:
        raise AssertionError("shape mismatch must fail closed")

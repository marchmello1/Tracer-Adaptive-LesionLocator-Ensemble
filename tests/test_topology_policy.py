import numpy as np
import pytest

from candidate_runtime.edt_stateless_fusion import (
    FusionPolicy,
    PromptSet,
    TopologyLedger,
    fuse_clicked_components,
)


def test_prompt_set_discards_malformed_and_out_of_bounds_points():
    prompts = PromptSet.from_mapping(
        {
            "tumor": [[1, 2, 3], [-1, 2, 3], [2, 2], ["bad", 0, 0]],
            "background": [[3.9, 2, 1], [9, 9, 9]],
        },
        (5, 5, 5),
    )
    np.testing.assert_array_equal(prompts.tumor, [[1, 2, 3]])
    np.testing.assert_array_equal(prompts.background, [[3, 2, 1]])


def test_policy_rejects_unknown_tracer():
    with pytest.raises(ValueError, match="Unsupported tracer"):
        FusionPolicy.create(
            "unknown",
            disable_background_edits=False,
            certified_background_points=False,
            certified_tumor_points=False,
            psma_max_components=128,
        )


def test_policy_rejects_invalid_background_consensus_fraction():
    with pytest.raises(ValueError, match="retained fraction"):
        FusionPolicy.create(
            "fdg",
            disable_background_edits=False,
            certified_background_points=False,
            certified_tumor_points=False,
            psma_max_components=128,
            consensus_background_deletion=True,
            background_max_retained_fraction=1.01,
        )


def test_fusion_requires_three_dimensional_grids():
    with pytest.raises(ValueError, match="3-D mask"):
        fuse_clicked_components(
            np.zeros((3, 3), dtype=bool),
            np.zeros((3, 3), dtype=bool),
            {},
            "fdg",
        )


def test_ledger_accepts_a_disconnected_foreground_component():
    initial = np.zeros((9, 3, 3), dtype=bool)
    initial[1, 1, 1] = True
    addition = np.zeros_like(initial)
    addition[7, 1, 1] = True
    ledger = TopologyLedger(initial)

    assert ledger.add_if_nonmerging(addition)
    assert ledger.component_count == 2
    assert ledger.mask.sum() == 2


def test_ledger_rejects_an_addition_that_merges_components():
    initial = np.zeros((9, 3, 3), dtype=bool)
    initial[1, 1, 1] = True
    initial[7, 1, 1] = True
    bridge = np.zeros_like(initial)
    bridge[2:7, 1, 1] = True
    ledger = TopologyLedger(initial)

    assert not ledger.add_if_nonmerging(bridge)
    np.testing.assert_array_equal(ledger.mask, initial)


def test_ledger_accepts_removing_an_isolated_component():
    initial = np.zeros((9, 3, 3), dtype=bool)
    initial[1:3, 1, 1] = True
    initial[7, 1, 1] = True
    removal = np.zeros_like(initial)
    removal[7, 1, 1] = True
    ledger = TopologyLedger(initial)

    assert ledger.remove_if_nonfragmenting(removal)
    assert ledger.component_count == 1
    assert ledger.mask.sum() == 2


def test_ledger_rejects_removal_that_splits_a_component():
    initial = np.zeros((9, 3, 3), dtype=bool)
    initial[2:7, 1, 1] = True
    removal = np.zeros_like(initial)
    removal[4, 1, 1] = True
    ledger = TopologyLedger(initial)

    assert not ledger.remove_if_nonfragmenting(removal)
    np.testing.assert_array_equal(ledger.mask, initial)


def test_certified_foreground_application_is_idempotent():
    initial = np.zeros((9, 3, 3), dtype=bool)
    scribbles = {"tumor": [[7, 1, 1]], "background": []}
    once = fuse_clicked_components(
        initial,
        initial,
        scribbles,
        "fdg",
        certified_tumor_points=True,
    )
    twice = fuse_clicked_components(
        once,
        initial,
        scribbles,
        "fdg",
        certified_tumor_points=True,
    )
    np.testing.assert_array_equal(twice, once)


def test_background_consensus_deletes_only_donor_rejected_component():
    initial = np.zeros((9, 3, 3), dtype=bool)
    initial[1:4, 1, 1] = True
    initial[7, 1, 1] = True
    donor = initial.copy()
    donor[1:3, 1, 1] = False  # Retains one third of the prompted object.

    actual = fuse_clicked_components(
        initial,
        donor,
        {"tumor": [], "background": [[2, 1, 1]]},
        "psma",
        consensus_background_deletion=True,
        background_max_retained_fraction=0.5,
    )

    assert actual.sum() == 1
    assert actual[7, 1, 1]


def test_background_consensus_keeps_ambiguous_component_at_boundary():
    initial = np.zeros((8, 3, 3), dtype=bool)
    initial[1:5, 1, 1] = True
    donor = initial.copy()
    donor[1:3, 1, 1] = False  # Exactly half remains: not enough consensus.

    actual = fuse_clicked_components(
        initial,
        donor,
        {"tumor": [], "background": [[2, 1, 1]]},
        "psma",
        consensus_background_deletion=True,
        background_max_retained_fraction=0.5,
    )

    np.testing.assert_array_equal(actual, initial)


def test_foreground_prompt_protects_background_consensus_component():
    initial = np.zeros((8, 3, 3), dtype=bool)
    initial[1:5, 1, 1] = True
    donor = np.zeros_like(initial)

    actual = fuse_clicked_components(
        initial,
        donor,
        {"tumor": [[4, 1, 1]], "background": [[2, 1, 1]]},
        "fdg",
        disable_background_edits=True,
        consensus_background_deletion=True,
    )

    np.testing.assert_array_equal(actual, initial)

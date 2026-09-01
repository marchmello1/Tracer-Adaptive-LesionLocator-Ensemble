import numpy as np

from candidate_runtime.edt_stateless_fusion import fuse_clicked_components


def test_psma_ignores_small_cumulative_tumor_scribble():
    initial = np.zeros((9, 3, 3), dtype=bool)
    initial[1, 1, 1] = True
    donor = np.zeros_like(initial)
    donor[5:7, 1, 1] = True
    actual = fuse_clicked_components(
        initial,
        donor,
        {"tumor": [[5, 1, 1]] * 5, "background": []},
        "psma",
    )
    np.testing.assert_array_equal(actual, initial)


def test_psma_accepts_six_point_missed_component_without_background_edit():
    initial = np.zeros((9, 3, 3), dtype=bool)
    initial[1, 1, 1] = True
    donor = np.zeros_like(initial)
    donor[5:7, 1, 1] = True
    actual = fuse_clicked_components(
        initial,
        donor,
        {"tumor": [[5, 1, 1]] * 6, "background": [[1, 1, 1]]},
        "psma",
    )
    assert actual.sum() == 3


def test_psma_accepts_six_point_correction_with_many_initial_components():
    initial = np.zeros((60, 3, 3), dtype=bool)
    initial[:40:2, 1, 1] = True
    donor = np.zeros_like(initial)
    donor[50, 1, 1] = True

    actual = fuse_clicked_components(
        initial,
        donor,
        {"tumor": [[50, 1, 1]] * 6, "background": []},
        "psma",
    )

    assert actual[50, 1, 1]


def test_psma_extreme_burden_stays_conservative_above_ceiling():
    initial = np.zeros((300, 3, 3), dtype=bool)
    initial[:260:2, 1, 1] = True
    donor = np.zeros_like(initial)
    donor[290, 1, 1] = True

    actual = fuse_clicked_components(
        initial,
        donor,
        {"tumor": [[290, 1, 1]] * 6, "background": []},
        "psma",
    )

    np.testing.assert_array_equal(actual, initial)


def test_fdg_background_edit_removes_clicked_false_component():
    initial = np.zeros((9, 3, 3), dtype=bool)
    initial[1, 1, 1] = True
    initial[7, 1, 1] = True
    donor = initial.copy()
    donor[7, 1, 1] = False
    actual = fuse_clicked_components(
        initial,
        donor,
        {"tumor": [], "background": [[7, 1, 1]]},
        "fdg",
    )
    assert actual.sum() == 1


def test_fdg_background_edit_can_be_disabled_for_rank2_safety():
    initial = np.zeros((9, 3, 3), dtype=bool)
    initial[1, 1, 1] = True
    initial[7, 1, 1] = True
    donor = initial.copy()
    donor[7, 1, 1] = False

    actual = fuse_clicked_components(
        initial,
        donor,
        {"tumor": [], "background": [[7, 1, 1]]},
        "fdg",
        disable_background_edits=True,
    )

    np.testing.assert_array_equal(actual, initial)


def test_tumor_bridge_between_existing_components_is_rejected():
    initial = np.zeros((9, 3, 3), dtype=bool)
    initial[1, 1, 1] = True
    initial[7, 1, 1] = True
    donor = np.zeros_like(initial)
    donor[1:8, 1, 1] = True
    actual = fuse_clicked_components(
        initial,
        donor,
        {"tumor": [[4, 1, 1]], "background": []},
        "fdg",
    )
    np.testing.assert_array_equal(actual, initial)


def test_new_lesion_cannot_hide_a_separate_harmful_bridge():
    initial = np.zeros((11, 3, 3), dtype=bool)
    initial[1, 1, 1] = True
    initial[5, 1, 1] = True
    donor = np.zeros_like(initial)
    donor[1:6, 1, 1] = True  # Harmful bridge between the two K0 lesions.
    donor[8, 1, 1] = True    # Valid independent missed lesion.

    actual = fuse_clicked_components(
        initial,
        donor,
        {"tumor": [[3, 1, 1], [8, 1, 1]], "background": []},
        "fdg",
    )

    assert not actual[3, 1, 1]
    assert actual[8, 1, 1]
    assert actual.sum() == 3


def test_certified_background_stroke_removes_safe_false_positive_voxels():
    initial = np.zeros((9, 3, 3), dtype=bool)
    initial[1:4, 1, 1] = True
    initial[7, 1, 1] = True

    actual = fuse_clicked_components(
        initial,
        initial,
        {"tumor": [], "background": [[1, 1, 1], [7, 1, 1]]},
        "fdg",
        disable_background_edits=True,
        certified_background_points=True,
    )

    assert not actual[1, 1, 1]
    assert not actual[7, 1, 1]
    assert actual.sum() == 2


def test_certified_background_stroke_rejects_component_split():
    initial = np.zeros((9, 3, 3), dtype=bool)
    initial[1:6, 1, 1] = True

    actual = fuse_clicked_components(
        initial,
        initial,
        {"tumor": [], "background": [[3, 1, 1]]},
        "fdg",
        disable_background_edits=True,
        certified_background_points=True,
    )

    np.testing.assert_array_equal(actual, initial)

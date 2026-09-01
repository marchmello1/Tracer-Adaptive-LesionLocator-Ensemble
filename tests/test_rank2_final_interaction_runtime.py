from candidate_runtime.challenge_io import has_effective_scribbles


def test_empty_clicks_keep_exact_rank2_k0() -> None:
    assert not has_effective_scribbles({"tumor": [], "background": []})


def test_any_valid_click_activates_interaction() -> None:
    assert has_effective_scribbles({"tumor": [[1, 2, 3]], "background": []})
    assert has_effective_scribbles({"tumor": [], "background": [[1, 2, 3]]})


def test_malformed_clicks_do_not_activate_interaction() -> None:
    assert not has_effective_scribbles(
        {"tumor": [None, [1, 2], "bad"], "background": [{"x": 1}]}
    )

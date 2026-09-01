import numpy as np

from nnunetv2.training.dataloading.data_loader_clicks import (
    _sample_official_ft_click_budgets,
)


def test_official_ft_click_budget_distribution():
    point_sampling_probs = np.log(np.linspace(2, 12, 11))[::-1]
    point_sampling_probs /= point_sampling_probs.sum()
    rng = np.random.default_rng(20260618)

    counts = {"none": 0, "fg_only": 0, "bg_only": 0, "both": 0}
    n_samples = 50000
    for _ in range(n_samples):
        pos_clicks, neg_clicks = _sample_official_ft_click_budgets(
            point_sampling_probs, rng=rng
        )
        assert 0 <= pos_clicks <= 10
        assert 0 <= neg_clicks <= 10

        if pos_clicks == 0 and neg_clicks == 0:
            counts["none"] += 1
        elif pos_clicks > 0 and neg_clicks == 0:
            counts["fg_only"] += 1
        elif pos_clicks == 0 and neg_clicks > 0:
            counts["bg_only"] += 1
        else:
            counts["both"] += 1

    observed = {key: value / n_samples for key, value in counts.items()}
    expected = {"none": 0.35, "fg_only": 0.30, "bg_only": 0.20, "both": 0.15}
    for key, target in expected.items():
        assert abs(observed[key] - target) < 0.02, (key, observed)


if __name__ == "__main__":
    test_official_ft_click_budget_distribution()

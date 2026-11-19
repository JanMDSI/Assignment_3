import numpy as np
import pandas as pd

from src.analysis import filter_models
from src.association import association_summary, association_text, compare_associations, spearman, with_all_three
from src.data import PROCESSED_DIR, load_processed


def _frame(n: int = 60, noise_params: float = 0.0, noise_data: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    params = 10 ** rng.uniform(6, 11, n)
    data = 10 ** rng.uniform(8, 13, n)
    compute = params * data * 6
    return pd.DataFrame({
        "model_name": [f"m{i}" for i in range(n)], "release_year": 2020, "domain": "Language",
        "parameters": params * 10 ** rng.normal(0, noise_params, n),
        "training_dataset_size": data * 10 ** rng.normal(0, noise_data, n),
        "training_compute_flops": compute,
    })


def test_spearman_matches_known_values() -> None:
    assert abs(spearman(pd.Series([1, 2, 3, 4]), pd.Series([10, 20, 30, 40])) - 1) < 1e-12
    assert abs(spearman(pd.Series([1, 2, 3, 4]), pd.Series([4, 3, 2, 1])) + 1) < 1e-12
    assert abs(spearman(pd.Series([1, 2, 2, 4]), pd.Series([1, 2, 3, 4])) - 0.9486832980505138) < 1e-9


def test_the_cleaner_measure_wins_and_the_range_excludes_zero() -> None:
    result = compare_associations(_frame(200, noise_params=0.0, noise_data=2.0))
    assert result["status"] == "ok" and result["rho_params"] > result["rho_data"]
    assert result["diff_lo"] > 0
    assert "Compute moves more closely with parameters" in association_text(result)


def test_two_equally_noisy_measures_are_not_called_different() -> None:
    result = compare_associations(_frame(noise_params=0.8, noise_data=0.8))
    assert result["diff_lo"] <= 0 <= result["diff_hi"]
    assert "within sampling noise" in association_text(result)


def test_too_few_models() -> None:
    result = compare_associations(_frame(5))
    assert result["status"] == "too_few" and "Too few" in association_text(result)


def test_saved_associations_match_a_fresh_calculation() -> None:
    tables = load_processed(PROCESSED_DIR)
    fresh = association_summary(tables["models"])
    pd.testing.assert_frame_equal(fresh.reset_index(drop=True), tables["associations"].reset_index(drop=True), check_dtype=False)
    assert len(with_all_three(tables["models"])) == int(tables["associations"].loc[0, "n_models"])


def test_coverage_agrees_with_the_other_tables_and_filters() -> None:
    tables = load_processed(PROCESSED_DIR)
    cov = tables["coverage"]
    assert int((cov["has_parameters"] & cov["has_dataset_size"]).sum()) == len(tables["models"])
    assert int((cov["has_parameters"] & cov["has_compute"]).sum()) == len(tables["compute"])
    chosen = filter_models(cov, (2020, 2026), "gpt", ["Language"])
    assert 0 < len(chosen) < len(cov)
    assert chosen["model_name"].str.contains("gpt", case=False).all() and chosen["release_year"].between(2020, 2026).all()
    assert filter_models(cov, (2020, 2026), "zzzzzz", []).empty

import math

import numpy as np
import pandas as pd

from src.attribution import compute_scenarios, describe_split, estimate_split, headline, independent_compute
from src.data import PROCESSED_DIR, load_processed


def _synthetic(params_dex: float, data_dex: float, tokens: bool = False) -> pd.DataFrame:
    years = np.array(list(range(2015, 2025)) * 2, dtype=float)
    params = 10 ** (params_dex * (years - 2015) + 6)
    data = 10 ** (data_dex * (years - 2015) + 6)
    frame = pd.DataFrame({
        "model_name": [f"m{i}" for i in range(len(years))],
        "release_year": years.astype(int), "decimal_year": years + 0.5,
        "parameters": params, "training_dataset_size": data,
        "training_compute_flops": 6 * params * data,
    })
    return frame


def test_growth_split_is_three_quarters_model_size() -> None:
    for method in ("dataset_size", "tokens_processed"):
        result = estimate_split(_synthetic(0.3, 0.1), method)
        assert result["status"] == "ok"
        assert abs(result["params_share"] - 0.75) < 1e-9 and abs(result["data_share"] - 0.25) < 1e-9
        assert abs(result["params_factor"] - 10**0.3) < 1e-9
        assert abs(result["growth_factor"] - 10**0.4) < 1e-9


def test_split_refuses_when_one_factor_falls_or_data_is_thin() -> None:
    result = estimate_split(_synthetic(0.3, -0.1), "dataset_size")
    assert result["status"] == "not_both_rising" and math.isnan(result["params_share"])
    assert estimate_split(_synthetic(0.3, 0.1).head(5), "dataset_size")["status"] == "too_few"


def test_bootstrap_range_contains_the_estimate_and_is_repeatable() -> None:
    rng = np.random.default_rng(1)
    frame = _synthetic(0.3, 0.1)
    frame["parameters"] *= 10 ** rng.normal(0, 0.3, len(frame))
    first = estimate_split(frame, "dataset_size", bootstrap=True)
    again = estimate_split(frame, "dataset_size", bootstrap=True)
    assert first["share_lo"] <= first["params_share"] <= first["share_hi"]
    assert (first["share_lo"], first["share_hi"]) == (again["share_lo"], again["share_hi"])


def test_describe_split_branches() -> None:
    assert "Too few" in describe_split(estimate_split(_synthetic(0.3, 0.1).head(4), "dataset_size"), "dataset_size")
    assert "Bigger models explain 75%" in describe_split(estimate_split(_synthetic(0.3, 0.1), "dataset_size"), "dataset_size")
    assert "no split" in describe_split(estimate_split(_synthetic(0.3, -0.1), "dataset_size"), "dataset_size")


def _summary(rows) -> pd.DataFrame:
    return pd.DataFrame([{"status": "ok", "method": m, "params_share": s} for m, s in rows])


def test_headline_covers_each_branch() -> None:
    assert headline(_summary([("dataset_size", 0.3), ("tokens_processed", 0.4)])).startswith("More of the growth came from more training data")
    assert headline(_summary([("dataset_size", 0.7), ("tokens_processed", 0.8)])).startswith("More of the growth came from bigger models")
    assert headline(_summary([("dataset_size", 0.3), ("tokens_processed", 0.8)])).startswith("The answer depends on how more data is counted")
    assert headline(_summary([("dataset_size", 0.3), ("dataset_size", 0.7)])).startswith("Neither factor clearly wins")
    none = pd.DataFrame([{"status": "too_few", "method": "dataset_size", "params_share": np.nan}])
    assert "does not give a stable answer" in headline(none)


def test_saved_summary_equals_a_fresh_calculation_and_headline_uses_it() -> None:
    from src.attribution import answer_summary

    tables = load_processed(PROCESSED_DIR)
    fresh = answer_summary(tables["models"], tables["compute"], tables["frontier"])
    pd.testing.assert_frame_equal(fresh.reset_index(drop=True), tables["summary"].reset_index(drop=True), check_dtype=False)
    ok = tables["summary"][tables["summary"]["status"] == "ok"]
    assert ((ok["params_share"] + ok["data_share"] - 1).abs() < 1e-3).all()
    assert (ok["share_lo"] <= ok["params_share"] + 1e-9).all() and (ok["params_share"] <= ok["share_hi"] + 1e-9).all()
    assert set(ok["lens_id"]) == {"all_models", "compute", "frontier"}


def test_scenarios_and_independent_compute() -> None:
    compute = load_processed(PROCESSED_DIR)["compute"]
    scenarios = dict(compute_scenarios(compute, 2012, 2026))
    assert len(scenarios) == 4
    assert all(frame["release_year"].between(2012, 2026).all() for frame in scenarios.values())
    assert 0 < len(independent_compute(compute)) < len(compute)

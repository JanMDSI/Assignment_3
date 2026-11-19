import math

import numpy as np
import pandas as pd

from src.analysis import (
    answer_text, apply_filters, category_labels, compute_rule_check, format_count,
    growth_by_slice, growth_decomposition, loglog_fit, option_counts, selection_text,
    short_count, top_organizations, yearly_medians,
)
from src.data import PROCESSED_DIR, load_processed, parse_count, split_organizations


def test_parse_count_handles_epoch_formats() -> None:
    assert parse_count("45000000000000") == 4.5e13
    assert parse_count("1.5e9") == 1.5e9
    assert parse_count("5e9,5e9") == 5e9
    assert math.isnan(parse_count("2167000000000,36000000000000"))
    assert parse_count("36T") == 3.6e13
    assert parse_count("1,200,000") == 1_200_000
    assert math.isnan(parse_count(None)) and math.isnan(parse_count("n/a"))


def test_organizations_keep_legal_suffixes_together() -> None:
    assert split_organizations("Shanghai Xingzhi Technology Co., Ltd.") == ["Shanghai Xingzhi Technology Co., Ltd."]
    assert split_organizations("Stanford University,SambaNova Systems, Inc,Adobe") == [
        "Stanford University", "SambaNova Systems, Inc", "Adobe",
    ]


def test_formatting() -> None:
    assert format_count(1.5e9) == "1.5 billion"
    assert format_count(950) == "950"
    assert short_count(1.5e9) == "1.5B" and short_count(300e6) == "300M" and short_count(950) == "950"


def test_loglog_fit_recovers_a_power_law() -> None:
    data = pd.DataFrame({"training_dataset_size": [10.0**k for k in range(3, 9)]})
    data["parameters"] = 5 * data["training_dataset_size"] ** 0.7
    fit = loglog_fit(data)
    assert abs(fit["slope"] - 0.7) < 1e-9 and abs(fit["r2"] - 1) < 1e-9


def _synthetic(params_dex: float, data_dex: float) -> pd.DataFrame:
    years = list(range(2015, 2025)) * 2
    return pd.DataFrame({
        "model_name": [f"m{i}" for i in range(len(years))],
        "release_year": years,
        "parameters": [10 ** (params_dex * (y - 2015) + 6) for y in years],
        "training_dataset_size": [10 ** (data_dex * (y - 2015) + 6) for y in years],
    })


def test_growth_split_is_three_quarters_model_size() -> None:
    result = growth_decomposition(_synthetic(0.3, 0.1))
    assert result["status"] == "ok"
    assert abs(result["params_share"] - 0.75) < 1e-9
    assert abs(result["data_share"] - 0.25) < 1e-9
    assert abs(result["params_factor"] - 10**0.3) < 1e-9


def test_growth_split_refuses_when_one_factor_falls() -> None:
    result = growth_decomposition(_synthetic(0.3, -0.1))
    assert result["status"] == "not_both_rising" and math.isnan(result["params_share"])
    assert growth_decomposition(_synthetic(0.3, 0.1).head(5))["status"] == "too_few"


def test_processed_files_support_the_answer() -> None:
    models, slices, cleaning = load_processed(PROCESSED_DIR)
    assert len(models) == cleaning["models_remaining"].iloc[-1]
    # the saved slice table equals a fresh calculation from models.csv
    fresh = growth_by_slice(models)
    pd.testing.assert_frame_equal(fresh.reset_index(drop=True), slices.reset_index(drop=True), check_dtype=False)
    text = answer_text(slices)
    assert "training data" in text and "%" in text
    valid = slices[slices["status"] == "ok"]
    assert ((valid["params_share"] + valid["data_share"] - 1).abs() < 1e-3).all()


def test_answer_text_covers_each_branch() -> None:
    base = {"status": "ok", "params_share": 0.4, "data_share": 0.6}
    assert answer_text(pd.DataFrame([base, base])).startswith("Most of the growth came from training data")
    flipped = {"status": "ok", "params_share": 0.7, "data_share": 0.3}
    assert answer_text(pd.DataFrame([flipped])).startswith("Most of the growth came from larger models")
    assert answer_text(pd.DataFrame([base, flipped])).startswith("The answer depends")
    assert "does not give a stable answer" in answer_text(pd.DataFrame([{"status": "too_few", "params_share": np.nan, "data_share": np.nan}]))


def test_selection_text_branches() -> None:
    assert "Too few" in selection_text(growth_decomposition(_synthetic(0.3, 0.1).head(4)))
    assert "75% of the growth from model size" in selection_text(growth_decomposition(_synthetic(0.3, 0.1)))
    assert "no split" in selection_text(growth_decomposition(_synthetic(0.3, -0.1)))


def test_filters_charts_and_tables_use_the_same_rows() -> None:
    models, _, _ = load_processed(PROCESSED_DIR)
    language = apply_filters(models, (2010, 2026), [], ["Language"])
    assert 0 < len(language) < len(models)
    assert apply_filters(models, (2010, 2026), ["OpenAI"], ["Biology"]).empty
    yearly = yearly_medians(language)
    assert yearly["models"].sum() == len(language)
    top = top_organizations(language, 10)
    assert len(top) <= 10 and top["models"].is_monotonic_decreasing
    assert option_counts(models, "domain").index[0] == "Language"
    assert set(category_labels(models, "domain_group", keep=3)) <= set(models["domain_group"]) | {"Other"}


def test_compute_rule_check() -> None:
    frame = pd.DataFrame({
        "parameters": [1e6, 1e6], "training_dataset_size": [1e6, 1e6],
        "training_compute_flops": [6e12, 1e15],
    })
    assert compute_rule_check(frame) == {"n_models": 2, "share_equal": 0.5}
    assert compute_rule_check(frame.assign(training_compute_flops=np.nan)) is None

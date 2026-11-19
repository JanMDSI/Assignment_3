import math

import numpy as np
import pandas as pd

from src.analysis import (
    apply_filters, category_labels, compute_rule_check, format_count, implied_passes, loglog_fit, option_counts,
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


def test_filters_and_chart_ingredients_use_the_same_rows() -> None:
    models = load_processed(PROCESSED_DIR)["models"]
    language = apply_filters(models, (2010, 2026), [], ["Language"])
    assert 0 < len(language) < len(models)
    assert apply_filters(models, (2010, 2026), ["OpenAI"], ["Biology"]).empty
    assert yearly_medians(language)["models"].sum() == len(language)
    top = top_organizations(language, 10)
    assert len(top) <= 10 and top["models"].is_monotonic_decreasing
    assert option_counts(models, "domain").index[0] == "Language"
    assert set(category_labels(models, "domain_group", keep=3)) <= set(models["domain_group"]) | {"Other"}


def test_implied_passes_and_rule_check() -> None:
    frame = pd.DataFrame({
        "parameters": [1e6, 1e6], "training_dataset_size": [1e6, 1e6], "training_compute_flops": [6e12, 1.2e13],
    })
    assert list(implied_passes(frame)) == [1.0, 2.0]
    assert compute_rule_check(frame) == {"n_models": 2, "share_equal": 0.5}
    assert compute_rule_check(frame.assign(training_compute_flops=np.nan)) is None

"""The processed files must be rebuildable from the raw file and from the source sample."""

import pandas as pd

from src.data import OUTPUT_FILES, PROCESSED_DIR, RAW, SAMPLE, load_processed, make_sample, write_processed


def test_sample_runs_through_the_same_pipeline(tmp_path) -> None:
    summary = write_processed(SAMPLE, tmp_path)
    assert summary["raw_rows"] > 50 and 0 < summary["models"] < summary["raw_rows"] and summary["compute_models"] > 0
    tables = load_processed(tmp_path)
    assert (tables["models"]["parameters"] > 0).all() and (tables["models"]["training_dataset_size"] > 0).all()
    assert (tables["compute"]["training_compute_flops"] > 0).all()
    cleaning = tables["cleaning"]
    for _, group in cleaning.groupby("table"):
        assert group["models_remaining"].iloc[0] == summary["raw_rows"]
        assert group["models_remaining"].is_monotonic_decreasing


def test_pipeline_is_repeatable(tmp_path) -> None:
    write_processed(SAMPLE, tmp_path / "a")
    write_processed(SAMPLE, tmp_path / "b")
    for name in OUTPUT_FILES:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_committed_processed_files_match_a_fresh_run(tmp_path) -> None:
    write_processed(RAW, tmp_path)
    for name in OUTPUT_FILES:
        fresh = pd.read_csv(tmp_path / name, keep_default_na=False)
        saved = pd.read_csv(PROCESSED_DIR / name, keep_default_na=False)
        pd.testing.assert_frame_equal(fresh, saved)


def test_every_row_traces_back_to_the_raw_file() -> None:
    raw = pd.read_csv(RAW, low_memory=False)
    tables = load_processed(PROCESSED_DIR)
    for key in ("models", "compute"):
        table = tables[key].sample(40, random_state=0)
        for row in table.itertuples():
            source = raw.iloc[row.raw_row - 2]
            assert " ".join(str(source["Model"]).split()) == row.model_name
            assert float(source["Parameters"]) == row.parameters


def test_frontier_flags_follow_the_documented_rules() -> None:
    tables = load_processed(PROCESSED_DIR)
    frontier, models = tables["frontier"], tables["models"]
    assert frontier["frontier_model"].all() and (frontier["domain"] == "Language").all() and (frontier["release_year"] >= 2019).all()
    assert frontier["raw_row"].isin(models["raw_row"]).all()
    derived = frontier["base_model"].str.len() > 0
    assert (~frontier.loc[derived, "fresh_dense"]).all()
    assert "Switch" in set(frontier.loc[~frontier["fresh_dense"], "model_name"])
    assert (frontier.loc[~frontier["fresh_dense"], "exclusion_reason"].str.len() > 0).all()


def test_sample_is_every_thirtieth_raw_row(tmp_path) -> None:
    out = tmp_path / "sample.csv"
    count = make_sample(RAW, out)
    pd.testing.assert_frame_equal(pd.read_csv(out, low_memory=False), pd.read_csv(SAMPLE, low_memory=False))
    assert count == len(pd.read_csv(SAMPLE, low_memory=False))

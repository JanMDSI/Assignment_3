"""The processed files must be rebuildable from the raw file and from the source sample."""

import pandas as pd

from src.data import PROCESSED_DIR, RAW, SAMPLE, load_processed, make_sample, write_processed


def test_sample_runs_through_the_same_pipeline(tmp_path) -> None:
    summary = write_processed(SAMPLE, tmp_path)
    assert summary["raw_rows"] > 50 and 0 < summary["models"] < summary["raw_rows"]
    models, slices, cleaning = load_processed(tmp_path)
    assert (models["parameters"] > 0).all() and (models["training_dataset_size"] > 0).all()
    assert cleaning["models_remaining"].iloc[0] == summary["raw_rows"]
    assert cleaning["models_remaining"].is_monotonic_decreasing


def test_pipeline_is_repeatable(tmp_path) -> None:
    write_processed(SAMPLE, tmp_path / "a")
    write_processed(SAMPLE, tmp_path / "b")
    for name in ("models.csv", "growth_by_slice.csv", "cleaning_steps.csv"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_committed_processed_files_match_a_fresh_run(tmp_path) -> None:
    write_processed(RAW, tmp_path)
    for name in ("models.csv", "growth_by_slice.csv", "cleaning_steps.csv"):
        fresh = pd.read_csv(tmp_path / name, keep_default_na=False)
        saved = pd.read_csv(PROCESSED_DIR / name, keep_default_na=False)
        pd.testing.assert_frame_equal(fresh, saved)


def test_sample_is_every_thirtieth_raw_row(tmp_path) -> None:
    out = tmp_path / "sample.csv"
    count = make_sample(RAW, out)
    pd.testing.assert_frame_equal(pd.read_csv(out, low_memory=False), pd.read_csv(SAMPLE, low_memory=False))
    assert count == len(pd.read_csv(SAMPLE, low_memory=False))

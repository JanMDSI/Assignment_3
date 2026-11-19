"""Splitting the growth in training scale between bigger models and more data.

The idea in plain words. A common rule of thumb says training compute is about
6 x parameters x data. Take base 10 logs and it becomes a sum:

    log(compute) = log(6) + log(parameters) + log(data)

so the yearly growth of compute (the slope against time) is the growth of
parameters plus the growth of data, and each part divided by the total is its
share. "Data" can be counted two ways, and the app shows both:

    dataset_size      the size of the training dataset Epoch records
    tokens_processed  compute / (6 x parameters), the amount of data the model
                      actually read, repeats included

Share of growth from bigger models = parameter slope / (parameter slope + data slope).
It only makes sense when both slopes are positive. The 95% range comes from
resampling the models 1,000 times with replacement.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.analysis import has_domain


MIN_MODELS = 10
MIN_YEARS = 3
N_BOOT = 1000
SEED = 0
LENS2_START = 2012

METHOD_COLUMN = {"dataset_size": "training_dataset_size", "tokens_processed": "training_compute_flops"}
METHOD_PHRASE = {
    "dataset_size": "the size of the training dataset",
    "tokens_processed": "the tokens actually processed (compute ÷ 6 × parameters)",
}
METHOD_SHORT = {"dataset_size": "Dataset size", "tokens_processed": "Tokens processed"}

# Lens 1 slices. Each cut is a judgement call, which is the point: the answer is
# shown under several reasonable choices instead of one.
LENS1_SLICES = [
    {"id": "all_years", "label": "All models, every year", "domain": None, "start": None, "end": None},
    {"id": "all_2010", "label": "All domains, 2010 onward", "domain": None, "start": 2010, "end": None},
    {"id": "language_2010", "label": "Language, 2010 onward", "domain": "Language", "start": 2010, "end": None},
    {"id": "language_2010_2019", "label": "Language, 2010 to 2019", "domain": "Language", "start": 2010, "end": 2019},
    {"id": "language_2020", "label": "Language, 2020 onward", "domain": "Language", "start": 2020, "end": None},
    {"id": "vision_2010", "label": "Vision, 2010 onward", "domain": "Vision", "start": 2010, "end": None},
]

LENS_LABELS = {
    "all_models": "All models",
    "compute": "Compute based",
    "frontier": "Frontier language",
}


# ------------------------------------------------------------------ the maths


def _slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray | float:
    """Least squares slope of y on x. Works on 1 D arrays or row by row on 2 D arrays."""
    xc = x - x.mean(axis=-1, keepdims=True)
    return (xc * (y - y.mean(axis=-1, keepdims=True))).sum(axis=-1) / (xc**2).sum(axis=-1)


def _usable(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    column = METHOD_COLUMN[method]
    rows = frame.dropna(subset=["parameters", column, "decimal_year", "release_year"])
    return rows[(rows["parameters"] > 0) & (rows[column] > 0)]


def _logs(rows: pd.DataFrame, method: str) -> tuple[np.ndarray, np.ndarray]:
    log_params = np.log10(rows["parameters"].to_numpy(float))
    if method == "dataset_size":
        log_data = np.log10(rows["training_dataset_size"].to_numpy(float))
    else:
        log_data = np.log10(rows["training_compute_flops"].to_numpy(float)) - log_params - math.log10(6)
    return log_params, log_data


def _interval(x: np.ndarray, log_params: np.ndarray, log_data: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    """95% range for the parameter share. Resamples whole models, so pairs stay together."""
    n = len(x)
    idx = np.random.default_rng(seed).integers(0, n, (n_boot, n))
    xs = x[idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        slope_p = _slopes(xs, log_params[idx])
        slope_d = _slopes(xs, log_data[idx])
        total = slope_p + slope_d
        shares = np.where(total > 0, slope_p / total, np.nan)
    shares = shares[np.isfinite(shares)]
    if shares.size == 0:
        return float("nan"), float("nan")
    shares = np.clip(shares, 0, 1)  # a share cannot fall below 0% or pass 100%
    return float(np.percentile(shares, 2.5)), float(np.percentile(shares, 97.5))


def estimate_split(frame: pd.DataFrame, method: str, bootstrap: bool = False, min_models: int = MIN_MODELS) -> dict:
    """Yearly growth of parameters and data, and how growth splits between them.

    `status` is one of:
      ok               both rose, shares are valid
      not_both_rising  at least one did not rise, shares are left empty
      too_few          fewer than min_models models or MIN_YEARS release years
    """
    rows = _usable(frame, method)
    nan = float("nan")
    result = {
        "n_models": int(len(rows)),
        "first_year": int(rows["release_year"].min()) if len(rows) else None,
        "last_year": int(rows["release_year"].max()) if len(rows) else None,
        "params_factor": nan, "data_factor": nan, "growth_factor": nan,
        "params_share": nan, "data_share": nan, "share_lo": nan, "share_hi": nan,
        "status": "too_few",
    }
    if len(rows) < min_models or rows["release_year"].nunique() < MIN_YEARS:
        return result
    x = rows["decimal_year"].to_numpy(float)
    log_params, log_data = _logs(rows, method)
    slope_p, slope_d = float(_slopes(x, log_params)), float(_slopes(x, log_data))
    result["params_factor"] = 10**slope_p
    result["data_factor"] = 10**slope_d
    result["growth_factor"] = 10 ** (slope_p + slope_d)
    if slope_p > 0 and slope_d > 0:
        total = slope_p + slope_d
        result["params_share"] = slope_p / total
        result["data_share"] = slope_d / total
        result["status"] = "ok"
        if bootstrap:
            result["share_lo"], result["share_hi"] = _interval(x, log_params, log_data, N_BOOT, SEED)
    else:
        result["status"] = "not_both_rising"
    return result


# --------------------------------------------------------------- slices and scenarios


def slice_frame(models: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Rows of `models` that belong to one lens 1 slice."""
    mask = pd.Series(True, index=models.index)
    if spec["domain"]:
        mask &= has_domain(models, spec["domain"])
    if spec["start"] is not None:
        mask &= models["release_year"] >= spec["start"]
    if spec["end"] is not None:
        mask &= models["release_year"] <= spec["end"]
    return models.loc[mask]


def window(frame: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    return frame[frame["release_year"].between(start, end)]


def independent_compute(frame: pd.DataFrame) -> pd.DataFrame:
    """Models whose compute was reported by the developer or estimated from hardware.

    Their compute did not come from 6 x parameters x data, so they can test the
    rule instead of repeating it.
    """
    method = frame["compute_method"].fillna("").str.lower()
    return frame[method.str.contains("reported") | method.str.contains("hardware")]


def compute_scenarios(compute_models: pd.DataFrame, start: int, end: int) -> list[tuple[str, pd.DataFrame]]:
    """The same split under other sample choices, to show whether the answer depends on them."""
    return [
        (f"All models, {start} to {end}", window(compute_models, start, end)),
        (f"Developer reported or hardware compute only, {start} to {end}", window(independent_compute(compute_models), start, end)),
        (f"Language models only, {start} to {end}", window(compute_models[has_domain(compute_models, "Language")], start, end)),
        (f"Last five years only, {end - 4} to {end}", window(compute_models, end - 4, end)),
    ]


def _summary_row(lens_id: str, label: str, method: str, result: dict) -> dict:
    return {
        "lens_id": lens_id, "lens_label": LENS_LABELS[lens_id], "row_label": label, "method": method,
        "n_models": result["n_models"], "first_year": result["first_year"], "last_year": result["last_year"],
        "params_factor_per_year": round(result["params_factor"], 4),
        "data_factor_per_year": round(result["data_factor"], 4),
        "growth_factor_per_year": round(result["growth_factor"], 4),
        "params_share": round(result["params_share"], 4), "data_share": round(result["data_share"], 4),
        "share_lo": round(result["share_lo"], 4), "share_hi": round(result["share_hi"], 4),
        "status": result["status"],
    }


def answer_summary(models: pd.DataFrame, compute_models: pd.DataFrame, frontier: pd.DataFrame) -> pd.DataFrame:
    """One row per estimate across the three lenses. Saved as data/processed/answer_summary.csv."""
    rows = []
    for spec in LENS1_SLICES:
        rows.append(_summary_row("all_models", spec["label"], "dataset_size", estimate_split(slice_frame(models, spec), "dataset_size", bootstrap=True)))
    if not compute_models.empty:
        end = int(compute_models["release_year"].max())
        for label, frame in compute_scenarios(compute_models, LENS2_START, end):
            rows.append(_summary_row("compute", label, "tokens_processed", estimate_split(frame, "tokens_processed", bootstrap=True)))
    used = frontier[frontier["fresh_dense"]] if not frontier.empty else frontier
    for method in ("dataset_size", "tokens_processed"):
        rows.append(_summary_row("frontier", f"Frontier language models, {METHOD_SHORT[method].lower()}", method, estimate_split(used, method, bootstrap=True)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- sentences


def _side(low: float, high: float) -> str:
    if high < 0.5:
        return "data"
    if low > 0.5:
        return "models"
    return "mixed"


def headline(summary: pd.DataFrame) -> str:
    """One plain answer to the research question, built from answer_summary."""
    ok = summary[summary["status"] == "ok"]
    if ok.empty:
        return "The data does not give a stable answer: no estimate shows both model size and data rising."
    sides, parts = {}, []
    for method in ("dataset_size", "tokens_processed"):
        group = ok[ok["method"] == method]
        if group.empty:
            continue
        low, high = group["params_share"].min(), group["params_share"].max()
        sides[method] = _side(low, high)
        lean_data = int((group["params_share"] < 0.5).sum())
        parts.append(
            f"Counting data as {METHOD_PHRASE[method]}, bigger models explain {low:.0%} to {high:.0%} of the yearly growth "
            f"and more data {1 - high:.0%} to {1 - low:.0%} ({lean_data} of {len(group)} estimates lean toward more data)."
        )
    verdicts = set(sides.values())
    if verdicts == {"data"}:
        lead = "More of the growth came from more training data than from bigger models, whichever way data is counted."
    elif verdicts == {"models"}:
        lead = "More of the growth came from bigger models than from more training data, whichever way data is counted."
    elif len(sides) > 1 and len(verdicts) > 1:
        lead = "The answer depends on how more data is counted."
    else:
        lead = "Neither factor clearly wins: the estimates straddle an even split."
    return f"{lead} " + " ".join(parts)


def describe_split(result: dict, method: str) -> str:
    """One sentence for one estimate, used in the explorers."""
    span = f"{result['first_year']} to {result['last_year']}"
    phrase = METHOD_PHRASE[method]
    if result["status"] == "too_few":
        return f"Too few models or release years for a growth estimate (needs {MIN_MODELS} models across {MIN_YEARS} different years)."
    base = (
        f"For {result['n_models']:,} models ({span}), parameters grew {result['params_factor']:.2f} times per year "
        f"and {phrase} {result['data_factor']:.2f} times per year"
    )
    if result["status"] == "not_both_rising":
        return base + ". One of the two did not rise, so no split between them is given."
    text = base + f". Bigger models explain {result['params_share']:.0%} of the growth and more data {result['data_share']:.0%}"
    if not math.isnan(result["share_lo"]):
        text += f" (95% range for the bigger models share: {result['share_lo']:.0%} to {result['share_hi']:.0%})"
    return text + "."

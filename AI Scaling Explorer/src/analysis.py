"""Plain pandas calculations behind every number on the page.

Nothing in this file touches Streamlit or Plotly, so the same functions feed
the data pipeline (which saves the results to data/processed/) and the app
(which recalculates them live for whatever the user has filtered).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


ORG_SEP = "; "
DOMAIN_SEP = ", "

# A growth estimate needs enough models and enough different years to be a trend.
MIN_MODELS = 10
MIN_YEARS = 3

# Slices used for the "does the answer change?" chart. Each one is a judgement
# call, which is the point: the answer is shown under several reasonable choices.
SLICES = [
    {"id": "all_years", "label": "All models, every year", "domain": None, "start": None, "end": None},
    {"id": "all_2010", "label": "All domains, 2010 onward", "domain": None, "start": 2010, "end": None},
    {"id": "language_2010", "label": "Language, 2010 onward", "domain": "Language", "start": 2010, "end": None},
    {"id": "language_2010_2019", "label": "Language, 2010 to 2019", "domain": "Language", "start": 2010, "end": 2019},
    {"id": "language_2020", "label": "Language, 2020 onward", "domain": "Language", "start": 2020, "end": None},
    {"id": "vision_2010", "label": "Vision, 2010 onward", "domain": "Vision", "start": 2010, "end": None},
]


# ---------------------------------------------------------------- formatting


def format_count(value: float) -> str:
    """Readable size, for example 1.5e9 becomes '1.5 billion'."""
    if value is None or pd.isna(value):
        return "n/a"
    for threshold, name in ((1e15, "quadrillion"), (1e12, "trillion"), (1e9, "billion"), (1e6, "million"), (1e3, "thousand")):
        if abs(value) >= threshold:
            return f"{round(value / threshold, 1):,g} {name}"
    return f"{value:,.0f}"


def short_count(value: float) -> str:
    """Compact label for a chart point, for example 1.5e9 becomes '1.5B'."""
    if value is None or pd.isna(value):
        return ""
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= threshold:
            return f"{value / threshold:.3g}{suffix}"
    return f"{value:,.0f}"


def date_label(value: object) -> str:
    """'3 Nov 2026' style date, or 'unknown'."""
    stamp = pd.to_datetime(value, errors="coerce", utc=True)
    return "unknown" if pd.isna(stamp) else f"{stamp.day} {stamp.strftime('%b %Y')}"


# ------------------------------------------------------------------ filtering


def has_domain(frame: pd.DataFrame, name: str) -> pd.Series:
    """True where the model lists `name` among its domains."""
    return frame["domain"].map(lambda text: name in text.split(DOMAIN_SEP))


def option_counts(frame: pd.DataFrame, column: str) -> pd.Series:
    """Individual organisations or domains with their model counts, most common first."""
    sep = ORG_SEP if column == "organization" else DOMAIN_SEP
    return frame[column].map(lambda text: [t for t in text.split(sep) if t]).explode().value_counts()


def apply_filters(
    frame: pd.DataFrame,
    year_range: tuple[int, int],
    organizations: list[str],
    domains: list[str],
) -> pd.DataFrame:
    """Rows in the year range that match any chosen organisation and any chosen domain.

    An empty organisation or domain list means no restriction.
    """
    mask = frame["release_year"].between(year_range[0], year_range[1])
    if organizations:
        wanted = set(organizations)
        mask &= frame["organization"].map(lambda text: bool(wanted & set(text.split(ORG_SEP))))
    if domains:
        wanted = set(domains)
        mask &= frame["domain"].map(lambda text: bool(wanted & set(text.split(DOMAIN_SEP))))
    return frame.loc[mask].reset_index(drop=True)


# ------------------------------------------------------------------- statistics


def ols(x: pd.Series, y: pd.Series) -> dict | None:
    """Least squares line y = intercept + slope * x, with R squared. None if undefined."""
    if len(x) < 3 or x.nunique() < 2:
        return None
    x_mean, y_mean = x.mean(), y.mean()
    sxx = ((x - x_mean) ** 2).sum()
    sxy = ((x - x_mean) * (y - y_mean)).sum()
    syy = ((y - y_mean) ** 2).sum()
    slope = float(sxy / sxx)
    return {
        "slope": slope,
        "intercept": float(y_mean - slope * x_mean),
        "r2": float(sxy**2 / (sxx * syy)) if syy > 0 else float("nan"),
    }


def loglog_fit(frame: pd.DataFrame) -> dict | None:
    """Fit log10(parameters) against log10(dataset size)."""
    return ols(np.log10(frame["training_dataset_size"]), np.log10(frame["parameters"]))


def growth_decomposition(frame: pd.DataFrame) -> dict:
    """Yearly growth of parameters and dataset size, and how the growth splits between them.

    A common rule of thumb says training compute is about 6 x parameters x
    training data. Taking logs, log(compute) = log(6) + log(parameters) + log(data),
    so the yearly slope of log10(parameters) plus the yearly slope of
    log10(data) is the yearly growth of that product. Each slope divided by the
    sum is that factor's share. The shares only make sense when both rise.

    `status` is one of:
      ok               both rose, shares are valid
      not_both_rising  at least one factor did not rise, shares are left empty
      too_few          fewer than MIN_MODELS models or MIN_YEARS release years
    """
    result: dict = {
        "n_models": int(len(frame)),
        "first_year": int(frame["release_year"].min()) if len(frame) else None,
        "last_year": int(frame["release_year"].max()) if len(frame) else None,
        "params_factor": float("nan"),
        "data_factor": float("nan"),
        "params_share": float("nan"),
        "data_share": float("nan"),
        "status": "too_few",
    }
    if len(frame) < MIN_MODELS or frame["release_year"].nunique() < MIN_YEARS:
        return result
    years = frame["release_year"].astype(float)
    params = ols(years, np.log10(frame["parameters"]))
    data = ols(years, np.log10(frame["training_dataset_size"]))
    if params is None or data is None:
        return result
    result["params_factor"] = float(10 ** params["slope"])
    result["data_factor"] = float(10 ** data["slope"])
    if params["slope"] > 0 and data["slope"] > 0:
        total = params["slope"] + data["slope"]
        result["params_share"] = float(params["slope"] / total)
        result["data_share"] = float(data["slope"] / total)
        result["status"] = "ok"
    else:
        result["status"] = "not_both_rising"
    return result


def slice_frame(models: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Rows of `models` that belong to one slice definition."""
    mask = pd.Series(True, index=models.index)
    if spec["domain"]:
        mask &= has_domain(models, spec["domain"])
    if spec["start"] is not None:
        mask &= models["release_year"] >= spec["start"]
    if spec["end"] is not None:
        mask &= models["release_year"] <= spec["end"]
    return models.loc[mask]


def growth_by_slice(models: pd.DataFrame) -> pd.DataFrame:
    """One row per slice with its growth numbers. Saved as data/processed/growth_by_slice.csv."""
    rows = []
    for spec in SLICES:
        result = growth_decomposition(slice_frame(models, spec))
        rows.append(
            {
                "slice_id": spec["id"],
                "slice_label": spec["label"],
                "n_models": result["n_models"],
                "first_year": result["first_year"],
                "last_year": result["last_year"],
                "params_factor_per_year": round(result["params_factor"], 4),
                "data_factor_per_year": round(result["data_factor"], 4),
                "params_share": round(result["params_share"], 4),
                "data_share": round(result["data_share"], 4),
                "status": result["status"],
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------ chart ingredients


def yearly_medians(frame: pd.DataFrame) -> pd.DataFrame:
    """Models per release year with the median parameters and median dataset size."""
    grouped = frame.groupby("release_year")
    out = grouped.agg(
        models=("model_name", "count"),
        median_parameters=("parameters", "median"),
        median_dataset_size=("training_dataset_size", "median"),
    )
    return out.reset_index()


def top_organizations(frame: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Organisations with the most models. A joint model counts once for each organisation."""
    counts = option_counts(frame, "organization").drop(labels=["Not specified"], errors="ignore")
    top = counts.head(n).rename_axis("organization").reset_index(name="models")
    return top


def category_labels(frame: pd.DataFrame, column: str, keep: int = 6) -> pd.Series:
    """Keep the `keep` most common categories and fold the rest into 'Other'."""
    keep_values = list(frame[column].value_counts().index[:keep])
    return frame[column].where(frame[column].isin(keep_values), "Other")


def orders_of_magnitude(series: pd.Series) -> float:
    """How many powers of ten separate the smallest and largest value."""
    return float(math.log10(series.max()) - math.log10(series.min()))


def compute_rule_check(frame: pd.DataFrame) -> dict | None:
    """How often Epoch's reported compute equals 6 x parameters x dataset size (within 1 percent).

    A high share means the compute column was probably built from the same
    rule, so it cannot be used as an independent test of the rule.
    """
    have = frame.dropna(subset=["training_compute_flops"])
    have = have[have["training_compute_flops"] > 0]
    if have.empty:
        return None
    ratio = have["training_compute_flops"] / (6 * have["parameters"] * have["training_dataset_size"])
    return {"n_models": int(len(have)), "share_equal": float(((ratio - 1).abs() <= 0.01).mean())}


# ------------------------------------------------------------------- sentences


def answer_text(slices: pd.DataFrame) -> str:
    """One plain sentence answering the research question, built from growth_by_slice."""
    valid = slices[slices["status"] == "ok"]
    if valid.empty:
        return "The data does not give a stable answer: no slice shows both model size and training data rising."
    low_data, high_data = valid["data_share"].min(), valid["data_share"].max()
    low_size, high_size = valid["params_share"].min(), valid["params_share"].max()
    count = len(valid)
    skipped = len(slices) - count
    note = "" if skipped == 0 else f" ({skipped} more slice{'s' if skipped != 1 else ''} had no rise in one of the two, so no split is shown)"
    if high_size < 0.5:
        lead = "Most of the growth came from training data, not from larger models."
    elif low_size > 0.5:
        lead = "Most of the growth came from larger models, not from training data."
    else:
        lead = "The answer depends on the slice: neither factor wins in every case."
    return (
        f"{lead} Across {count} slices of the data, training data accounts for {low_data:.0%} to "
        f"{high_data:.0%} of the yearly growth in parameters times data, and model size for "
        f"{low_size:.0%} to {high_size:.0%}{note}."
    )


def selection_text(result: dict) -> str:
    """Plain sentence for the models the user currently has selected."""
    span = f"{result['first_year']} to {result['last_year']}"
    if result["status"] == "too_few":
        return (
            f"Too few models or release years in this selection for a growth estimate "
            f"(needs {MIN_MODELS} models across {MIN_YEARS} different years)."
        )
    base = (
        f"For the {result['n_models']:,} models selected ({span}), parameters grew "
        f"{result['params_factor']:.2f} times per year and training data {result['data_factor']:.2f} times per year"
    )
    if result["status"] == "not_both_rising":
        return base + ". One of the two did not rise, so no split between them is given."
    return (
        base + f". That is {result['params_share']:.0%} of the growth from model size and "
        f"{result['data_share']:.0%} from training data."
    )

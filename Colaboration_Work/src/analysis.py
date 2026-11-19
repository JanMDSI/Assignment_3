"""Plain pandas helpers shared by every part of the app.

Nothing here touches Streamlit or Plotly, so the same functions feed the data
pipeline (which saves results to data/processed/) and the app (which
recalculates them live for whatever the user has filtered). The growth split
itself lives in src/attribution.py.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


ORG_SEP = "; "
DOMAIN_SEP = ", "


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


def filter_models(frame: pd.DataFrame, year_range: tuple[int, int], search: str, domains: list[str]) -> pd.DataFrame:
    """Rows in the year range whose name contains `search` (any case) and that match any chosen domain.

    An empty search or domain list means no restriction.
    """
    mask = frame["release_year"].between(year_range[0], year_range[1])
    if search.strip():
        mask &= frame["model_name"].str.contains(search.strip(), case=False, regex=False)
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
    return counts.head(n).rename_axis("organization").reset_index(name="models")


def category_labels(frame: pd.DataFrame, column: str, keep: int = 6) -> pd.Series:
    """Keep the `keep` most common categories and fold the rest into 'Other'."""
    keep_values = list(frame[column].value_counts().index[:keep])
    return frame[column].where(frame[column].isin(keep_values), "Other")


def orders_of_magnitude(series: pd.Series) -> float:
    """How many powers of ten separate the smallest and largest value."""
    return float(math.log10(series.max()) - math.log10(series.min()))


def implied_passes(frame: pd.DataFrame) -> pd.Series:
    """Training compute divided by 6 x parameters x dataset size, row by row.

    If compute really is about 6 x parameters x tokens processed, this ratio is
    how many times the model read through its dataset. It is an implied figure,
    not a recorded one, and it is NaN where compute is missing.
    """
    compute = frame["training_compute_flops"]
    return compute / (6 * frame["parameters"] * frame["training_dataset_size"])


def compute_rule_check(frame: pd.DataFrame) -> dict | None:
    """How often Epoch's reported compute equals 6 x parameters x dataset size (within 1 percent).

    A high share means the compute column was probably built from the same
    rule, so it cannot be used as an independent test of the rule.
    """
    have = frame.dropna(subset=["training_compute_flops"])
    have = have[have["training_compute_flops"] > 0]
    if have.empty:
        return None
    ratio = implied_passes(have)
    return {"n_models": int(len(have)), "share_equal": float(((ratio - 1).abs() <= 0.01).mean())}

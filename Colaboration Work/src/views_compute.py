"""Lens 2: models with recorded training compute. Data is counted as tokens processed."""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from src.attribution import LENS2_START, MIN_MODELS, compute_scenarios, estimate_split, window
from src.views_common import first_url, render_interval

FIRST_SLIDER_YEAR = 2010
DEFAULT_SPLIT = 2020


def _rows(items: list[tuple[str, pd.DataFrame]]) -> tuple[pd.DataFrame, list[str]]:
    """Estimate each (label, frame) and return chart rows plus labels that could not be drawn."""
    rows, dropped = [], []
    for label, frame in items:
        result = estimate_split(frame, "tokens_processed", bootstrap=True)
        if result["status"] == "ok":
            rows.append({"label": label, "n_models": result["n_models"], "method": "tokens_processed",
                         "params_share": result["params_share"], "share_lo": result["share_lo"], "share_hi": result["share_hi"]})
        else:
            dropped.append(label)
    return pd.DataFrame(rows, columns=["label", "n_models", "method", "params_share", "share_lo", "share_hi"]), dropped


def render_compute(compute: pd.DataFrame) -> None:
    st.markdown(
        "**Lens 2 counts data as tokens processed.** It does not use Epoch's dataset size at all. For every model that records "
        "compute and parameters, the data it processed is worked out as compute ÷ (6 × parameters). That counts repeated passes "
        "over the same data, which the dataset size does not."
    )
    first, last = int(compute["release_year"].min()), int(compute["release_year"].max())
    low = max(first, FIRST_SLIDER_YEAR)
    c1, c2 = st.columns(2)
    start, end = c1.slider("Years", low, last, (max(low, LENS2_START), last), key="l2_years")
    split = c2.number_input("Compare before and after this year", low + 1, last, DEFAULT_SPLIT, key="l2_split")
    if end - start < 2:
        st.warning("Choose at least three years apart, for example by widening the slider.")
        return
    if split - start < 2 or end - split < 2:
        st.warning("The comparison year must leave at least two years on each side. Change it or the slider.")
        return

    selected = window(compute, start, end)
    whole = estimate_split(selected, "tokens_processed", bootstrap=True)
    if whole["status"] == "too_few":
        st.warning(f"Too few models in {start} to {end} (needs {MIN_MODELS} models across at least three years). Widen the years.")
        return
    if whole["status"] != "ok":
        st.info("In this window compute, or one of its two parts, did not rise, so no split is given. Try other years.")
        return

    st.subheader(
        f"{start} to {end}: about {whole['params_share']:.0%} of compute growth came from larger models "
        f"and {whole['data_share']:.0%} from more training data."
    )
    st.write(
        f"Compute grew about {whole['growth_factor']:.1f} times a year, parameters {whole['params_factor']:.1f} times and tokens "
        f"processed {whole['data_factor']:.1f} times. 95% range for the larger models share: {whole['share_lo']:.0%} to "
        f"{whole['share_hi']:.0%} ({whole['n_models']:,} models, release years {whole['first_year']} to {whole['last_year']})."
    )

    periods, dropped = _rows([
        (f"{start} to {split - 1}", window(compute, start, split - 1)),
        (f"{split} to {end}", window(compute, split, end)),
        (f"{start} to {end}", selected),
    ])
    st.markdown("##### Did the split change over time?")
    if periods.empty:
        st.info("No period has enough models for a split.")
    else:
        render_interval(
            periods, "Share of compute growth from bigger models, before and after your comparison year",
            "Dot is the estimate and the line is the 95% range. The dashed line is an even split. "
            + ("" if not dropped else "Left out because of too few models or no rise: " + ", ".join(dropped) + "."),
        )

    scenarios, dropped = _rows(compute_scenarios(compute, start, end))
    st.markdown("##### Does the answer change with other choices?")
    if scenarios.empty:
        st.info("No scenario has enough models.")
    else:
        render_interval(
            scenarios, "Share of compute growth from bigger models under four sample choices",
            "If the dots sit close together the answer is robust to these choices. 'Developer reported or hardware compute' keeps only "
            "models whose compute was not itself derived from parameters and data. "
            + ("" if not dropped else "Left out: " + ", ".join(dropped) + "."),
        )
    _trace(selected, whole)


def _trace(selected: pd.DataFrame, whole: dict) -> None:
    st.markdown("##### Trace a number")
    slope_c = math.log10(whole["growth_factor"])
    slope_p = math.log10(whole["params_factor"])
    slope_t = math.log10(whole["data_factor"])
    st.write(
        f"Growth in orders of magnitude a year: compute **{slope_c:.3f}** = parameters **{slope_p:.3f}** + tokens processed "
        f"**{slope_t:.3f}**. Larger models share = {slope_p:.3f} ÷ {slope_c:.3f} = **{whole['params_share']:.1%}**."
    )
    ordered = selected.sort_values("training_compute_flops", ascending=False)
    names = dict(zip(ordered["raw_row"], ordered["model_name"]))
    pick = st.selectbox("Pick a model", list(names), format_func=lambda row: f"row {int(row)}: {names[row]}", key="l2_pick")
    model = ordered[ordered["raw_row"] == pick].iloc[0]
    st.write(
        f"Tokens processed = compute ÷ (6 × parameters) = {model['training_compute_flops']:.3e} ÷ (6 × {model['parameters']:.3e}) "
        f"= **{model['tokens_processed']:.3e}**"
    )
    link = first_url(model["source_link"])
    st.caption(
        f"Same model in the raw file: spreadsheet row {int(model['raw_row'])}, published {model['publication_date']}, domain "
        f"{model['domain']}, compute method: {model['compute_method']}." + (f" Source: {link}" if link else "")
    )
    st.caption(
        "Limits of this lens: compute ÷ (6 × parameters) is only a rule of thumb. Epoch estimates some compute figures, and "
        "only models that record both compute and parameters are used, so the biggest undisclosed models are missing. "
        "Compute measures cost, not capability."
    )

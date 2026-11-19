"""Lens 3: frontier language models since 2019, with both ways of counting data side by side."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analysis import implied_passes, short_count, yearly_medians
from src.attribution import describe_split, estimate_split
from src.views_common import (
    BLACK, BLUE, DATA_LABEL, EMPTY_TEXT, ORANGE, PARAM_LABEL, clean_layout, first_url, line_chart,
    plot, render_change, render_download, render_table, span, stacked_share_chart,
)

SORT_OPTIONS = {
    "Release year (newest first)": ("release_year", False),
    "Parameters (largest first)": ("parameters", False),
    "Training dataset size (largest first)": ("training_dataset_size", False),
}
TABLE_COLUMNS = ["model_name", "release_year", "parameters", "training_dataset_size", "training_compute_flops", "confidence", "used", "source_link"]
TABLE_CONFIG = {
    "model_name": "Model",
    "release_year": st.column_config.NumberColumn("Year", format="%d"),
    "parameters": st.column_config.NumberColumn(PARAM_LABEL, format="%.0f"),
    "training_dataset_size": st.column_config.NumberColumn(DATA_LABEL, format="%.0f"),
    "training_compute_flops": st.column_config.NumberColumn("Training compute (FLOP)", format="%.3g"),
    "confidence": "Source confidence",
    "used": "Used in answer",
    "source_link": st.column_config.LinkColumn("Source link", display_text="open"),
}


def render_frontier(frontier: pd.DataFrame, snapshot: str) -> None:
    st.markdown(
        "**Lens 3 looks at frontier language models only**: those Epoch ranks among the ten most compute heavy when released, "
        "released 2019 or later. It is a small, well documented set, and it shows both ways of counting data side by side."
    )
    if frontier.empty:
        st.warning("No frontier language models are in this data file.")
        return
    year_min, year_max = int(frontier["release_year"].min()), int(frontier["release_year"].max())
    confidence_options = sorted(frontier["confidence"].unique())
    flagged_count = int((~frontier["fresh_dense"]).sum())
    c1, c2, c3 = st.columns([3, 3, 4])
    if year_min < year_max:
        year_range = c1.slider("Release year", year_min, year_max, (year_min, year_max), key="l3_years")
    else:
        year_range = (year_min, year_max)
    confidence = c2.multiselect("Source confidence", confidence_options, default=confidence_options, key="l3_confidence",
                                help="Epoch's own rating of how certain the recorded values are.")
    exclude = c3.checkbox(
        f"Exclude fine tuned and mixture of experts models ({flagged_count})", value=True, key="l3_exclude",
        help="Their recorded training data is not a full training run of a dense model. The reason for each is in the model panel below.",
    )

    filtered = frontier[frontier["release_year"].between(*year_range) & frontier["confidence"].isin(confidence)].copy()
    if filtered.empty:
        st.warning(EMPTY_TEXT)
        return
    answer = filtered[filtered["fresh_dense"]] if exclude else filtered
    filtered["used"] = filtered["fresh_dense"].map(lambda v: "Yes" if (v or not exclude) else "No")

    st.markdown("##### The answer for the models selected")
    by_size = estimate_split(answer, "dataset_size", bootstrap=True)
    by_tokens = estimate_split(answer, "tokens_processed", bootstrap=True)
    st.info(describe_split(by_size, "dataset_size"))
    st.info(describe_split(by_tokens, "tokens_processed"))
    if by_size["status"] == "ok" and by_tokens["status"] == "ok" and abs(by_size["params_share"] - by_tokens["params_share"]) > 0.1:
        st.markdown(
            f"**The answer depends on how more data is counted.** Dataset size puts {by_size['params_share']:.0%} of the growth on bigger "
            f"models, tokens processed puts {by_tokens['params_share']:.0%}."
        )
    _share_bars(by_size, by_tokens)
    _metrics(filtered, answer)

    if len(answer) >= 2 and answer["release_year"].nunique() >= 2:
        st.markdown("##### How model size and training data changed over time")
        render_change(answer, yearly_medians(answer), snapshot, "l3")
        _passes(answer)

    st.markdown("##### Explore and compare models")
    names = dict(zip(filtered["raw_row"], filtered["model_name"]))
    pick = st.selectbox("Select a model to highlight on the chart", list(names), format_func=lambda row: names[row], key="l3_pick")
    _scatter(filtered, int(pick), exclude)
    _verify(filtered[filtered["raw_row"] == pick].iloc[0])
    render_table(filtered, "l3", TABLE_COLUMNS, TABLE_CONFIG, SORT_OPTIONS)
    render_download(filtered.drop(columns=["used"]), "Download the selected frontier models (CSV)", "selected_frontier_models.csv", "l3_download")


def _share_bars(by_size: dict, by_tokens: dict) -> None:
    rows = []
    for result, label in ((by_size, "Data counted as dataset size"), (by_tokens, "Data counted as tokens processed")):
        if result["status"] == "ok":
            rows.append((f"{label} ({result['n_models']:,} models)", result["params_share"], result["data_share"]))
    if not rows:
        st.info("In this selection parameters or data did not rise, or there are too few models, so no split is drawn.")
        return
    st.markdown("**Where the growth in training compute came from, by way of counting data**")
    plot(stacked_share_chart(rows, "More data"))
    st.caption(
        "Shares of the yearly growth in parameters times data, on a log scale. The tokens processed bar uses Epoch's recorded compute and "
        "treats compute ÷ (6 × parameters) as the data. The dataset size bar uses the size of the dataset instead."
    )


def _metrics(filtered: pd.DataFrame, answer: pd.DataFrame) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Models displayed", f"{len(filtered):,}")
    c2.metric("Used in the answer", f"{len(answer):,}")
    c3.metric("Release years", span(filtered))


def _passes(answer: pd.DataFrame) -> None:
    """Median implied passes over the dataset by year, which explains why the two measures can disagree."""
    data = answer.assign(passes=implied_passes(answer)).dropna(subset=["passes"])
    if data["release_year"].nunique() < 2:
        return
    yearly = data.groupby("release_year").agg(models=("model_name", "count"), median_passes=("passes", "median")).reset_index()
    st.markdown("**Median implied passes over the training data, by release year**")
    plot(line_chart(yearly, "median_passes", ORANGE, "Implied passes over the data (times, log scale)"))
    first, last = yearly.iloc[0], yearly.iloc[-1]
    st.caption(
        f"Implied passes = compute ÷ (6 × parameters × dataset size). It is {first['median_passes']:.1f} for the {int(first['models'])} model(s) "
        f"released in {int(first['release_year'])} and {last['median_passes']:.1f} for the {int(last['models'])} released in {int(last['release_year'])}. "
        "It is implied by the rule of thumb, not recorded: where it is far from 1, the two ways of counting data disagree."
    )


def _scatter(filtered: pd.DataFrame, pick: int, exclude: bool) -> None:
    st.markdown("**Parameters against training dataset size, selected model highlighted**")
    chosen = filtered[filtered["raw_row"] == pick]
    others = filtered[filtered["raw_row"] != pick]
    used = others[others["fresh_dense"]] if exclude else others
    flagged = others[~others["fresh_dense"]] if exclude else others.iloc[0:0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=used["training_dataset_size"], y=used["parameters"], mode="markers", name="Models used in the answer",
        marker={"color": BLUE, "size": 10, "symbol": "circle", "opacity": 0.75}, text=used["model_name"],
        hovertemplate="%{text}<br>%{x:,.0f} datapoints<br>%{y:,.0f} parameters<extra></extra>",
    ))
    if not flagged.empty:
        fig.add_trace(go.Scatter(
            x=flagged["training_dataset_size"], y=flagged["parameters"], mode="markers", name="Fine tuned or mixture of experts (excluded)",
            marker={"color": BLACK, "size": 10, "symbol": "x"}, text=flagged["model_name"],
            hovertemplate="%{text}<br>%{x:,.0f} datapoints<br>%{y:,.0f} parameters<extra></extra>",
        ))
    row = chosen.iloc[0]
    fig.add_trace(go.Scatter(
        x=[row["training_dataset_size"]], y=[row["parameters"]], mode="markers+text", name=f"Selected: {row['model_name']}",
        marker={"color": ORANGE, "size": 20, "symbol": "star", "line": {"color": "black", "width": 1.5}},
        text=[f"{row['model_name']}: {short_count(row['parameters'])} parameters, {short_count(row['training_dataset_size'])} datapoints"],
        textposition="top center", cliponaxis=False,
    ))
    fig.update_xaxes(type="log", title="Training dataset size (datapoints, log scale)", exponentformat="power")
    fig.update_yaxes(type="log", title="Parameters (count, log scale)", exponentformat="power")
    fig.update_layout(legend={"orientation": "h", "y": -0.2})
    plot(clean_layout(fig, 480))
    st.caption(
        f"Each mark is one frontier language model released {span(filtered)}. The star is the selected model and its values are printed on the chart. "
        "Both axes are logarithmic because sizes span a wide range. The chart shows association, not cause."
    )


def _verify(row: pd.Series) -> None:
    st.markdown(f"##### Verify a model: {row['model_name']}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Parameter count", f"{row['parameters']:,.0f}")
    c2.metric("Dataset size (datapoints)", f"{row['training_dataset_size']:,.0f}")
    c3.metric("Training compute (FLOP)", "not recorded" if pd.isna(row["training_compute_flops"]) else f"{row['training_compute_flops']:.3g}")
    c4.metric("Release year", f"{int(row['release_year'])}")
    st.write(f"**Source confidence:** {row['confidence']}")
    if not row["fresh_dense"]:
        st.info(f"Excluded from the answer by default. {row['exclusion_reason']}")
    link = first_url(row["source_link"])
    if link:
        st.markdown(f"**Source reference:** [open original source]({link})")
    with st.expander("Parameter count notes"):
        st.write(row["parameters_notes"] or "No parameter count notes are recorded for this model.")
    with st.expander("Training data notes"):
        st.write(row["dataset_size_notes"] or "No training data notes are recorded. Check the value against the original source before relying on it.")

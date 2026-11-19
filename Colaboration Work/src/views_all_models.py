"""Lens 1: all models that report a dataset size. Data is counted as dataset size."""

from __future__ import annotations

import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analysis import (
    apply_filters, category_labels, format_count, loglog_fit, option_counts, top_organizations, yearly_medians,
)
from src.attribution import describe_split, estimate_split
from src.views_common import (
    BLACK, BLUE, DATA_LABEL, EMPTY_TEXT, PALETTE, PARAM_LABEL, SYMBOLS, clean_layout, plot,
    render_change, render_download, render_table, span, stacked_share_chart,
)


DEFAULT_START_YEAR = 2010
COLOR_OPTIONS = {"Domain": "domain_group", "Organization sector": "organization_sector"}
SORT_OPTIONS = {
    "Release year (newest first)": ("release_year", False),
    "Parameters (largest first)": ("parameters", False),
    "Training dataset size (largest first)": ("training_dataset_size", False),
}
TABLE_COLUMNS = ["model_name", "release_year", "organization", "domain", "parameters", "training_dataset_size", "source_link"]
TABLE_CONFIG = {
    "model_name": "Model",
    "release_year": st.column_config.NumberColumn("Year", format="%d"),
    "organization": "Organization",
    "domain": "Domain",
    "parameters": st.column_config.NumberColumn(PARAM_LABEL, format="%.0f"),
    "training_dataset_size": st.column_config.NumberColumn(DATA_LABEL, format="%.0f"),
    "source_link": st.column_config.LinkColumn("Source link", display_text="open"),
}


def render_all_models(models: pd.DataFrame, summary: pd.DataFrame, snapshot: str) -> None:
    st.markdown(
        "**Lens 1 counts data as the size of the training dataset** Epoch records, for every model that reports both "
        "parameters and dataset size. Choose a slice below and the answer, charts and table all follow it."
    )
    year_min, year_max = int(models["release_year"].min()), int(models["release_year"].max())
    domain_options = list(option_counts(models, "domain").index)
    c1, c2, c3, c4 = st.columns([2, 3, 3, 2])
    year_range = c1.slider("Release year", year_min, year_max, (max(year_min, DEFAULT_START_YEAR), year_max), key="l1_years")
    organizations = c2.multiselect(
        "Organization", list(option_counts(models, "organization").index), key="l1_orgs",
        help="A model matches if any of its organizations is selected. Leave empty for all.",
    )
    domains = c3.multiselect(
        "Domain", domain_options, default=["Language"] if "Language" in domain_options else [], key="l1_domains",
        help="A model matches if any of its domains is selected. Leave empty for all.",
    )
    color_by = c4.radio("Colour points by", list(COLOR_OPTIONS), key="l1_color")
    st.caption(
        "Dataset size is Epoch's count of training datapoints: roughly tokens or words for language models, but images, "
        "frames or other items elsewhere. Language is selected by default so the units are comparable."
    )

    filtered = apply_filters(models, year_range, organizations, domains)
    if filtered.empty:
        st.warning(EMPTY_TEXT)
        return

    selection = estimate_split(filtered, "dataset_size", bootstrap=True)
    st.info(describe_split(selection, "dataset_size"))
    _metric_cards(filtered, models)
    _share_chart(summary, selection)
    st.markdown("##### How fast did each one rise?")
    render_change(filtered, yearly_medians(filtered), snapshot, "l1")
    _scatter(filtered, color_by)
    _ranking(filtered)
    render_table(filtered, "l1", TABLE_COLUMNS, TABLE_CONFIG, SORT_OPTIONS)
    render_download(filtered, "Download the selected models (CSV)", "selected_ai_models.csv", "l1_download")


def _metric_cards(filtered: pd.DataFrame, models: pd.DataFrame) -> None:
    fit = loglog_fit(filtered)
    first, second, third, fourth = st.columns(4)
    first.metric("Models in selection", f"{len(filtered):,}", help=f"Out of {len(models):,} models in this lens.")
    second.metric("Median parameters (count)", format_count(filtered["parameters"].median()))
    third.metric("Median data (datapoints)", format_count(filtered["training_dataset_size"].median()))
    fourth.metric(
        "Trend slope", "n/a" if fit is None else f"{fit['slope']:.2f}", None if fit is None else f"R² = {fit['r2']:.2f}", delta_color="off",
        help="Slope of log10(parameters) against log10(dataset size) across the selected models. "
        "A slope of 0.5 means ten times more data goes with about 3.2 times more parameters.",
    )
    st.caption(
        f"Selection: {len(filtered):,} models released {span(filtered)}. Baseline, all {len(models):,} models in this lens "
        f"released {span(models)}: median parameters {format_count(models['parameters'].median())}, "
        f"median training data {format_count(models['training_dataset_size'].median())} datapoints."
    )


def _share_chart(summary: pd.DataFrame, selection: dict) -> None:
    st.markdown("##### Who gets the credit for the growth?")
    st.markdown("**Share of yearly growth in parameters × dataset size, by slice of the data**")
    rows = []
    if selection["status"] == "ok":
        rows.append((f"Your current selection ({selection['n_models']:,} models)", selection["params_share"], selection["data_share"]))
    mine = summary[(summary["lens_id"] == "all_models") & (summary["status"] == "ok")]
    for record in mine.itertuples(index=False):
        rows.append((f"{record.row_label} ({record.n_models:,} models)", record.params_share, record.data_share))
    if not rows:
        st.info("No slice has both model size and training data rising, so no split can be drawn.")
        return
    plot(stacked_share_chart(rows, "More data (dataset size)"))
    left_out = summary[(summary["lens_id"] == "all_models") & (summary["status"] != "ok")]
    note = "" if left_out.empty else " Left out because one of the two did not rise or there were too few models: " + "; ".join(left_out["row_label"]) + "."
    st.caption(
        "Each bar splits 100% of the growth in parameters times dataset size between the two. The same data gives different "
        "splits depending on which models you count, which is why the question stays open to argument." + note
    )


def _scatter(filtered: pd.DataFrame, color_by: str) -> None:
    st.markdown("##### Do bigger models go with bigger datasets?")
    st.markdown("**Parameters against training dataset size, one point per model, with a fitted trend line**")
    column = COLOR_OPTIONS[color_by]
    data = filtered.copy()
    data["category"] = category_labels(data, column)
    order = [c for c in data["category"].value_counts().index if c != "Other"]
    if (data["category"] == "Other").any():
        order.append("Other")
    colors = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(order)}
    colors["Other"] = BLACK
    fig = px.scatter(
        data, x="training_dataset_size", y="parameters", color="category", symbol="category",
        color_discrete_map=colors, category_orders={"category": order}, symbol_sequence=SYMBOLS,
        log_x=True, log_y=True, hover_name="model_name", opacity=0.75,
        hover_data={"release_year": True, "organization": True, "training_dataset_size": ":,.0f", "parameters": ":,.0f", "category": False},
        labels={"training_dataset_size": f"{DATA_LABEL}, log scale", "parameters": f"{PARAM_LABEL}, log scale",
                "release_year": "Release year", "organization": "Organization", "category": color_by},
    )
    fit = loglog_fit(filtered)
    if fit is not None:
        low, high = filtered["training_dataset_size"].min(), filtered["training_dataset_size"].max()
        fig.add_trace(go.Scatter(
            x=[low, high], y=[10 ** (fit["intercept"] + fit["slope"] * math.log10(v)) for v in (low, high)],
            mode="lines", name=f"Trend (slope {fit['slope']:.2f}, R² {fit['r2']:.2f})", line={"color": "black", "dash": "dash"}, hoverinfo="skip",
        ))
    biggest = filtered.nlargest(5, "parameters")
    fig.add_trace(go.Scatter(
        x=biggest["training_dataset_size"], y=biggest["parameters"], mode="text", text=[n[:24] for n in biggest["model_name"]],
        textposition="top center", showlegend=False, hoverinfo="skip", cliponaxis=False,
    ))
    fig.update_xaxes(exponentformat="power")
    fig.update_yaxes(exponentformat="power")
    fig.update_layout(legend={"orientation": "h", "yanchor": "top", "y": -0.2, "title": {"text": color_by}})
    plot(clean_layout(fig, 520))
    st.caption(
        f"{len(filtered):,} models released {span(filtered)}. Both axes are logarithmic: each gridline is ten times the one before. "
        "The five models with the most parameters are named, and every point's exact values are in the verification table below. "
        "The dashed line is a least squares fit on the logged values. It describes how the two move together in these models, "
        "not what causes what."
    )


def _ranking(filtered: pd.DataFrame) -> None:
    st.markdown("##### Whose models fill this lens?")
    st.markdown("**Ten organizations with the most models in your selection**")
    top = top_organizations(filtered, 10)
    if top.empty:
        st.info("No named organizations in this selection.")
        return
    top = top.sort_values("models")
    fig = go.Figure(go.Bar(
        x=top["models"], y=top["organization"], orientation="h", marker_color=BLUE,
        text=top["models"], textposition="outside", cliponaxis=False,
    ))
    fig.update_xaxes(title="Models in selection (count)")
    fig.update_yaxes(title="Organization")
    plot(clean_layout(fig, 90 + 34 * len(top)))
    st.caption(
        f"Models released {span(filtered)}. A model built by several organizations counts once for each. "
        "This shows who is over represented in the data, which matters for how far the results can be trusted."
    )

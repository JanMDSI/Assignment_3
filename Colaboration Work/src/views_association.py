"""Lens 4: what moves with training compute across models, and what is missing."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analysis import filter_models, option_counts
from src.association import association_text, compare_associations, spearman, with_all_three
from src.views_common import (
    BLUE, EMPTY_TEXT, ORANGE, clean_layout, line_chart, plot, render_download, render_source_record, span,
)

DEFAULT_START_YEAR = 2010
MAX_POINTS_LABELLED = 20


def _sci(value: float) -> str:
    return f"{value:.1e}"


def _lookup(models: pd.DataFrame, compute: pd.DataFrame) -> pd.DataFrame:
    """Every model that has parameters with dataset size or with compute, one row each."""
    extra = compute[~compute["raw_row"].isin(models["raw_row"])]
    table = pd.concat([models, extra], ignore_index=True)
    for column in ("parameters_notes", "dataset_size_notes", "source_link"):
        table[column] = table[column].fillna("")
    return table


def render_association(coverage: pd.DataFrame, compute: pd.DataFrame, models: pd.DataFrame, snapshot: str) -> None:
    st.markdown(
        "**Lens 4 asks a different question: across models, does training compute move more closely with parameters or with "
        "dataset size?** It also shows how much of each number is missing. It measures association, not growth."
    )
    year_min, year_max = int(coverage["release_year"].min()), int(coverage["release_year"].max())
    c1, c2, c3 = st.columns([3, 2, 3])
    years = c1.slider("Publication year", year_min, year_max, (max(year_min, DEFAULT_START_YEAR), year_max), key="l4_years")
    search = c2.text_input("Search model", placeholder="for example GPT", key="l4_search")
    domains = c3.multiselect("Model domain", list(option_counts(coverage, "domain").index), key="l4_domains",
                             help="A model matches if any of its domains is selected. Leave empty for all.")

    cov = filter_models(coverage, years, search, domains)
    if cov.empty:
        st.warning(EMPTY_TEXT)
        return
    comp = filter_models(compute, years, search, domains)
    sel_models = filter_models(models, years, search, domains)

    _counts(cov, coverage)
    _over_time(comp, cov)
    _correlation(sel_models, comp, cov)
    _missing(cov)
    _explore(filter_models(_lookup(models, compute), years, search, domains))


def _counts(cov: pd.DataFrame, coverage: pd.DataFrame) -> None:
    a, b, c, d = st.columns(4)
    a.metric("Models in selection", f"{len(cov):,}")
    b.metric("Compute reported", f"{int(cov['has_compute'].sum()):,}")
    c.metric("Parameter values", f"{int(cov['has_parameters'].sum()):,}")
    d.metric("Dataset size values", f"{int(cov['has_dataset_size'].sum()):,}")
    st.caption(
        f"Counts are for models published {span(cov)}. Baseline, all {len(coverage):,} dated models in the snapshot: "
        f"{int(coverage['has_compute'].sum()):,} report compute, {int(coverage['has_parameters'].sum()):,} parameters and "
        f"{int(coverage['has_dataset_size'].sum()):,} a dataset size. Missing values are never treated as zero."
    )


def _over_time(comp: pd.DataFrame, cov: pd.DataFrame) -> None:
    st.markdown("##### Training compute over time")
    if comp.empty:
        st.info("No model in this selection reports both compute and parameters, so there is nothing to plot.")
        return
    yearly = comp.groupby("release_year").agg(models=("model_name", "count"), median_compute=("training_compute_flops", "median")).reset_index()
    left, right = st.columns(2)
    with left:
        st.markdown("**Training compute of each model, by publication date**")
        fig = go.Figure(go.Scatter(
            x=pd.to_datetime(comp["publication_date"]), y=comp["training_compute_flops"], mode="markers", text=comp["model_name"],
            marker={"color": BLUE, "size": 7, "opacity": 0.7}, hovertemplate="%{text}<br>%{x|%Y-%m-%d}<br>%{y:.3g} FLOP<extra></extra>",
        ))
        fig.update_yaxes(type="log", title="Training compute (FLOP, log scale)", exponentformat="power")
        fig.update_xaxes(title="Publication date")
        plot(clean_layout(fig, 360))
    with right:
        st.markdown("**Median training compute of models published each year**")
        plot(line_chart(yearly, "median_compute", ORANGE, "Median compute (FLOP, log scale)", label_fn=_sci))
    note = "Labels show each year's median." if len(yearly) <= MAX_POINTS_LABELLED else "Too many years to label every point, so the values are in the table below."
    st.caption(
        f"{len(comp):,} models published {span(comp)} that report both compute and parameters. {note} The vertical axes are logarithmic, "
        "so equal steps mean equal multiples. Years with only one or two models can swing a lot."
    )
    with st.expander("Numbers behind the median chart"):
        st.dataframe(yearly, hide_index=True, width="stretch", column_config={
            "release_year": st.column_config.NumberColumn("Publication year", format="%d"),
            "models": st.column_config.NumberColumn("Models (count)", format="%d"),
            "median_compute": st.column_config.NumberColumn("Median compute (FLOP)", format="%.3e"),
        })


def _correlation(sel_models: pd.DataFrame, comp: pd.DataFrame, cov: pd.DataFrame) -> None:
    st.markdown("##### What moves with training compute?")
    result = compare_associations(sel_models)
    st.info(association_text(result))
    if result["status"] != "ok":
        return
    a, b = st.columns(2)
    a.metric("Spearman, compute with parameters", f"{result['rho_params']:.2f}")
    b.metric("Spearman, compute with dataset size", f"{result['rho_data']:.2f}")
    rows = with_all_three(sel_models)
    left, right = st.columns(2)
    for column, label, color, name in (
        (left, "Parameters (count, log scale)", BLUE, "parameters"),
        (right, "Training dataset size (datapoints, log scale)", ORANGE, "training_dataset_size"),
    ):
        with column:
            st.markdown(f"**Training compute against {'parameters' if name == 'parameters' else 'dataset size'}, one point per model**")
            fig = go.Figure(go.Scatter(
                x=rows[name], y=rows["training_compute_flops"], mode="markers", text=rows["model_name"],
                marker={"color": color, "size": 7, "opacity": 0.7},
                hovertemplate="%{text}<br>%{x:,.0f}<br>%{y:.3g} FLOP<extra></extra>",
            ))
            fig.update_xaxes(type="log", title=label, exponentformat="power")
            fig.update_yaxes(type="log", title="Training compute (FLOP, log scale)", exponentformat="power")
            plot(clean_layout(fig, 360))
    pair_p = comp.dropna(subset=["parameters", "training_compute_flops"])
    pair_note = ""
    if len(pair_p) >= 2:
        pair_note = (
            f" Using every model that has each pair separately, compute with parameters is {spearman(pair_p['training_compute_flops'], pair_p['parameters']):.2f} "
            f"over {len(pair_p):,} models, which is a different group from the {result['n_models']:,} used above."
        )
    st.caption(
        f"Both charts use the same {result['n_models']:,} models published {span(rows)}, so the two correlations compare like with like. Every point's "
        "values are in the verification table below and each chart axis is logarithmic. Spearman correlation uses ranks, so it is not thrown off by the huge range of the numbers. "
        "A positive value means higher compute goes with higher values." + pair_note
    )


def _missing(cov: pd.DataFrame) -> None:
    st.markdown("##### What is missing?")
    st.markdown("**How many models in your selection report each number**")
    measures = [("Parameters", "has_parameters"), ("Training compute", "has_compute"), ("Dataset size", "has_dataset_size")]
    reported = [int(cov[c].sum()) for _, c in measures]
    missing = [len(cov) - r for r in reported]
    labels = [m for m, _ in measures]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=labels, x=reported, orientation="h", name="Reported", marker_color=BLUE, text=reported, textposition="inside", textfont={"color": "white"}))
    fig.add_trace(go.Bar(y=labels, x=missing, orientation="h", name="Missing", marker_color=ORANGE, text=missing, textposition="inside", textfont={"color": "black"}))
    fig.update_layout(barmode="stack", legend={"orientation": "h", "y": -0.3})
    fig.update_xaxes(title="Models in selection (count)")
    fig.update_yaxes(title="Number", autorange="reversed")
    plot(clean_layout(fig, 260))
    st.caption(
        f"{len(cov):,} models published {span(cov)}. Reported means present and above zero, not independently verified. The three bars can "
        "cover different models, which is why comparisons between them need care."
    )


def _explore(lookup: pd.DataFrame) -> None:
    st.markdown("##### Explore a model")
    if lookup.empty:
        st.info("No model in this selection reports parameters together with a dataset size or compute.")
        return
    ordered = lookup.sort_values("model_name")
    names = dict(zip(ordered["raw_row"], ordered["model_name"]))
    pick = st.selectbox("Choose a model", list(names), format_func=lambda row: f"{names[row]} (row {int(row)})", key="l4_pick")
    render_source_record(ordered[ordered["raw_row"] == pick].iloc[0])
    render_download(lookup, "Download the models in this selection (CSV)", "selected_models_lens4.csv", "l4_download")

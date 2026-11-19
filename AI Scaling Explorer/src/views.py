"""Everything drawn on the page. All numbers come from src/analysis.py.

Chart colours follow the Okabe Ito palette, which was designed to stay
distinguishable for the common kinds of colour blindness. Categories also
differ by marker shape, so colour is never the only cue.

Why the main chart uses log scales: parameters and dataset size each span
many powers of ten. On linear axes a handful of huge models would stretch the
scale and push nearly every other point into one corner. On log scales a
power law becomes a straight line, so the slope can be read and measured.
"""

from __future__ import annotations

import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analysis import (
    category_labels,
    compute_rule_check,
    date_label,
    format_count,
    growth_decomposition,
    loglog_fit,
    orders_of_magnitude,
    selection_text,
    short_count,
    top_organizations,
    yearly_medians,
)


BLUE = "#0072B2"
ORANGE = "#E69F00"
GREY = "#000000"
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
SYMBOLS = ["circle", "square", "diamond", "triangle-up", "cross", "x", "star"]

EMPTY_TEXT = (
    "No models match these filters. Try widening the release year range, or clear the "
    "organization or domain choices in the sidebar. Leaving a filter empty means all."
)
COLOR_OPTIONS = {"Domain": "domain_group", "Organization sector": "organization_sector"}
TABLE_COLUMNS = ["model_name", "release_year", "organization", "domain", "parameters", "training_dataset_size", "source_link"]
SORT_OPTIONS = {
    "Release year (newest first)": ("release_year", False),
    "Parameters (largest first)": ("parameters", False),
    "Training dataset size (largest first)": ("training_dataset_size", False),
}
PARAM_LABEL = "Parameters (count)"
DATA_LABEL = "Training dataset size (datapoints)"
MAX_LABELLED_YEARS = 20


def _plot(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def _clean_layout(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(height=height, margin={"l": 10, "r": 20, "t": 20, "b": 10})
    return fig


def _span(frame: pd.DataFrame) -> str:
    return f"{int(frame['release_year'].min())} to {int(frame['release_year'].max())}"


# ------------------------------------------------------- opening screen pieces


def render_answer(answer: str) -> None:
    st.success(f"**Answer from the data:** {answer}")
    st.caption(
        "Descriptive only. The question assumes compute grew exponentially. This page does not test that: it measures "
        "how fast parameters and training data rose, and their product stands in for compute. It follows the trend "
        "across every model that discloses both numbers, not only the largest ones, and it does not show why labs "
        "chose those sizes. Slice definitions and results are in data/processed/growth_by_slice.csv."
    )


def render_glance(models: pd.DataFrame, total_raw: int) -> None:
    """Facts about the whole dataset, shown before any filter is applied."""
    st.subheader("The whole dataset at a glance")
    first, second, third, fourth = st.columns(4)
    first.metric("Models with both numbers", f"{len(models):,}", help=f"Out of {total_raw:,} models in the Epoch AI snapshot.")
    second.metric("Release years covered", _span(models))
    third.metric("Median parameters (count)", format_count(models["parameters"].median()))
    fourth.metric("Median data (datapoints)", format_count(models["training_dataset_size"].median()))
    st.caption(
        f"All {len(models):,} models, every domain, {_span(models)}. Medians are the middle value across models, "
        f"so a few giant models do not drag them. Parameters range over about "
        f"{orders_of_magnitude(models['parameters']):.0f} powers of ten and dataset size over about "
        f"{orders_of_magnitude(models['training_dataset_size']):.0f}."
    )


# ----------------------------------------------------------- part to whole


def render_share_chart(slices: pd.DataFrame, selection: dict) -> None:
    st.subheader("1. Who gets the credit for the growth?")
    st.markdown("**Share of yearly growth in parameters × training data, by slice of the data**")

    rows = []
    if selection["status"] == "ok":
        rows.append((f"Your current selection ({selection['n_models']:,} models)", selection["params_share"], selection["data_share"]))
    shown = slices[slices["status"] == "ok"]
    for record in shown.itertuples(index=False):
        rows.append((f"{record.slice_label} ({record.n_models:,} models)", record.params_share, record.data_share))

    if not rows:
        st.info("No slice has both model size and training data rising, so no split can be drawn.")
        return
    labels = [row[0] for row in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=[row[1] * 100 for row in rows], orientation="h", name="Model size (parameters)",
        marker_color=BLUE, text=[f"{row[1]:.0%}" for row in rows], textposition="inside", textfont={"color": "white"},
    ))
    fig.add_trace(go.Bar(
        y=labels, x=[row[2] * 100 for row in rows], orientation="h", name="Training data (datapoints)",
        marker_color=ORANGE, text=[f"{row[2]:.0%}" for row in rows], textposition="inside", textfont={"color": "black"},
    ))
    fig.update_layout(barmode="stack", legend={"orientation": "h", "y": -0.25})
    fig.update_xaxes(title="Share of yearly growth in parameters × training data (%)", range=[0, 100])
    fig.update_yaxes(title="Slice of the data (models in slice)", autorange="reversed")
    _plot(_clean_layout(fig, 90 + 48 * len(rows)))

    skipped = slices[slices["status"] != "ok"]
    note = ""
    if not skipped.empty:
        note = " Left out because one of the two did not rise or there were too few models: " + "; ".join(skipped["slice_label"]) + "."
    st.caption(
        "Each bar splits 100% of the growth in parameters times training data between the two. "
        "The same data gives different splits depending on which models you count, which is why the "
        "question stays open to argument." + note
    )


# ------------------------------------------------------------------ change


def _line_chart(yearly: pd.DataFrame, column: str, color: str, y_title: str) -> go.Figure:
    labelled = len(yearly) <= MAX_LABELLED_YEARS
    fig = go.Figure(go.Scatter(
        x=yearly["release_year"], y=yearly[column], mode="lines+markers+text" if labelled else "lines+markers",
        text=[short_count(v) for v in yearly[column]] if labelled else None,
        textposition="top center", cliponaxis=False,
        line={"color": color, "width": 3}, marker={"color": color, "size": 7},
        hovertemplate="%{x}: %{y:,.0f}<extra></extra>",
    ))
    fig.update_yaxes(type="log", title=y_title, exponentformat="power")
    fig.update_xaxes(title="Release year")
    return _clean_layout(fig, 360)


def render_change(filtered: pd.DataFrame, snapshot: str) -> None:
    st.subheader("2. How fast did each one rise?")
    yearly = yearly_medians(filtered)
    left, right = st.columns(2)
    with left:
        st.markdown("**Median parameters of models released each year**")
        _plot(_line_chart(yearly, "median_parameters", BLUE, "Median parameters (count, log scale)"))
    with right:
        st.markdown("**Median training dataset size of models released each year**")
        _plot(_line_chart(yearly, "median_dataset_size", ORANGE, "Median dataset size (datapoints, log scale)"))
    labelled_note = "Labels show each year's median." if len(yearly) <= MAX_LABELLED_YEARS else "Too many years to label every point, so the full values are in the table below."
    st.caption(
        f"Models released {_span(filtered)}, {len(filtered):,} models in your selection. {labelled_note} "
        f"The vertical axes are logarithmic, so equal steps mean equal multiples. The final year can be "
        f"incomplete because the snapshot was taken on {snapshot}."
    )
    with st.expander("Numbers behind these two charts"):
        st.dataframe(
            yearly, hide_index=True, width="stretch",
            column_config={
                "release_year": st.column_config.NumberColumn("Release year", format="%d"),
                "models": st.column_config.NumberColumn("Models (count)", format="%d"),
                "median_parameters": st.column_config.NumberColumn("Median parameters (count)", format="%.0f"),
                "median_dataset_size": st.column_config.NumberColumn("Median dataset size (datapoints)", format="%.0f"),
            },
        )


# ------------------------------------------------------------- correlation


def render_metric_cards(filtered: pd.DataFrame, models: pd.DataFrame) -> None:
    fit = loglog_fit(filtered)
    first, second, third, fourth = st.columns(4)
    first.metric("Models in selection", f"{len(filtered):,}", help=f"Out of {len(models):,} models in the whole dataset.")
    second.metric("Median parameters (count)", format_count(filtered["parameters"].median()))
    third.metric("Median data (datapoints)", format_count(filtered["training_dataset_size"].median()))
    fourth.metric(
        "Trend slope",
        "n/a" if fit is None else f"{fit['slope']:.2f}",
        None if fit is None else f"R² = {fit['r2']:.2f}",
        delta_color="off",
        help="Slope of log10(parameters) against log10(dataset size) across the selected models. "
        "A slope of 0.5 means ten times more data goes with about 3.2 times more parameters.",
    )
    st.caption(
        f"Selection: {len(filtered):,} models released {_span(filtered)}. Baseline, all {len(models):,} models "
        f"released {_span(models)}: median parameters {format_count(models['parameters'].median())}, "
        f"median training data {format_count(models['training_dataset_size'].median())} datapoints."
    )


def render_scatter(filtered: pd.DataFrame, color_by: str) -> None:
    st.subheader("3. Do bigger models go with bigger datasets?")
    st.markdown("**Parameters against training dataset size, one point per model, with a fitted trend line**")
    column = COLOR_OPTIONS[color_by]
    plot = filtered.copy()
    plot["category"] = category_labels(plot, column)
    order = [c for c in plot["category"].value_counts().index if c != "Other"]
    if (plot["category"] == "Other").any():
        order.append("Other")
    colors = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(order)}
    colors["Other"] = GREY

    fig = px.scatter(
        plot, x="training_dataset_size", y="parameters", color="category", symbol="category",
        color_discrete_map=colors, category_orders={"category": order}, symbol_sequence=SYMBOLS,
        log_x=True, log_y=True, hover_name="model_name", opacity=0.75,
        hover_data={"release_year": True, "organization": True, "training_dataset_size": ":,.0f", "parameters": ":,.0f", "category": False},
        labels={
            "training_dataset_size": f"{DATA_LABEL}, log scale",
            "parameters": f"{PARAM_LABEL}, log scale",
            "release_year": "Release year", "organization": "Organization", "category": color_by,
        },
    )
    fit = loglog_fit(filtered)
    if fit is not None:
        x_low, x_high = filtered["training_dataset_size"].min(), filtered["training_dataset_size"].max()
        fig.add_trace(go.Scatter(
            x=[x_low, x_high],
            y=[10 ** (fit["intercept"] + fit["slope"] * math.log10(x)) for x in (x_low, x_high)],
            mode="lines", name=f"Trend (slope {fit['slope']:.2f}, R² {fit['r2']:.2f})",
            line={"color": "black", "dash": "dash"}, hoverinfo="skip",
        ))
    biggest = filtered.nlargest(5, "parameters")
    fig.add_trace(go.Scatter(
        x=biggest["training_dataset_size"], y=biggest["parameters"], mode="text",
        text=[name[:24] for name in biggest["model_name"]], textposition="top center",
        showlegend=False, hoverinfo="skip", cliponaxis=False,
    ))
    fig.update_xaxes(exponentformat="power")
    fig.update_yaxes(exponentformat="power")
    fig.update_layout(legend={"orientation": "h", "yanchor": "top", "y": -0.2, "title": {"text": color_by}})
    _plot(_clean_layout(fig, 520))
    st.caption(
        f"{len(filtered):,} models released {_span(filtered)}. Both axes are logarithmic: each gridline is ten times "
        "the one before. The five models with the most parameters are named, and every point's exact values are in the "
        "verification table below. The dashed line is a least squares fit on the logged values. It describes how the two "
        "move together in these models, not what causes what."
    )


# ----------------------------------------------------------------- ranking


def render_ranking(filtered: pd.DataFrame) -> None:
    st.subheader("4. Whose models fill this dataset?")
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
    _plot(_clean_layout(fig, 90 + 34 * len(top)))
    st.caption(
        f"Models released {_span(filtered)}. A model built by several organizations counts once for each. "
        "This shows who is over represented in the data, which matters for how far the results can be trusted."
    )


# ------------------------------------------------------------------- table


def render_table(filtered: pd.DataFrame) -> None:
    """Paginated, sortable table with a source panel for a clicked row."""
    st.subheader("Verification table: every point behind the charts")
    controls = st.columns(3)
    sort_label = controls[0].selectbox("Sort by", list(SORT_OPTIONS), key="table_sort")
    page_size = controls[1].selectbox("Rows per page", [10, 25, 50], index=1, key="table_page_size")
    sort_column, ascending = SORT_OPTIONS[sort_label]
    ordered = filtered.sort_values([sort_column, "model_name"], ascending=[ascending, True]).reset_index(drop=True)

    pages = max(1, math.ceil(len(ordered) / page_size))
    page = controls[2].number_input("Page", min_value=1, max_value=pages, value=1, step=1, key="table_page")
    start = (int(page) - 1) * page_size
    shown = ordered.iloc[start : start + page_size]
    st.caption(f"Rows {start + 1:,} to {start + len(shown):,} of {len(ordered):,}. Click a row to see its source record.")

    event = st.dataframe(
        shown[TABLE_COLUMNS], hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
        key=f"table_{sort_column}_{page_size}_{int(page)}_{len(ordered)}",
        column_config={
            "model_name": "Model",
            "release_year": st.column_config.NumberColumn("Year", format="%d"),
            "organization": "Organization",
            "domain": "Domain",
            "parameters": st.column_config.NumberColumn(PARAM_LABEL, format="%.0f"),
            "training_dataset_size": st.column_config.NumberColumn(DATA_LABEL, format="%.0f"),
            "source_link": st.column_config.LinkColumn("Source link", display_text="open"),
        },
    )
    selected = event.selection.rows if event and event.selection else []
    if selected:
        row = shown.iloc[selected[0]]
        st.markdown(f"**Source record: {row['model_name']}** (published {date_label(row['publication_date'])})")
        st.write(f"Parameters: {row['parameters']:,.0f} ({format_count(row['parameters'])})")
        st.write(f"{DATA_LABEL}: {row['training_dataset_size']:,.0f} ({format_count(row['training_dataset_size'])})")
        if row["parameters_notes"]:
            st.write(f"Epoch notes on parameters: {row['parameters_notes']}")
        if row["dataset_size_notes"]:
            st.write(f"Epoch notes on dataset size: {row['dataset_size_notes']}")
        if row["source_link"]:
            st.write(f"Source: {row['source_link']}")


def render_download(filtered: pd.DataFrame) -> None:
    st.download_button(
        "Download the selected models (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="selected_ai_models.csv",
        mime="text/csv",
        key="download_selected",
    )


# ---------------------------------------------------- limits and provenance


def render_limitations(models: pd.DataFrame, cleaning: pd.DataFrame) -> None:
    st.subheader("What this data cannot show")
    total = int(cleaning["models_remaining"].iloc[0])
    check = compute_rule_check(models)
    few = models.groupby("release_year").size()
    sparse = int((few < 10).sum())
    lines = [
        f"**Only disclosers appear.** {len(models):,} of {total:,} models in the Epoch snapshot ({len(models) / total:.0%}) report both "
        "numbers. Labs that keep sizes secret are missing, so the pattern may not hold for them.",
        "**Dataset size is not one unit.** It counts tokens for most language models but images, frames or other items elsewhere. "
        "Comparing it across domains is rough, which is why Language is selected by default.",
        "**Dataset size is not data processed.** A model that reads its data several times uses more compute without a larger dataset, "
        "so the split here is not an accounting of the compute actually spent.",
        "**The split leans on a rule of thumb.** It assumes training compute is about 6 × parameters × data.",
    ]
    if check is not None:
        lines.append(
            f"Of the {check['n_models']:,} models with a reported compute figure, {check['share_equal']:.0%} equal that rule within 1%, "
            "so part of Epoch's compute column appears to be built from the same rule and cannot test it independently."
        )
    lines += [
        "**Total parameters may overstate the work.** Models that use only part of their parameters for each token can report the full count.",
        f"**Early years are thin.** {sparse} release years have fewer than 10 models, so their medians are noisy.",
        "**Correlation, not cause.** Nothing here shows why labs chose these sizes or what will happen next.",
    ]
    for line in lines:
        st.markdown(f"* {line}")


def render_sources(source: dict, snapshot: str) -> None:
    accessed = date_label(source["accessed"])
    st.caption(
        f"Source: {source['publisher']}, \"{source['title']}\" ({source['url']}), licence {source['licence']}. "
        f"Downloaded {accessed}. Latest record change in the file: {snapshot}."
    )


def render_cleaning(cleaning: pd.DataFrame, source: dict) -> None:
    with st.expander("Data source and cleaning steps"):
        st.markdown(f"**Citation:** {source['citation']}  \nFile used: `{source['file']}`, downloaded {date_label(source['accessed'])}.")
        st.table(cleaning.rename(columns={"step": "Cleaning step", "models_remaining": "Models remaining (count)"}).set_index("Cleaning step"))
        st.markdown(
            "Only models that report **both** parameters and a usable dataset size can be placed on the charts. "
            "Two comma separated dataset sizes that disagreed were dropped, not averaged."
        )


def render_log_note(models: pd.DataFrame) -> None:
    with st.expander("Why logarithmic axes instead of linear ones?"):
        st.markdown(
            f"Parameters run from {format_count(models['parameters'].min())} to {format_count(models['parameters'].max())}, and dataset size "
            f"from {format_count(models['training_dataset_size'].min())} to {format_count(models['training_dataset_size'].max())} datapoints. "
            "On a linear axis a few giant models would stretch the scale and squash nearly everything else into one corner. "
            "On a logarithmic axis a power law is a straight line whose slope can be read directly."
        )


def render_explorer(filtered: pd.DataFrame, models: pd.DataFrame, slices: pd.DataFrame, color_by: str, snapshot: str) -> None:
    """Metrics and the four chart sections for the filtered models, or a helpful empty message."""
    if filtered.empty:
        st.warning(EMPTY_TEXT)
        return
    selection = growth_decomposition(filtered)
    st.info(selection_text(selection))
    render_metric_cards(filtered, models)
    render_share_chart(slices, selection)
    render_change(filtered, snapshot)
    render_scatter(filtered, color_by)
    render_ranking(filtered)
    render_table(filtered)
    render_download(filtered)

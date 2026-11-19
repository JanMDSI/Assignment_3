"""Chart and table pieces shared by every lens.

Chart colours follow the Okabe Ito palette, which was designed to stay
distinguishable for the common kinds of colour blindness. Categories also
differ by marker shape, so colour is never the only cue.

Why the scatter plots use log scales: parameters and dataset size each span
many powers of ten. On linear axes a handful of huge models would stretch the
scale and push nearly every other point into one corner. On log scales a
power law becomes a straight line, so the slope can be read and measured.
"""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analysis import date_label, format_count, short_count
from src.attribution import METHOD_SHORT


BLUE = "#0072B2"
ORANGE = "#E69F00"
BLACK = "#000000"
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
SYMBOLS = ["circle", "square", "diamond", "triangle-up", "cross", "x", "star"]
METHOD_STYLE = {
    "dataset_size": {"color": BLUE, "symbol": "circle"},
    "tokens_processed": {"color": "#D55E00", "symbol": "diamond"},
}
MAX_LABELLED_YEARS = 20
PARAM_LABEL = "Parameters (count)"
DATA_LABEL = "Training dataset size (datapoints)"

EMPTY_TEXT = (
    "No models match these filters. Try widening the release year range, or clear the "
    "other choices above. Leaving a list empty means all."
)


def plot(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def clean_layout(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(height=height, margin={"l": 10, "r": 20, "t": 20, "b": 10})
    return fig


def span(frame: pd.DataFrame) -> str:
    return f"{int(frame['release_year'].min())} to {int(frame['release_year'].max())}"


def first_url(text: object) -> str:
    """First web address in a cell that may hold several separated by semicolons or spaces."""
    for token in str(text).replace(";", " ").split():
        if token.startswith("http"):
            return token
    return ""


# ----------------------------------------------------- uncertainty (forest) chart


def interval_chart(rows: pd.DataFrame, height: int | None = None) -> go.Figure:
    """Dot and range chart of the share of growth from bigger models.

    `rows` needs: label, n_models, method, params_share, share_lo, share_hi (fractions).
    The dashed line at 50% is an even split: dots left of it mean more data mattered more.
    """
    labels = [f"{r.label} ({r.n_models:,} models)" for r in rows.itertuples(index=False)]
    rows = rows.assign(axis_label=labels)
    fig = go.Figure()
    for method, group in rows.groupby("method", sort=False):
        style = METHOD_STYLE[method]
        share = group["params_share"] * 100
        low = group["share_lo"].fillna(group["params_share"]) * 100
        high = group["share_hi"].fillna(group["params_share"]) * 100
        fig.add_trace(go.Scatter(
            x=share, y=group["axis_label"], mode="markers+text", name=METHOD_SHORT[method],
            marker={"color": style["color"], "symbol": style["symbol"], "size": 12},
            text=[f"{v:.0f}%" for v in share], textposition="top center", cliponaxis=False,
            error_x={"type": "data", "symmetric": False, "array": (high - share).tolist(),
                     "arrayminus": (share - low).tolist(), "color": style["color"], "thickness": 2},
            hovertemplate="%{y}<br>%{x:.1f}% from bigger models<extra></extra>",
        ))
    fig.add_vline(x=50, line_dash="dash", line_color="grey")
    fig.update_xaxes(title="Share of yearly growth from bigger models (%), dot and 95% range", range=[-5, 105])
    fig.update_yaxes(title="Estimate (models in it)", categoryorder="array", categoryarray=labels, autorange="reversed")
    fig.update_layout(legend={"orientation": "h", "y": -0.25, "title": {"text": "Data counted as"}})
    return clean_layout(fig, height or 110 + 38 * len(rows))


def render_interval(rows: pd.DataFrame, title: str, caption: str) -> None:
    st.markdown(f"**{title}**")
    plot(interval_chart(rows))
    st.caption(caption)


# ------------------------------------------------------------- part to whole


def stacked_share_chart(rows: list[tuple[str, float, float]], data_name: str) -> go.Figure:
    """100% stacked bars. rows are (label, params_share, data_share) as fractions."""
    labels = [row[0] for row in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=[row[1] * 100 for row in rows], orientation="h", name="Bigger models (parameters)",
        marker_color=BLUE, text=[f"{row[1]:.0%}" for row in rows], textposition="inside", textfont={"color": "white"},
    ))
    fig.add_trace(go.Bar(
        y=labels, x=[row[2] * 100 for row in rows], orientation="h", name=data_name,
        marker_color=ORANGE, text=[f"{row[2]:.0%}" for row in rows], textposition="inside", textfont={"color": "black"},
    ))
    fig.update_layout(barmode="stack", legend={"orientation": "h", "y": -0.25})
    fig.update_xaxes(title="Share of yearly growth in parameters × data (%)", range=[0, 100])
    fig.update_yaxes(title="Slice of the data (models in it)", autorange="reversed")
    return clean_layout(fig, 90 + 48 * len(rows))


# -------------------------------------------------------------------- change


def line_chart(yearly: pd.DataFrame, column: str, color: str, y_title: str, log: bool = True, label_fn=None) -> go.Figure:
    labelled = len(yearly) <= MAX_LABELLED_YEARS
    fig = go.Figure(go.Scatter(
        x=yearly["release_year"], y=yearly[column], mode="lines+markers+text" if labelled else "lines+markers",
        text=[(label_fn or (short_count if log else (lambda v: f"{v:.2g}")))(v) for v in yearly[column]] if labelled else None,
        textposition="top center", cliponaxis=False,
        line={"color": color, "width": 3}, marker={"color": color, "size": 7},
        hovertemplate="%{x}: %{y:,.2f}<extra></extra>",
    ))
    fig.update_yaxes(type="log" if log else "linear", title=y_title, exponentformat="power")
    fig.update_xaxes(title="Release year")
    return clean_layout(fig, 360)


def render_change(filtered: pd.DataFrame, yearly: pd.DataFrame, snapshot: str, key_prefix: str) -> None:
    left, right = st.columns(2)
    with left:
        st.markdown("**Median parameters of models released each year**")
        plot(line_chart(yearly, "median_parameters", BLUE, "Median parameters (count, log scale)"))
    with right:
        st.markdown("**Median training dataset size of models released each year**")
        plot(line_chart(yearly, "median_dataset_size", ORANGE, "Median dataset size (datapoints, log scale)"))
    note = "Labels show each year's median." if len(yearly) <= MAX_LABELLED_YEARS else "Too many years to label every point, so the full values are in the table below."
    st.caption(
        f"Models released {span(filtered)}, {len(filtered):,} models in your selection. {note} "
        f"The vertical axes are logarithmic, so equal steps mean equal multiples. The final year can be "
        f"incomplete because the snapshot was taken on {snapshot}. Years with only one or two models can swing a lot."
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


# ---------------------------------------------------------------------- table


def render_table(
    frame: pd.DataFrame,
    prefix: str,
    columns: list[str],
    column_config: dict,
    sort_options: dict[str, tuple[str, bool]],
) -> None:
    """Paginated, sortable table with a source panel for a clicked row."""
    st.markdown("**Verification table: every point behind the charts**")
    controls = st.columns(3)
    sort_label = controls[0].selectbox("Sort by", list(sort_options), key=f"{prefix}_sort")
    page_size = controls[1].selectbox("Rows per page", [10, 25, 50], index=1, key=f"{prefix}_page_size")
    sort_column, ascending = sort_options[sort_label]
    ordered = frame.sort_values([sort_column, "model_name"], ascending=[ascending, True]).reset_index(drop=True)

    pages = max(1, math.ceil(len(ordered) / page_size))
    page = controls[2].number_input("Page", min_value=1, max_value=pages, value=1, step=1, key=f"{prefix}_page")
    start = (int(page) - 1) * page_size
    shown = ordered.iloc[start : start + page_size]
    st.caption(f"Rows {start + 1:,} to {start + len(shown):,} of {len(ordered):,}. Click a row to see its source record.")

    event = st.dataframe(
        shown[columns], hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
        key=f"{prefix}_table_{sort_column}_{page_size}_{int(page)}_{len(ordered)}", column_config=column_config,
    )
    selected = event.selection.rows if event and event.selection else []
    if selected:
        render_source_record(shown.iloc[selected[0]])


def render_source_record(row: pd.Series) -> None:
    st.markdown(f"**Source record: {row['model_name']}** (published {date_label(row['publication_date'])}, row {int(row['raw_row'])} of the raw file)")
    st.write(f"Parameters: {row['parameters']:,.0f} ({format_count(row['parameters'])})")
    if "training_dataset_size" in row and pd.notna(row["training_dataset_size"]):
        st.write(f"{DATA_LABEL}: {row['training_dataset_size']:,.0f} ({format_count(row['training_dataset_size'])})")
    if "training_compute_flops" in row and pd.notna(row["training_compute_flops"]):
        st.write(f"Training compute (FLOP): {row['training_compute_flops']:.3g}")
    for column, label in (("parameters_notes", "Epoch notes on parameters"), ("dataset_size_notes", "Epoch notes on dataset size")):
        if column in row and row[column]:
            st.write(f"{label}: {row[column]}")
    link = first_url(row.get("source_link", ""))
    if link:
        st.write(f"Source: {link}")


def render_download(frame: pd.DataFrame, label: str, file_name: str, key: str) -> None:
    st.download_button(label, data=frame.to_csv(index=False).encode("utf-8"), file_name=file_name, mime="text/csv", key=key)

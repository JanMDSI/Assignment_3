"""The opening screen and the closing sections: answer, whole dataset facts, limits, sources."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analysis import compute_rule_check, date_label, format_count, orders_of_magnitude
from src.attribution import independent_compute
from src.views_common import render_interval


def render_answer(text: str, cross_check: str = "") -> None:
    st.success(f"**Answer from the data:** {text}")
    if cross_check:
        st.markdown(f"**Cross check from lens 4:** {cross_check}")
    st.caption(
        "Descriptive only. The question assumes compute grew exponentially. This app does not test that: it measures how fast "
        "parameters and data rose, and their product stands in for compute. It follows the trend across every model that "
        "discloses the numbers, not only the largest ones, and it does not show why labs chose those sizes. The 95% ranges "
        "show sampling noise only, not the choice of method."
    )


def render_glance(models: pd.DataFrame, compute: pd.DataFrame, frontier: pd.DataFrame, total_raw: int) -> None:
    """Facts about the whole data package, shown before any filter is applied."""
    st.subheader("The whole dataset at a glance")
    cols = st.columns(5)
    cols[0].metric("Models in the Epoch snapshot", f"{total_raw:,}")
    cols[1].metric("With parameters and dataset size", f"{len(models):,}", help="The models lens 1 and lens 3 use.")
    cols[2].metric("With parameters and compute", f"{len(compute):,}", help="The models lens 2 uses.")
    cols[3].metric("Frontier language models", f"{len(frontier):,}", help="The models lens 3 uses.")
    cols[4].metric("Newest model published", date_label(models["publication_date"].max()))
    st.caption(
        f"All counts are for the whole snapshot, every domain, release years {int(models['release_year'].min())} to "
        f"{int(models['release_year'].max())} for models with a dataset size. Among them, median parameters are "
        f"{format_count(models['parameters'].median())} (count) and median training data {format_count(models['training_dataset_size'].median())} "
        f"datapoints. Parameters span about {orders_of_magnitude(models['parameters']):.0f} powers of ten and dataset size about "
        f"{orders_of_magnitude(models['training_dataset_size']):.0f}."
    )


def render_all_estimates(summary: pd.DataFrame) -> None:
    ok = summary[summary["status"] == "ok"].copy()
    st.subheader("Every estimate in one picture")
    if ok.empty:
        st.info("No estimate has both model size and data rising, so there is nothing to draw.")
        return
    ok["label"] = ok["lens_label"] + ": " + ok["row_label"]
    render_interval(
        ok, "Share of yearly growth from bigger models, across all three lenses",
        "Each dot is one estimate and the line is its 95% range. Dots left of the dashed line mean more data mattered more, dots right "
        "mean bigger models mattered more. Blue dots count data as dataset size, red diamonds as tokens actually processed. "
        "Windows differ by row (see the labels), and each row is a defensible choice, which is why the answer is arguable.",
    )


def render_limitations(models: pd.DataFrame, compute: pd.DataFrame, frontier: pd.DataFrame, cleaning: pd.DataFrame) -> None:
    st.subheader("What this data cannot show")
    total = int(cleaning[cleaning["table"] == "models"]["models_remaining"].iloc[0])
    check = compute_rule_check(models)
    sparse = int((models.groupby("release_year").size() < 10).sum())
    independent = len(independent_compute(compute))
    flagged = int((~frontier["fresh_dense"]).sum()) if not frontier.empty else 0
    lines = [
        f"**Only disclosers appear.** {len(models):,} of {total:,} models in the snapshot ({len(models) / total:.0%}) report a parameter count and "
        f"a dataset size, and {len(compute):,} ({len(compute) / total:.0%}) report parameters and compute. Labs that keep these secret are missing, "
        "so the pattern may not hold for them.",
        "**Dataset size is not one unit.** It counts tokens for most language models but images, frames or other items elsewhere. Comparing it "
        "across domains is rough, which is why Language is selected by default in lens 1.",
        "**Dataset size is not data processed.** A model that reads its data several times uses more compute without a larger dataset. That is "
        "why lens 2 and lens 3 also count tokens processed, and why the two measures can disagree.",
        "**The split leans on a rule of thumb.** It assumes training compute is about 6 × parameters × data.",
    ]
    if check is not None:
        lines.append(
            f"Of the {check['n_models']:,} models with a reported compute figure, {check['share_equal']:.0%} equal that rule within 1%, so part of Epoch's "
            f"compute column appears to be built from the same rule and cannot test it independently. Only {independent:,} models in lens 2 have "
            "compute that the developer reported or that Epoch estimated from hardware."
        )
    lines += [
        "**Total parameters may overstate the work.** Models that use only part of their parameters for each token can report the full count. "
        f"Lens 3 sets aside {flagged} fine tuned or mixture of experts frontier models by default.",
        f"**The frontier sample is small.** Lens 3 has {len(frontier)} models, so single models move its estimates a lot. The 95% ranges show this.",
        f"**Early years are thin.** {sparse} release years have fewer than 10 models, so their medians are noisy.",
        "**The last year is partial.** Models published after the snapshot date are not in it.",
        "**Lens 4 shows association, not growth.** A rank correlation says how closely two numbers move together across models. It does not say how fast "
        "either one grew, and compute figures that Epoch derived from parameters and data are expected to correlate with both.",
        "**Correlation, not cause.** Nothing here shows why labs chose these sizes or what will happen next. Compute measures cost, not capability.",
    ]
    for line in lines:
        st.markdown(f"* {line}")


def render_sources(source: dict, snapshot: str) -> None:
    st.caption(
        f"Source: {source['publisher']}, \"{source['title']}\" ({source['url']}), licence {source['licence']}. "
        f"Downloaded {date_label(source['accessed'])}. Latest record change in the file: {snapshot}."
    )


def render_cleaning(cleaning: pd.DataFrame, source: dict) -> None:
    with st.expander("Data source and cleaning steps"):
        st.markdown(f"**Citation:** {source['citation']}  \nFile used: `{source['file']}`, downloaded {date_label(source['accessed'])}.")
        names = {"models": "Models table (parameters and dataset size)", "compute_models": "Compute table (parameters and compute)"}
        for table, group in cleaning.groupby("table", sort=False):
            st.markdown(f"**{names.get(table, table)}**")
            st.table(group.drop(columns="table").rename(columns={"step": "Cleaning step", "models_remaining": "Models remaining (count)"}).set_index("Cleaning step"))
        st.markdown(
            "Two comma separated dataset sizes that disagreed were dropped, not averaged. Row numbers in the app (for example in "
            "the verification tables) are spreadsheet row numbers in the raw file, where the header is row 1."
        )


def render_method() -> None:
    with st.expander("How the split is worked out"):
        st.markdown(
            "A common rule of thumb says training compute is about 6 × parameters × data. In base 10 logs that is a sum, so the yearly "
            "growth of compute is the yearly growth of parameters plus the yearly growth of data, and each part divided by the total is "
            "its share. Growth is the slope of a straight line fitted to the logged values against release date. The 95% range comes "
            "from resampling the models 1,000 times with replacement. If either part did not rise, no share is given."
        )

"""The Rise of Artificial Intelligence: Data Narrative Studio, AI Model Scaling Explorer."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analysis import answer_text, apply_filters, date_label, option_counts
from src.data import load_processed, load_source_info
from src.views import (
    COLOR_OPTIONS,
    render_answer,
    render_cleaning,
    render_explorer,
    render_glance,
    render_limitations,
    render_log_note,
    render_sources,
)


QUESTION = (
    "As AI training compute grew exponentially, how much of that growth stemmed "
    "from larger models versus expanded training datasets?"
)
DEFAULT_START_YEAR = 2010

st.set_page_config(page_title="AI Model Scaling Explorer", layout="wide")


@st.cache_data(show_spinner="Loading data...")
def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    models, slices, cleaning = load_processed()
    return models, slices, cleaning, load_source_info()


models, slices, cleaning, source = load_all()
snapshot = date_label(pd.to_datetime(models["last_modified"], errors="coerce", utc=True).max())
total_raw = int(cleaning["models_remaining"].iloc[0])

# ---- opening screen: question, answer, glance at the whole dataset ----------
st.title("The Rise of Artificial Intelligence: Data Narrative Studio")
st.header("AI Model Scaling Explorer")
st.markdown(f"> **Question:** {QUESTION}")
render_answer(answer_text(slices))
st.markdown(
    "**For:** a technology policy analyst deciding whether to focus on the cost of building larger models "
    "or on access to training data. Pick the models you care about in the sidebar, then read the answer for "
    "your selection and trace any number to its source row."
)
render_sources(source, snapshot)
render_glance(models, total_raw)

# ---- sidebar controls ----------------------------------------------------------
with st.sidebar:
    st.header("Filters")
    year_min, year_max = int(models["release_year"].min()), int(models["release_year"].max())
    year_range = st.slider(
        "Release year",
        min_value=year_min,
        max_value=year_max,
        value=(max(year_min, DEFAULT_START_YEAR), year_max),
        key="year_range",
    )
    organizations = st.multiselect(
        "Organization",
        list(option_counts(models, "organization").index),
        key="organizations",
        help="A model matches if any of its organizations is selected. Leave empty for all.",
    )
    domain_options = list(option_counts(models, "domain").index)
    domains = st.multiselect(
        "Domain",
        domain_options,
        default=["Language"] if "Language" in domain_options else [],
        key="domains",
        help="A model matches if any of its domains is selected. Leave empty for all.",
    )
    color_by = st.radio("Colour points by", list(COLOR_OPTIONS), key="color_by")
    st.caption(
        "Dataset size is Epoch's count of training datapoints: roughly tokens or words for language "
        "models, but images, frames or other items elsewhere. Language is selected by default so the "
        "units are comparable."
    )

# ---- explorer ----------------------------------------------------------------------
render_log_note(models)
filtered = apply_filters(models, year_range, organizations, domains)
render_explorer(filtered, models, slices, color_by, snapshot)

# ---- limits and provenance ---------------------------------------------------------
render_limitations(models, cleaning)
render_cleaning(cleaning, source)
render_sources(source, snapshot)

"""AI Scaling Explorer: where did the growth in AI training scale come from?"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.analysis import date_label
from src.association import association_text
from src.attribution import headline
from src.data import load_processed, load_source_info
from src.views_all_models import render_all_models
from src.views_association import render_association
from src.views_compute import render_compute
from src.views_frontier import render_frontier
from src.views_overview import (
    render_all_estimates, render_answer, render_cleaning, render_glance, render_limitations, render_method, render_sources,
)


QUESTION = (
    "As AI training compute grew exponentially, how much of that growth stemmed "
    "from larger models versus expanded training datasets?"
)

# Each lens is one team member's analysis. To add another, write a render function
# in its own src/views_*.py file, give it the tables it needs, and add a line here.
st.set_page_config(page_title="AI Scaling Explorer", layout="wide")


@st.cache_data(show_spinner="Loading data...")
def load_all() -> tuple[dict, dict]:
    return load_processed(), load_source_info()


tables, source = load_all()
models, compute, frontier = tables["models"], tables["compute"], tables["frontier"]
summary, cleaning = tables["summary"], tables["cleaning"]
coverage, associations = tables["coverage"], tables["associations"]
snapshot = date_label(pd.to_datetime(models["last_modified"], errors="coerce", utc=True).max())
total_raw = int(cleaning[cleaning["table"] == "models"]["models_remaining"].iloc[0])

# ---- opening screen: question, answer, the whole dataset ---------------------------------
st.title("AI Scaling Explorer")
st.markdown(f"> **Question:** {QUESTION}")
overall = associations[associations["scope_id"] == "all"].iloc[0].to_dict()
render_answer(headline(summary), association_text(overall))
st.markdown(
    "**For:** a technology policy analyst deciding whether to focus on the cost of building larger models or on access to training "
    "data. Read the answer, compare the estimates, then open a lens below to filter to the models you care about and trace any "
    "number to its source row."
)
render_sources(source, snapshot)
render_glance(models, compute, frontier, total_raw)
render_all_estimates(summary)
render_method()

# ---- the lenses ------------------------------------------------------------------------------
st.subheader("Explore the four lenses")
tab_all, tab_compute, tab_frontier, tab_assoc, tab_limits = st.tabs([
    "Lens 1: all models, dataset size",
    "Lens 2: compute, tokens processed",
    "Lens 3: frontier language models",
    "Lens 4: what moves with compute",
    "Limits, sources and cleaning",
])
with tab_all:
    render_all_models(models, summary, snapshot)
with tab_compute:
    render_compute(compute)
with tab_frontier:
    render_frontier(frontier, snapshot)
with tab_assoc:
    render_association(coverage, compute, models, snapshot)
with tab_limits:
    render_limitations(models, compute, frontier, cleaning)
    render_cleaning(cleaning, source)
    render_sources(source, snapshot)

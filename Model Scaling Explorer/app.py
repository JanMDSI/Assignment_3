import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from pathlib import Path

# --------------------------------------------------
# 0. Settings
# --------------------------------------------------
QUESTION = (
    "When AI training compute grew, did that come from bigger models, "
    "or from training them on more data?"
)
PERSONA = "technology policy analysts and AI researchers"

DATA_SOURCE_NAME = "Epoch AI – Data on AI models (CC BY 4.0)"
DATA_SOURCE_URL = "https://epoch.ai/data/ai-models"
DATA_DOWNLOADED = "18 November 2026"

TRAINING_UNIT = "tokens"

# Models that are not a fresh pretraining run of a dense model.
# Their recorded training data covers only extra training on top of an
# existing model, or (Switch) they are mixture-of-experts models, where only
# part of the parameters is used for each token.
NOT_FRESH_DENSE = {
    "Flan-PaLM 540B": "instruction fine-tune of PaLM 540B (data = 1.4B fine-tuning tokens)",
    "U-PaLM (540B)": "continued training of PaLM 540B (data = 1.3B extra tokens)",
    "Minerva (540B)": "fine-tune of PaLM 540B on maths data (26B tokens)",
    "FLAN 137B": "instruction fine-tune of LaMDA-PT (data = the base model's pretraining)",
    "Llama Nemotron Ultra 253B": "derived from Llama 3.1-405B by distillation and post-training",
    "Switch": "mixture-of-experts model: 1.57T parameters, but few are active per token",
}

# Okabe-Ito colours: distinguishable with common colour vision deficiencies
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREY = "#999999"

P_COL = "Parameter count (as recorded)"
D_COL = "Training data size (as recorded)"
C_COL = "Training compute (FLOP, as recorded)"
Y_COL = "Release year"

# --------------------------------------------------
# 1. Page setup
# --------------------------------------------------
st.set_page_config(
    page_title="AI Model Scaling Explorer",
    page_icon="📊",
    layout="wide"
)

st.title("AI Model Scaling Explorer")
st.markdown(f"#### {QUESTION}")
st.markdown(
    f"Built for **{PERSONA}** who need to know whether growth in training "
    "compute has come mainly from larger models or from more training data."
)

# --------------------------------------------------
# 2. Load data
# --------------------------------------------------
DATA_FILE = "member3_language_models_cleaned.csv"
APP_DIR = Path(__file__).parent
CANDIDATE_PATHS = [
    APP_DIR / "data" / "processed" / DATA_FILE,
    APP_DIR.parent / "data" / "processed" / DATA_FILE,  # if app sits in pages/
    APP_DIR / DATA_FILE,
]
DATA_PATH = next((p for p in CANDIDATE_PATHS if p.exists()), CANDIDATE_PATHS[0])


@st.cache_data
def load_data(path):
    raw = pd.read_csv(path)
    n_raw = len(raw)
    df = raw.copy()

    for col in [Y_COL, P_COL, D_COL, C_COL]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Model", Y_COL, P_COL, D_COL])
    df = df[(df[P_COL] > 0) & (df[D_COL] > 0)].copy()
    df[Y_COL] = df[Y_COL].astype(int)
    df["Fresh dense pretraining"] = ~df["Model"].isin(NOT_FRESH_DENSE)

    return df, n_raw


def compact(x):
    """Readable large numbers: 1.5e9 -> '1.5B'."""
    for value, suffix in [(1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")]:
        if abs(x) >= value:
            return f"{x / value:.3g}{suffix}"
    return f"{x:.3g}"


def log_slope(d, col):
    """Slope of log10(value) against release year (least squares)."""
    slope, _ = np.polyfit(
        d[Y_COL].to_numpy(float), np.log10(d[col].to_numpy(float)), 1
    )
    return slope


def verdict(share_models):
    if abs(share_models - 0.5) <= 0.05:
        return "the two contributed roughly equally"
    if share_models > 0.5:
        return "bigger models contributed more"
    return "more data contributed more"


try:
    df, n_raw = load_data(str(DATA_PATH))
except FileNotFoundError:
    st.error(
        f"CSV file not found. Put '{DATA_FILE}' in data/processed/ "
        "(or next to app.py)."
    )
    st.stop()

if df.empty:
    st.error("No valid model records were found in the CSV.")
    st.stop()

has_compute = C_COL in df.columns

st.caption(
    f"Source: [{DATA_SOURCE_NAME}]({DATA_SOURCE_URL}) · "
    f"Data downloaded: {DATA_DOWNLOADED} · "
    f"Processed file: data/processed/{DATA_FILE} · "
    f"{len(df)} of {n_raw} models in the file have both a parameter count "
    "and a training-data size."
)

# --------------------------------------------------
# 3. Sidebar filters
# --------------------------------------------------
st.sidebar.header("Filter models")

min_year = int(df[Y_COL].min())
max_year = int(df[Y_COL].max())

if min_year < max_year:
    year_range = st.sidebar.slider(
        "Release-year range",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1
    )
else:
    year_range = (min_year, max_year)

confidence_options = sorted(df["Source confidence"].dropna().unique().tolist())

selected_confidence = st.sidebar.multiselect(
    "Source confidence",
    options=confidence_options,
    default=confidence_options
)

n_flagged = int((~df["Fresh dense pretraining"]).sum())
exclude_flagged = st.sidebar.checkbox(
    f"Exclude fine-tuned and mixture-of-experts models ({n_flagged})",
    value=True,
    help="Their recorded training data is not a full pretraining run of a "
         "dense model, so they distort the growth estimates. "
         "See 'What this data cannot show'."
)

filtered_df = df[df[Y_COL].between(year_range[0], year_range[1])]

if selected_confidence:
    filtered_df = filtered_df[
        filtered_df["Source confidence"].isin(selected_confidence)
    ]
else:
    filtered_df = filtered_df.iloc[0:0]

# Models used in the answer (flagged ones optionally removed)
answer_df = (
    filtered_df[filtered_df["Fresh dense pretraining"]]
    if exclude_flagged else filtered_df
)

# --------------------------------------------------
# 4. Summary metrics
# --------------------------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Models displayed", len(filtered_df))

if not filtered_df.empty:
    col2.metric("Earliest release year", int(filtered_df[Y_COL].min()))
    col3.metric("Latest release year", int(filtered_df[Y_COL].max()))
else:
    col2.metric("Earliest release year", "—")
    col3.metric("Latest release year", "—")

if filtered_df.empty:
    st.warning(
        "No models match these filters. Change the release-year range "
        "or select at least one source-confidence category."
    )
    st.stop()

# --------------------------------------------------
# 5. The answer
# --------------------------------------------------
st.divider()
st.subheader("The answer for the models selected")

can_fit = answer_df[Y_COL].nunique() >= 2 and len(answer_df) >= 3

if not can_fit:
    st.info(
        "Select at least two release years (and three or more models) to "
        "estimate how fast parameters and training data grew."
    )
else:
    a0, a1 = int(answer_df[Y_COL].min()), int(answer_df[Y_COL].max())
    s_p = log_slope(answer_df, P_COL)
    s_d = log_slope(answer_df, D_COL)

    splits = []  # (label, share from bigger models)

    # Method 1: recorded compute. Tokens processed = compute / (6 x params),
    # so log(compute) growth = log(params) growth + log(tokens processed) growth.
    s_c = None
    if has_compute and answer_df[C_COL].gt(0).all():
        s_c = log_slope(answer_df, C_COL)
        if s_c > 0 and 0 < s_p < s_c:
            splits.append(("Data = tokens processed\n(from recorded compute)",
                           s_p / s_c))

    # Method 2: dataset size, using compute ≈ 6 x params x dataset size
    if s_p > 0 and s_d > 0:
        splits.append(("Data = training-dataset\nsize (as recorded)",
                       s_p / (s_p + s_d)))

    text = (
        f"Across **{len(answer_df)} language models released {a0}–{a1}**"
        + (" (fine-tuned and mixture-of-experts models excluded)"
           if exclude_flagged else "")
        + f", parameter counts grew about **{10 ** s_p:.1f}× per year**"
    )
    if s_c is not None:
        text += (
            f", recorded training compute about **{10 ** s_c:.1f}× per year**, "
            f"and the number of tokens processed in training about "
            f"**{10 ** (s_c - s_p):.1f}× per year**"
        )
    text += (
        f". Training-dataset size grew about **{10 ** s_d:.1f}× per year** "
        "(all rates are trends fitted on a log scale)."
    )
    st.markdown(text)

    if splits:
        lines = []
        for label, share in splits:
            name = label.replace("\n", " ").replace("Data = ", "")
            lines.append(
                f"- Measuring data as *{name}*: **{share:.0%}** of compute "
                f"growth came from bigger models and **{1 - share:.0%}** from "
                f"more data, so {verdict(share)}."
            )
        st.markdown("\n".join(lines))

        if len(splits) == 2 and abs(splits[0][1] - splits[1][1]) > 0.1:
            st.markdown(
                "**The answer depends on how 'more data' is measured.** "
                "Datasets grew fast, but early models such as GPT-2 and "
                "RoBERTa made many passes over small datasets, while recent "
                "models make about one pass over very large ones. So the "
                "tokens actually processed grew more slowly than dataset size"
                + (", and most of the growth in compute came from bigger models."
                   if splits[0][1] > 0.55 else ".")
            )

        fig_s, ax_s = plt.subplots(figsize=(11, 0.9 + 0.75 * len(splits)))
        for i, (label, share) in enumerate(splits):
            ax_s.barh(i, share * 100, color=BLUE)
            ax_s.barh(i, (1 - share) * 100, left=share * 100, color=ORANGE)
            ax_s.text(share * 50, i, f"Bigger models {share:.0%}",
                      ha="center", va="center", color="white",
                      fontweight="bold")
            ax_s.text(share * 100 + (1 - share) * 50, i,
                      f"More data {1 - share:.0%}",
                      ha="center", va="center", color="white",
                      fontweight="bold")
        ax_s.set_yticks(range(len(splits)))
        ax_s.set_yticklabels([lab for lab, _ in splits], fontsize=9)
        ax_s.invert_yaxis()
        ax_s.set_xlim(0, 100)
        ax_s.set_xlabel("Share of growth in training compute (%)")
        ax_s.set_title(f"Where the growth in training compute came from, {a0}–{a1}")
        for side in ["top", "right"]:
            ax_s.spines[side].set_visible(False)
        fig_s.tight_layout()
        st.pyplot(fig_s)
        plt.close(fig_s)
        st.caption(
            "Shares of the yearly growth in training compute (on a log scale). "
            "Top bar: uses Epoch AI's recorded compute and treats compute ÷ "
            "(6 × parameters) as tokens processed. Bottom bar: assumes "
            "compute ≈ 6 × parameters × dataset size."
        )
    else:
        st.markdown(
            "In this selection parameters or data did not grow, so a split "
            "of compute growth is not meaningful."
        )

# --------------------------------------------------
# 6. Change over time: yearly medians
# --------------------------------------------------
if answer_df[Y_COL].nunique() >= 2:
    st.subheader("How model size and training data changed over time")

    yearly = (
        answer_df.groupby(Y_COL)
        .agg(models=("Model", "count"),
             median_params=(P_COL, "median"),
             median_data=(D_COL, "median"))
        .reset_index()
    )

    fig_t, (ax_p, ax_d) = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for ax, col, colour, ylabel, title in [
        (ax_p, "median_params", BLUE, "Median parameters (log scale)",
         "Median parameter count by release year"),
        (ax_d, "median_data", ORANGE,
         f"Median training-dataset size ({TRAINING_UNIT}, log scale)",
         "Median training-dataset size by release year"),
    ]:
        ax.plot(yearly[Y_COL], yearly[col], marker="o", color=colour)
        for _, r in yearly.iterrows():
            ax.annotate(compact(r[col]), (r[Y_COL], r[col]),
                        xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=8)
        ax.set_yscale("log")
        ax.margins(y=0.15)
        ax.set_xlabel("Release year")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(True, which="major", linestyle="--", alpha=0.3)
        ax.xaxis.get_major_locator().set_params(integer=True)
    fig_t.tight_layout()
    st.pyplot(fig_t)
    plt.close(fig_t)

    st.caption(
        "Each point is the median of the models released that year "
        "(same models as the answer above). Models per year: "
        + ", ".join(f"{int(y)}: {int(n)}" for y, n in
                    zip(yearly[Y_COL], yearly["models"]))
        + ". Years with one or two models can swing a lot."
    )

# --------------------------------------------------
# 7. Model selector
# --------------------------------------------------
st.divider()
st.subheader("Explore and compare models")

model_options = sorted(filtered_df["Model"].unique().tolist())

selected_model = st.selectbox(
    "Select a model to highlight on the chart",
    options=model_options,
    index=0,
    key="model_selector"
)

selected_row = filtered_df[filtered_df["Model"] == selected_model].iloc[0]

# --------------------------------------------------
# 8. Scatter plot with highlighted model
# --------------------------------------------------
st.subheader("Parameter count vs. training-dataset size")

fig, ax = plt.subplots(figsize=(11, 6))

others = filtered_df[filtered_df["Model"] != selected_model]
used = others[others["Fresh dense pretraining"]] if exclude_flagged else others
flagged = others[~others["Fresh dense pretraining"]] if exclude_flagged else others.iloc[0:0]

ax.scatter(used[P_COL], used[D_COL], alpha=0.7, s=65, color=BLUE,
           label="Models used in the answer")
if not flagged.empty:
    ax.scatter(flagged[P_COL], flagged[D_COL], s=65, marker="x", color=GREY,
               label="Fine-tuned / mixture-of-experts (excluded)")

ax.scatter(
    selected_row[P_COL],
    selected_row[D_COL],
    s=260,
    marker="*",
    color=ORANGE,
    edgecolors="black",
    linewidths=1.2,
    label=f"Selected: {selected_model}",
    zorder=5
)

ax.annotate(
    f"{selected_model}\n{compact(selected_row[P_COL])} parameters, "
    f"{compact(selected_row[D_COL])} {TRAINING_UNIT}",
    (selected_row[P_COL], selected_row[D_COL]),
    xytext=(10, 10),
    textcoords="offset points",
    fontsize=9,
    fontweight="bold",
    bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": GREY, "alpha": 0.9},
    arrowprops={"arrowstyle": "->", "linewidth": 0.8},
    zorder=6
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("Parameter count (number of parameters, log scale)")
ax.set_ylabel(f"Training-dataset size ({TRAINING_UNIT}, log scale)")
ax.set_title(
    f"Parameter count vs. training-dataset size, "
    f"{int(filtered_df[Y_COL].min())}–{int(filtered_df[Y_COL].max())}: "
    "selected model highlighted"
)

ax.grid(True, which="both", linestyle="--", alpha=0.25)
ax.legend()
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

st.caption(
    "Each mark is one model; the star is the selected model. Grey crosses are "
    "models excluded from the answer. Both axes use logarithmic scales because "
    "model sizes span a wide range. The chart shows association, not causation."
)

# --------------------------------------------------
# 9. Selected model details
# --------------------------------------------------
st.divider()
st.subheader(f"Verify a model: {selected_model}")

p1, p2, p3, p4 = st.columns(4)

p1.metric("Parameter count", f"{selected_row[P_COL]:,.0f}")
p2.metric(f"Training-dataset size ({TRAINING_UNIT})", f"{selected_row[D_COL]:,.0f}")
if has_compute and pd.notna(selected_row[C_COL]):
    p3.metric("Training compute (FLOP)", f"{selected_row[C_COL]:.3g}")
p4.metric("Release year", str(int(selected_row[Y_COL])))

st.write("**Source confidence:**", selected_row["Source confidence"])

if selected_model in NOT_FRESH_DENSE:
    st.info(f"Excluded from the answer by default: {NOT_FRESH_DENSE[selected_model]}.")

source_url = selected_row.get("Source URL", "")
if pd.notna(source_url) and str(source_url).strip():
    first_url = str(source_url).split(";")[0].strip()
    st.markdown(f"**Source reference:** [Open original source]({first_url})")

parameter_notes = selected_row.get("Parameter-count notes", "")
dataset_notes = selected_row.get("Dataset-size notes", "")

with st.expander("Parameter-count notes"):
    if pd.notna(parameter_notes) and str(parameter_notes).strip():
        st.write(parameter_notes)
    else:
        st.info("No parameter-count notes are recorded for this model.")

with st.expander("Training-data notes"):
    if pd.notna(dataset_notes) and str(dataset_notes).strip():
        st.write(dataset_notes)
    else:
        st.warning(
            "No training-data notes are recorded. Verify the value "
            "against the original source before drawing conclusions."
        )

# --------------------------------------------------
# 10. Filtered model table
# --------------------------------------------------
st.divider()
st.subheader("Model records")

table = filtered_df.sort_values([Y_COL, "Model"])
table_out = pd.DataFrame({
    "Model": table["Model"],
    "Release year": table[Y_COL].astype(str),
    "Parameters (count)": table[P_COL].map(lambda v: f"{v:,.0f}"),
    f"Training dataset ({TRAINING_UNIT})": table[D_COL].map(lambda v: f"{v:,.0f}"),
    "Training compute (FLOP)": (
        table[C_COL].map(lambda v: f"{v:.3g}" if pd.notna(v) else "—")
        if has_compute else "—"
    ),
    "Source confidence": table["Source confidence"],
    "Used in answer": table["Fresh dense pretraining"].map(
        lambda v: "Yes" if (v or not exclude_flagged) else "No"
    ),
})
st.dataframe(table_out, hide_index=True)

# --------------------------------------------------
# 11. Data limitations
# --------------------------------------------------
with st.expander("What this data cannot show", expanded=True):
    flagged_list = "; ".join(f"**{m}**: {why}" for m, why in NOT_FRESH_DENSE.items())
    st.markdown(
        f"""
        - **This is a small sample.** {len(df)} frontier language models (top 10 by compute when released);
          some years have only one or two models, and models that did
          not publish their training-data size are missing.
        - **Excluded models (by default).** {flagged_list}.
        - **"More data" can be measured two ways, and they disagree.** Dataset
          size counts the unique text available; tokens processed counts every
          pass over it. Early models repeated their data many times, so the
          two measures grew at different rates.
        - **Training-data values are often estimates.** For example, GPT-2's
          token count is estimated from the reported size of its WebText
          corpus (about 40 GB of text), not published by OpenAI.
        - **Compute is also recorded or estimated by Epoch AI**, and the
          6 × parameters × tokens rule only holds for dense transformer models.
        - Source confidence describes uncertainty in the recorded value; it
          does not guarantee the value is exact.
        - The relationships shown are associations and do not prove causation.
        """
    )
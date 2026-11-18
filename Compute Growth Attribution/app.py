"""Compute Growth Attribution. Run: streamlit run app.py"""
import altair as alt
import pandas as pd
import streamlit as st

import prepare_data as prep

st.set_page_config(page_title="Compute Growth Attribution", layout="wide")

if not prep.fetch_if_missing():
    st.error("Could not download the data. Save https://epoch.ai/data/all_ai_models.csv in data/raw/ (see README.md).")
    st.stop()


@st.cache_data
def load():
    usable, info = prep.load()
    return usable, info, pd.read_csv(prep.RAW, dtype=str)


try:
    usable, info, raw = load()
except ValueError as e:
    st.error(f"The data file is not in the expected format: {e}")
    st.stop()


def interval_chart(res, label, height):
    base = alt.Chart(res).encode(y=alt.Y(f"{label}:N", sort=None, title=None))
    return (
        base.mark_rule(strokeWidth=3, color="#1B4F72").encode(
            x=alt.X("lo:Q", title="Share of growth from larger models", axis=alt.Axis(format="%")), x2="hi:Q")
        + base.mark_point(size=160, filled=True, color="#1B4F72").encode(
            x="share:Q", tooltip=[label, alt.Tooltip("share:Q", format=".0%"), "n"])
        + alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(strokeDash=[4, 4], color="grey").encode(x="x:Q")
    ).properties(height=height)


st.title("Compute Growth Attribution")
st.write("**As AI training compute grew, how much of that growth came from larger models versus more training data?**")
st.caption("Audience: Technology policy analysts and AI researchers.")

c1, c2 = st.columns(2)
first, last = int(usable["year"].min()), int(usable["year"].max())
start, end = c1.slider("Years", max(first, 2010), last, (max(first, 2012), last))
if end - start < 2:
    st.warning("Choose at least three years apart.")
    st.stop()
split = c2.number_input("Compare before and after this year", start + 1, end, min(max(2020, start + 1), end))
if split - start < 2 or end - split < 2:
    st.warning("Split must leave at least two years on each side.")
    st.stop()

whole = prep.decompose(usable, start, end)
if whole is None:
    st.warning("Too few models (need 30) or no growth in compute. Widen the years.")
    st.stop()
if not (0 <= whole["share_params"] <= 1):
    st.warning("Estimated share is outside [0, 1]; widen the years.")
    st.stop()

st.subheader(
    f"{start} to {end}: about {whole['share_params']:.0%} of compute growth came from larger models "
    f"and {whole['share_tokens']:.0%} from more training data."
)
st.write(
    f"Compute grew about {whole['growth_per_year']:.1f}x a year. "
    f"95% range for the larger-models share: {whole['lo']:.0%} to {whole['hi']:.0%} ({whole['n']:,} models)."
)

st.subheader("Did the split change over time?")
rows = []
for label, a, b in [(f"{start}-{split - 1}", start, split - 1), (f"{split}-{end}", split, end), (f"{start}-{end}", start, end)]:
    r = prep.decompose(usable, a, b)
    if r:
        rows.append({"period": label, "share": r["share_params"], "lo": r["lo"], "hi": r["hi"], "n": r["n"]})
if rows:
    st.altair_chart(interval_chart(pd.DataFrame(rows), "period", 200), width="stretch")
    st.caption("Dot = estimate. Line = 95% range. Dashed line = even split.")
    expected = [f"{start}-{split - 1}", f"{split}-{end}", f"{start}-{end}"]
    missing_p = [p for p in expected if p not in {r["period"] for r in rows}]
    if missing_p:
        st.caption("Dropped (too few models): " + ", ".join(missing_p))

st.subheader("Does the answer change with other choices?")
sc = prep.scenarios(usable, start, end)
if sc.empty:
    st.info("No scenario had enough models.")
else:
    st.altair_chart(interval_chart(sc, "scenario", 40 * len(sc) + 60), width="stretch")
    st.caption("If the dots agree, the answer is robust.")
    all_scenarios = ["All models", "Developer-reported or hardware compute only", "Language models only", "Last five years only"]
    missing = [s for s in all_scenarios if s not in set(sc["scenario"])]
    if missing:
        st.caption("Dropped (too few models): " + ", ".join(missing))

st.subheader("Trace a number")
st.write(
    f"Compute slope **{whole['slope_compute']:.3f}** = parameters **{whole['slope_params']:.3f}** + data **{whole['slope_tokens']:.3f}** "
    f"(orders of magnitude a year). Larger-models share = {whole['slope_params']:.3f} / {whole['slope_compute']:.3f} = **{whole['share_params']:.1%}**."
)
w = usable[(usable["year"] >= start) & (usable["year"] < end + 1)].sort_values("compute", ascending=False)
pick = st.selectbox(
    "Pick a model",
    w["raw_idx"],
    format_func=lambda r: f"row {int(w.loc[w['raw_idx'] == r, 'raw_row'].iloc[0])}: {w.loc[w['raw_idx'] == r, 'model'].iloc[0]}",
)
m = w[w["raw_idx"] == pick].iloc[0]
st.write(f"Data per model = compute / (6 x parameters) = {m['compute']:.3e} / (6 x {m['params']:.3e}) = **{m['tokens']:.3e}**")
keep = [prep.COLUMNS[k] for k in ("model", "date", "params", "compute") if prep.COLUMNS[k] in raw.columns]
st.caption(f"Same model in the original file, spreadsheet row {int(m['raw_row'])}:")
st.dataframe(raw.iloc[int(m["raw_idx"])][keep].rename("value").to_frame(), width="stretch")

with st.expander("Data and method"):
    st.markdown(
        f"""
**Data:** Epoch AI, *Data on AI models* (CC BY 4.0), https://epoch.ai/data/all_ai_models.csv. SHA-256 `{info['sha256']}`. Latest model date {info['latest_date']}.

**Sample:** {info['rows_in_file']:,} models in the file. {info['rows_with_compute']:,} report compute and {info['rows_with_params']:,} report parameters.
**{info['rows_usable']:,} have both and are used.**

**Method:** data per model = compute / (6 x parameters). Take base-10 logs of compute, parameters and data, and fit a trend line against date for each.
The slopes add up exactly, so share from larger models = parameters slope / compute slope. The range comes from resampling the models 1,000 times.

**Limits:** the 6 x N x D rule is approximate. Only models that report both numbers are used, so the biggest undisclosed models are missing.
Some compute values are Epoch's estimates. Compute is a cost, not capability.
"""
    )
# AI Training Computation

A Streamlit app that looks at Epoch AI's public database of AI models and asks where the growth in AI training scale came from. It brings together four team analyses, called lenses, on one shared data package. Another lens can be added without touching the others (see "Adding another lens").

## Research question

As AI training compute grew exponentially, how much of that growth stemmed from larger models versus expanded training datasets?

This is a fair thing to argue about. The same data gives different answers depending on how "more data" is counted (the size of the dataset, or the tokens a model actually processed) and on which models you include. The app shows that on screen instead of hiding it.

## Persona and the decision it serves

**Maya, a technology policy analyst** at a government AI office. She has to recommend where limited attention should go: tracking and shaping the cost and supply of the machinery needed to build bigger models, or tracking and shaping access to the training data those models learn from. She does not need to run models herself. She needs a defensible, sourced answer to "which of the two has been growing faster?" and a clear idea of how far to trust it.

## Three user stories

1. As a technology policy analyst, I want to see one plain answer and every estimate behind it on the opening screen, so that I can judge at a glance how settled the answer is.
2. As a technology policy analyst, I want to filter a lens to the models, years or sources I care about and see how the split changes, so that I can check whether the answer holds for the area I work on.
3. As a technology policy analyst, I want to trace any number back to a model's source row in the raw file, so that I can cite it in a briefing without worrying about hand typed figures.

### Walking through each story in the running app

**Story 1.** Open the app. The question and a one sentence answer sit at the top, built from every estimate in the app. Scroll to "Every estimate in one picture". Each dot is one estimate of the share of growth that came from bigger models, with its 95% range, and the dashed line marks an even split. Blue dots count data as dataset size and red diamonds as tokens actually processed. If the dots sit on both sides of the line, the answer is open to argument.

**Story 2.** Open the tab "Lens 1: all models, dataset size". Change Domain (for example to Vision), narrow the Release year slider, or pick an Organization. The blue box rewrites itself for your selection, a new top bar called "Your current selection" appears, and the charts redraw. Open "Lens 2" and move the Years slider or the comparison year to see whether the split changed before and after. Open "Lens 3", untick the box that excludes fine tuned and mixture of experts models, and watch the two estimates move. Open "Lens 4", type part of a model name in Search model or pick a domain, and watch the correlations and the missing data bars change. Pick a combination with no models (for example OpenAI and Biology in lens 1) to see the empty state message.

**Story 3.** In lens 2, scroll to "Trace a number": the yearly slopes are printed, the share is worked out from them, and you can pick any model to see compute ÷ (6 × parameters) with its raw file row number and source link. In lens 4, "Explore a model" shows any model's recorded numbers and notes. In lens 1 and lens 3, the verification tables list every point behind the charts. Click a row to open its source record, with Epoch's notes and a link to the paper. The download buttons save the exact rows behind the charts. Sources and the download date are on the opening screen and again in the "Limits, sources and cleaning" tab.

## The three lenses

* **Lens 1, all models and dataset size** (Member A). Every model that reports parameters and a dataset size. Filters, a part to whole chart of the split by slice, trend lines, a scatter plot, a ranking of organizations and a verification table.
* **Lens 2, compute and tokens processed** (Member B, from the app "Compute Growth Attribution"). Every model that reports compute and parameters. It works out the data each model processed as compute ÷ (6 × parameters), gives 95% ranges, compares before and after a year you choose, tests four sample choices and lets you trace one number to its raw row.
* **Lens 3, frontier language models** (Member C, from the app "Model Scaling Explorer"). Epoch's frontier language models since 2019, with both ways of counting data side by side, a flag for fine tuned and mixture of experts models, a highlight and verify panel for any model, and a chart of implied passes over the data that helps explain why the two measures can disagree.
* **Lens 4, what moves with compute** (Member D, from the app "Data on AI models"). A different question: across models, does training compute rise more closely with parameters or with dataset size (Spearman rank correlation)? It adds a search box, a chart of compute over time, two scatter plots of compute against each measure, and a chart of how many models report each number. In the merge the two correlations are compared on the same models and the gap gets a 95% range, and the on screen verdict is worked out from the numbers instead of being fixed text.

All four now read the same Epoch snapshot through one pipeline, so their numbers can be compared. Before the merge each member had downloaded a slightly different copy (the four copies had between 3,628 and 3,728 rows), and the newest one is the one used.

## What is on the page

* **Opening screen:** the question, a one sentence answer computed from the data, a cross check from lens 4, the persona, sources and refresh date, five facts about the whole snapshot (not a filtered slice), and a chart of every estimate.
* **Charts:** a dot and range chart (deviation from an even split), 100% stacked bars (part to whole), line charts of yearly medians (change over time), scatter plots (correlation), a bar chart of organizations (ranking), stacked bars of reported against missing numbers (part to whole), and a line chart of implied passes.
* **Limits:** a visible section called "What this data cannot show".

Every chart has a title above it and a caption below it. Every axis names its unit. Points, bars and lines carry value labels where they fit, and every point on the scatter plots is also in a table, so nothing depends on hovering.

## How the answer is worked out

A common rule of thumb says training compute is about 6 × parameters × data. In base 10 logs that is a sum, so the yearly growth of compute is the yearly growth of parameters plus the yearly growth of data, and each part divided by the total is its share. Growth is the slope of a straight line fitted to the logged values against release date. The 95% range comes from resampling the models 1,000 times. If either part did not rise, no share is given. The code is in `src/attribution.py`.

## Run it

Python 3.11 is needed.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app reads only the small processed files in `data/processed/`, so it starts quickly. To rebuild them from the raw file:

```bash
python -m src.data
```

To check that the pipeline also works on the small source sample:

```bash
python -m src.data --raw data/samples/all_ai_models_sample.csv --out /tmp/sample_out
```

Run the tests (install pytest first with `pip install pytest`):

```bash
pytest
```

## Deploy

The simplest route is Streamlit Community Cloud. Push the repository to GitHub, choose the repository and `app.py`, and pick Python 3.11. Put the deployed link and the submitted commit hash or tag here before you hand in:

* Deployed URL: [add link]
* Submitted commit or tag: [add hash or tag]

Open the link in a private window with the network tab of your browser's developer tools open and check that it loads in under five seconds. Record the result in `PROVENANCE.md`.

## Data package

* `data/raw/all_ai_models.csv`: the full download from Epoch AI, untouched.
* `data/samples/all_ai_models_sample.csv`: every 30th row of the raw file, same layout, for quick pipeline checks.
* `data/curation/mixture_of_experts.csv`: the one hand made input, naming the mixture of experts model set aside in lens 3, with the evidence.
* `data/processed/`: the seven files the app reads.
* `data/source.json`: citation, licence and download date shown on screen.
* `DATA_DICTIONARY.md`: source, URL, licence, columns, units, key, coverage and transform for every file.
* `PROVENANCE.md`: who checked what, and which checks could have failed.
* `AI_DECLARATION.md`: how AI was used.

## Colours

Bars and lines use blue and orange or red. The scatter plot uses the Okabe Ito palette plus black, designed for colour blind readers, and each category also gets its own marker shape. As a first check the palette was run through a simulation of red green and blue yellow colour blindness, and the closest pair of colours still differed clearly. Before you hand in, take a screenshot and run it through Coblis or Color Oracle as well, and note the result in `PROVENANCE.md`.

## Limits in short

Only models that report the needed numbers appear. Dataset size is counted in different units across domains and measures the size of the data, not how many times it was read. The split depends on a rule of thumb. The frontier sample is small. Lens 4 shows association, not growth. The full list is on the page under "What this data cannot show".

## Layout

The app uses the wide layout and scrolls vertically only. Check it once in your browser's responsive mode at 1366 by 768.

## Adding another lens

1. Put the member's raw or cleaned input through `src/data.py`, so the processed file lands in `data/processed/` and is listed in `OUTPUT_FILES`.
2. Write `src/views_<name>.py` with one `render_<name>(...)` function. Use the helpers in `src/views_common.py` for charts, tables and downloads, and give every widget a key with a new prefix such as `l4_`.
3. If the lens produces a share of growth, add its estimates to `answer_summary()` in `src/attribution.py` so it appears in the opening chart and the headline. If it produces another kind of result, add a small summary table like `association_summary.csv`.
4. Add a tab in `app.py`, a section in `DATA_DICTIONARY.md`, a row in `PROVENANCE.md` and a line in `AI_DECLARATION.md`.

## Files

```
app.py                       opening screen and tabs
src/analysis.py              shared calculations (pure pandas)
src/attribution.py           the growth split, ranges and the headline sentence
src/association.py           lens 4 rank correlations and their gap
src/data.py                  the pipeline from raw file to processed files
src/views_common.py          shared charts, tables, downloads
src/views_overview.py        opening screen, limits, sources
src/views_all_models.py      lens 1
src/views_compute.py         lens 2
src/views_frontier.py        lens 3
src/views_association.py     lens 4
data/                        raw, samples, curation, processed, source.json
tests/                       analysis, attribution, association, pipeline and app tests
DATA_DICTIONARY.md  PROVENANCE.md  AI_DECLARATION.md
```

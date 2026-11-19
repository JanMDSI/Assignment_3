<<<<<<< HEAD
# AI Model Scaling Explorer

Part of *The Rise of Artificial Intelligence: Data Narrative Studio*. It is a Streamlit app that looks at the models in Epoch AI's public database and asks where the growth in AI training scale came from.

## Research question

As AI training compute grew exponentially, how much of that growth stemmed from larger models versus expanded training datasets?

This is a fair thing to argue about. The same data gives a different split depending on which models you count (all domains or only language, early years or recent ones), and the app shows that on screen instead of hiding it.

## Persona and the decision it serves

**Maya, a technology policy analyst** at a government AI office. She has to recommend where limited attention should go: tracking and shaping the cost and supply of the machinery needed to build bigger models, or tracking and shaping access to the training data those models learn from. She does not need to run the models herself. She needs a defensible, sourced answer to "which of the two has been growing faster?" and a clear idea of how far to trust it.

## Three user stories

1. As a technology policy analyst, I want to see how the growth in parameters times training data splits between model size and training data across several slices of the data, so that I can judge whether the answer holds up or depends on which models I count.
2. As a technology policy analyst, I want to filter to one domain, organization or span of years and see the split and the trends for just those models, so that I can check whether the pattern also holds for the area I work on.
3. As a technology policy analyst, I want to trace any number back to a model's source record and cite it, so that I can use it in a briefing without worrying about hand typed figures.

### Walking through each story in the running app

**Story 1.** Open the app. The question and a plain sentence answer sit at the top. Scroll to section 1, "Who gets the credit for the growth?". Each bar is a different slice of the data, split into the share from model size (blue) and from training data (orange). Compare the bars to see how steady the answer is.

**Story 2.** In the sidebar, change Domain (for example to Vision), narrow the Release year slider, or pick an Organization. The blue box under the question rewrites itself for your selection, a new top bar called "Your current selection" appears in section 1, and the line charts, scatter plot and ranking all redraw. Pick a combination with no models (for example OpenAI and Biology) to see the empty state message.

**Story 3.** Scroll to the verification table. Sort it, click a row, and the source record opens below it with Epoch's notes and a link to the paper. The "Numbers behind these two charts" expander shows every yearly median. The "Download the selected models" button saves exactly the rows behind the charts. Sources and the download date are on the opening screen and again at the bottom.

## What is on the page

* **Opening screen:** the question, a one sentence answer computed from the data, the persona, sources and refresh dates, and four facts about the whole dataset (not a filtered slice).
* **Section 1, part to whole:** 100% stacked bars of the growth split by slice, plus your current selection.
* **Section 2, change over time:** two line charts of the yearly median parameters and median dataset size, with every point labelled.
* **Section 3, correlation:** a scatter plot of parameters against dataset size on logarithmic axes, with a fitted trend line. Categories differ by colour and by marker shape.
* **Section 4, ranking:** a bar chart of the ten organizations with the most models in the selection.
* **Verification table, limits, sources:** a sortable table of every model behind the charts, a visible "What this data cannot show" section, and the cleaning steps.

Every chart has a title above it and a caption below it, every axis names its unit, and bars and lines carry their values as labels so nothing depends on hovering.

## How the answer is worked out

For the models in a slice, the app fits a straight line through log10(parameters) against release year, and another through log10(training dataset size) against release year. A common rule of thumb says training compute is about 6 × parameters × data, so the two yearly slopes add up to the yearly growth of that product, and each slope divided by the sum is that factor's share. If either factor does not rise, no split is shown. The method is descriptive. It does not prove why labs chose the sizes they did.

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
python -m src.data --raw data/samples/epoch_ai_models_sample.csv --out /tmp/sample_out
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

* `data/raw/epoch_ai_models.csv`: the full download from Epoch AI, untouched.
* `data/samples/epoch_ai_models_sample.csv`: every 30th row of the raw file, same layout, for quick pipeline checks.
* `data/processed/`: the three files the app reads (`models.csv`, `growth_by_slice.csv`, `cleaning_steps.csv`).
* `data/source.json`: citation, licence and download date shown on screen.
* `DATA_DICTIONARY.md`: source, URL, licence, columns, units, key, coverage and transform for every file.
* `PROVENANCE.md`: who checked what, and which checks could have failed.
* `AI_DECLARATION.md`: how AI was used.

## Colours

Bars and lines use blue and orange. The scatter plot uses the Okabe Ito palette plus black, designed for colour blind readers, and each category also gets its own marker shape. As a first check, the palette was run through a simulation of red green and blue yellow colour blindness, and the closest pair of colours still differed clearly. Before you hand in, take a screenshot and run it through Coblis or Color Oracle as well, and note the result in `PROVENANCE.md`.

## Limits in short

Only models that report both parameters and dataset size appear. Dataset size is counted in different units across domains. It measures the size of the data, not how many times it was read. The split depends on a rule of thumb. The full list is on the page under "What this data cannot show".

## Layout

The app uses the wide layout and scrolls vertically only. Check it once in your browser's responsive mode at 1366 by 768.

## Files

```
app.py                     page layout and sidebar
src/analysis.py            every calculation (pure pandas)
src/data.py                the pipeline from raw file to processed files
src/views.py               charts, tables and text
data/                      raw, samples, processed, source.json
tests/                     analysis, pipeline and app tests
DATA_DICTIONARY.md  PROVENANCE.md  AI_DECLARATION.md
```
=======
# Assignment_3
>>>>>>> b9a70527a1e37e2aeef9098348744ff9679f316d

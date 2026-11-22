# Data and AI Disclosure

**Project:** The Rise of Artificial Intelligence: Data Narrative Studio
**Component:** AI Scaling Explorer (merged from four team members' lenses)
**Date:** [Day Month Year]

Each member completes their own section. Keep the shared sections as they are.

## 1. Tools used

The team used **Claude (Anthropic)**, a generative AI assistant, while building and merging this component. [Other members: if you used any other tool, name it here.]

## 2. What the AI helped with

* Drafting and rewriting the Python code in `app.py` and `src/`, including the cleaning rules, the filter logic, the growth split with its 95% ranges, and the Plotly charts.
* Merging four members' separate apps (lens 1, "Compute Growth Attribution", "Model Scaling Explorer" and "Data on AI models") onto one data pipeline and one app, and converting their charts to one charting library.
* Drafting the tests, `README.md`, `DATA_DICTIONARY.md`, the provenance matrix template and this statement.
* Suggesting how to handle quirks in the data (dataset sizes stored as text, zero values that cannot sit on a log axis, line breaks inside model names, lists of organizations and domains in one cell).
* Running a simple colour blindness simulation on the chart palette.

## 3. What the AI did not do

* The research question, the persona, the choice of the four lenses, the exclusion list for fine tuned and mixture of experts models, and the idea of comparing compute with parameters and dataset size by correlation came from the team and the assignment brief.
* The AI did not create, estimate or edit any data values. The raw file is the unaltered Epoch AI download. Every number on screen is read from the processed files or calculated from them by code in `src/`.
* The interpretation of results and the presentation are the team's own. [Edit if this is not accurate.]

## 4. Data disclosure

* Source: Epoch AI, "Data on AI Models", https://epoch.ai/data/ai-models, licence CC BY 4.0, downloaded 18 November 2026.
* Derived by code, not by hand: release year, decimal year, domain group, organization sector, tokens processed, the frontier flags, the growth factors and shares with their ranges, and the cleaning step counts. The rules are in `DATA_DICTIONARY.md` and `src/data.py`.
* One hand made input: `data/curation/mixture_of_experts.csv`, which names the mixture of experts model set aside in lens 3 and cites the evidence.
* Rows removed during cleaning are counted on screen in the "Data source and cleaning steps" panel. Nothing was filled in or guessed.

## 5. Each member's verification and responsibility

For each member: say which lens you own, what you ran and checked yourself, and that you can explain your code.

* **Member A (Jan Keane Olajuwon, student ID: 26493320)**: lens 1 (all models, with data counted as dataset size), the shared pipeline `src/data.py` and the merge of the four lenses onto one snapshot, built with Visual Studio and edited in Code Claude (Anthropic) to make some code efficient. I moved the four apps onto the newest Epoch download (18 November 2026) and fixed a `ModuleNotFoundError` for `yaml` on my first local run by switching the citation file to JSON. I traced five random rows of `models.csv` back to their `raw_row` in the raw file and confirmed model name, parameters and dataset size matched in all five. I ran each original app's calculation on the merged raw file and compared it with `answer_summary.csv`: lens 1 and lens 2 were within one percentage point (largest gaps 0.71 and 0.00), and lens 3 tokens processed was 3.02 points apart, which I traced to the original using whole release years while the merge uses decimal years, and I recorded it as a fail with that cause in the provenance matrix. The team confirmed that all seven processed files regenerate identically from the raw file and that the test suite passes (30 of 30 on a clean clone of commit 97aed42). I deployed the app at https://assignment3apptest.streamlit.app/. I can explain my lens's code and calculations.
* **Member B (Qryne Alisya Siera, student ID: 26207023):** lens 2 (compute and tokens processed, from “Compute Growth Attribution”) and the chart and colour checks, built with Visual Studio Code, with DeepSeek for troubleshooting errors, the chart audit, colour blindness script and chart test. I wrote prepare_data.py and app.py, which count data as tokens processed (compute ÷ (6 × parameters)) and split the yearly growth of compute between larger models and more data. Jan later combined them into the shared pipeline. I ran my unchanged code on the newest raw file and recomputed the 2012 to 2026 split in a separate script: 1,192 models, larger models 50.1% of growth, the same as answer_summary.csv. My original, earlier download gave 49.5% across 1,173 models, so that difference comes from the data, not the code. I recomputed compute ÷ (6 × parameters) from the raw file for all 1,272 usable models, and my four sample choices match the merged lens 2. For the charts, I checked all 18 against the rubric points on form, ran the colour script (closest pair 16.1) and a Color Oracle check on a screenshot of the scatter plot, and ran the chart test. I can explain my lens’s code and calculations.
* **Member C** (Abhinandan Sharma, student ID: 26290078): lens 3 (frontier language models, from "Model Scaling Explorer"), the data dictionary and the provenance matrix. I rebuilt my cleaned file from Epoch's raw download with `clean_member3.py` and confirmed it matched column by column; recomputed my lens's growth splits independently with `verify_member3.py`; filtered the merged raw file myself and confirmed lens 3 has the same 31 models; checked all six excluded models against Epoch's notes and the papers; re-ran the pipeline and confirmed all seven processed files regenerate identically; and ran the test suite (30 of 30 passed). I can explain my lens's code and calculations.
* **Member D** (Rohit Khanna, student ID: 25669108): lens 4, the correlations and the missing data view. [What I ran and checked.]

Limits the team is aware of: dataset size is counted in different units across domains, only models that report the needed numbers appear, the frontier sample is small, and the growth split rests on a rule of thumb that compute is about 6 × parameters × data.

*Signed:* [Names] *Date:* [Date]

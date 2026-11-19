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

* **Member A** ([name], [student ID]): lens 1, the pipeline and the merge. [What I ran and checked.]
* **Member B** ([name], [student ID]): lens 2. [What I ran and checked.]
* **Member C** (Abhinandan Sharma, student ID: 26290078): lens 3 (frontier language models, from "Model Scaling Explorer"), the data dictionary and the provenance matrix. I rebuilt my cleaned file from Epoch's raw download with `clean_member3.py` and confirmed it matched column by column; recomputed my lens's growth splits independently with `verify_member3.py`; filtered the merged raw file myself and confirmed lens 3 has the same 31 models; checked all six excluded models against Epoch's notes and the papers; re-ran the pipeline and confirmed all seven processed files regenerate identically; and ran the test suite (30 of 30 passed). I can explain my lens's code and calculations.
* **Member D** ([name], [student ID]): lens 4, the correlations and the missing data view. [What I ran and checked.]

Limits the team is aware of: dataset size is counted in different units across domains, only models that report the needed numbers appear, the frontier sample is small, and the growth split rests on a rule of thumb that compute is about 6 × parameters × data.

*Signed:* [Names] *Date:* [Date]

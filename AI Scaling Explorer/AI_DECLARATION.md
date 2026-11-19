# Data and AI Disclosure

**Project:** The Rise of Artificial Intelligence: Data Narrative Studio
**Component:** AI Model Scaling Explorer
**Author:** Jan Keane Olajuwon | **Student ID:** 26493320 | **Course:** 36104 | **Date:** 5 November 2026

## 1. Tools used

I used **Claude (Anthropic)**, a generative AI assistant, while building and revising this component. No other generative AI tool was used. 

## 2. What the AI helped with

* Drafting and rewriting the Python code in `app.py`, `src/analysis.py`, `src/data.py` and `src/views.py`, including the cleaning rules, the filter logic, the growth split calculation and the Plotly charts.
* Drafting the tests in `tests/`, `README.md`, `DATA_DICTIONARY.md`, the provenance matrix template and this statement.
* Suggesting how to handle quirks in the data (dataset sizes stored as text, zero values that cannot sit on a log axis, lists of organizations and domains in one cell).
* Suggesting the slice comparison chart and the colour palette, and running a simple colour blindness simulation on that palette.

## 3. What the AI did not do

* The research question, the persona and the choice of logarithmic axes came from me and my team and the assignment brief.
* The AI did not create, estimate or edit any data values. The raw file is the unaltered Epoch AI download. Every number on screen is read from the processed files or calculated from them by code in `src/analysis.py`.
* The interpretation of results and my presentation are my own. 

## 4. Data disclosure

* Source: Epoch AI, "Data on AI Models", https://epoch.ai/data/ai-models, licence CC BY 4.0, downloaded 5 November 2026.
* Derived by code, not by hand: release year, domain group, organization sector, the growth factors and shares, and the cleaning step counts. The rules are written in `DATA_DICTIONARY.md` and in `src/data.py`.
* Rows removed during cleaning are counted on screen in the "Data source and cleaning steps" panel. Nothing was filled in or guessed.

## 5. My verification and responsibility

* I ran the app and the tests myself and checked the on screen numbers against the verification table and the source file.
* I read and understand the code and can explain it. I take responsibility for the submitted work, including any errors.
* Limits I am aware of: dataset size is counted in different units across domains, only models that report both numbers appear, and the growth split rests on a rule of thumb that compute is about 6 × parameters × data.

Jan Keane Olajuwon 

5 November 2026

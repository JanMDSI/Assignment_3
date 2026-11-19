# Data dictionary

Every file the project reads or writes is listed here. All four lenses share one raw snapshot. All of it comes from one source, Epoch AI's "Data on AI Models", plus one small hand made curation file.

* Source: Epoch AI, "Data on AI Models"
* URL: https://epoch.ai/data/ai-models (file: https://epoch.ai/data/all_ai_models.csv)
* Licence: CC BY 4.0 (credit Epoch AI)
* Downloaded: 18 November 2026

## 1. data/raw/all_ai_models.csv

* **What it is:** the full download, untouched. One row per model.
* **Primary key:** the row number in the file (header is row 1). Model names can repeat once line breaks are removed. The raw file has 3,728 rows and 57 columns.
* **Time coverage:** publication dates from 2 July 1950 to 17 November 2026. 13 rows have no usable date.
* **Latest observation:** latest publication date 17 November 2026. Latest record change 17 November 2026.
* **Transform applied:** none. This is the input to the pipeline.
* **Columns used by the app** (the other columns stay in the file but are not read):
  * Model: model name, text.
  * Publication date: date of the paper or announcement, YYYY-MM-DD.
  * Organization: comma separated list of organizations, text.
  * Organization categorization: Industry, Academia, Government, or a mix, text.
  * Domain: comma separated list such as Language or Vision, text.
  * Parameters: number of learnable parameters, count.
  * Training dataset size (total): training data size in datapoints, stored as text. Most values are plain or scientific numbers. A few hold two comma separated values.
  * Training compute (FLOP): training compute in floating point operations.
  * Training compute estimation method: how Epoch got the compute figure, text.
  * Frontier model: True when the model was among the ten most compute heavy when released.
  * Confidence: Epoch's rating of the recorded values.
  * Base model: the model this one was built from, blank for models trained from scratch.
  * Link, Parameters notes, Dataset size notes, Last modified: source link, Epoch's notes, timestamp of Epoch's last edit.

## 2. data/samples/all_ai_models_sample.csv

* **What it is:** a small source sample with the same 57 columns, for quick pipeline checks.
* **Primary key:** the row number in the raw file.
* **Time coverage:** publication dates from 30 November 1972 to 17 November 2026. It has 125 rows.
* **Latest observation:** 17 November 2026.
* **Transform applied:** keeps every 30th row of the raw file (`python -m src.data --make-sample`). No values are changed.

## 3. data/curation/mixture_of_experts.csv

* **What it is:** the one hand made input. It names frontier language models that use a mixture of experts design, which the raw file has no column for.
* **Primary key:** model_name.
* **Time coverage and latest observation:** not applicable.
* **Transform applied:** none, written by hand from Epoch's notes and the paper.
* **Columns:** model_name (text), evidence (text with the paper link).

## 4. data/processed/models.csv (lens 1, and the base for lens 3)

* **What it is:** one row per model that reports both parameters and a usable dataset size.
* **Primary key:** raw_row.
* **Time coverage:** release years 1950 to 2026, publication dates 2 July 1950 to 5 November 2026.
* **Latest observation:** latest publication date 5 November 2026. Latest record change 17 November 2026.
* **Transform applied** (in `build_models()` in `src/data.py`): keep rows with parameters above zero, a dataset size that parses to one positive number, and a valid publication date. Equal pairs of dataset sizes are collapsed to one value, conflicting pairs are dropped. Line breaks in names become spaces. Organizations and domains are tidied into lists. Release year, decimal year and a single domain group are derived.
* **Columns and units:**
  * raw_row: spreadsheet row number in the raw file.
  * model_name: text.
  * release_year: calendar year, whole number. decimal_year: year plus the fraction of the year passed, for example 2020.5.
  * publication_date: YYYY-MM-DD.
  * organization: organizations separated by "; ". organization_sector: Industry, Academia, Government, Research collective, Mixed or collaboration, or Not specified.
  * domain: domains separated by ", ". domain_group: the single domain, or "Multiple domains", or Not specified.
  * parameters: count of learnable parameters.
  * training_dataset_size: datapoints (mostly tokens for language models, images or other items elsewhere).
  * training_compute_flops: floating point operations, blank when not reported.
  * compute_method, confidence: text, "Not recorded" when blank.
  * frontier_model: True or False. base_model: text, blank when none.
  * source_link, parameters_notes, dataset_size_notes: text from Epoch, may be blank. last_modified: timestamp of Epoch's last edit.

## 5. data/processed/compute_models.csv (lens 2)

* **What it is:** one row per model with a valid date, positive parameters and positive training compute. Dataset size is not needed.
* **Primary key:** raw_row.
* **Time coverage:** release years 1950 to 2026, publication dates 2 July 1950 to 5 November 2026.
* **Latest observation:** latest publication date 5 November 2026. Latest record change 17 November 2026.
* **Transform applied** (in `build_compute_models()`): keep rows with a valid date, compute above zero and parameters above zero. tokens_processed is derived as compute ÷ (6 × parameters), and the pipeline checks that equation on every row.
* **Columns and units:** raw_row, model_name, release_year, decimal_year, publication_date, domain (as in models.csv); parameters (count); training_compute_flops (floating point operations); tokens_processed (tokens, derived); compute_method (text); frontier_model (True or False); source_link, last_modified (text).

## 6. data/processed/frontier_language_models.csv (lens 3)

* **What it is:** frontier language models released 2019 or later, taken from models.csv, with a flag for models that are not a fresh dense training run.
* **Primary key:** raw_row.
* **Time coverage:** publication dates 14 February 2019 to 18 March 2025.
* **Latest observation:** 18 March 2025. Latest record change 12 November 2026.
* **Transform applied** (in `build_frontier()`): keep rows from models.csv where frontier_model is True, the domain is exactly Language and the release year is 2019 or later. fresh_dense is False when Epoch lists a base model or the model is in the mixture of experts curation file. exclusion_reason says why.
* **Columns:** every column of models.csv, plus fresh_dense (True or False) and exclusion_reason (text, blank when not excluded).

## 7. data/processed/coverage.csv (lens 4)

* **What it is:** one row per model with a valid publication date, showing which of the three numbers it reports. It is how lens 4 draws the missing data chart and the counts.
* **Primary key:** raw_row.
* **Time coverage:** release years 1950 to 2026.
* **Latest observation:** latest publication date 5 November 2026 among models reporting any of the three numbers, and 17 November 2026 for the file as a whole.
* **Transform applied** (in `build_coverage()`): keep rows with a valid date. A number counts as reported when it is above zero, and for dataset size when it also parses to one value. The pipeline checks that the counts agree with models.csv and compute_models.csv.
* **Columns and units:** raw_row; model_name (text); release_year (calendar year); domain (domains separated by ", "); has_parameters, has_compute, has_dataset_size (True or False).

## 8. data/processed/answer_summary.csv

* **What it is:** the answer table. One row per estimate across the three lenses, drawn in the opening chart and summarised in the headline sentence.
* **Primary key:** lens_id together with row_label.
* **Time coverage:** whatever each estimate covers, shown in first_year and last_year.
* **Latest observation:** release year 2026 for lenses 1 and 2 (a partial year), 2025 for lens 3.
* **Transform applied** (in `answer_summary()` in `src/attribution.py`): for each slice or scenario, least squares of the logged parameters and logged data (dataset size, or tokens processed) on decimal year. The two slopes become yearly growth factors and shares of their sum. The 95% range resamples the models 1,000 times with a fixed random seed. Shares are blank unless both slopes are positive and there are at least 10 models across at least 3 release years.
* **Columns and units:**
  * lens_id, lens_label, row_label: which lens and which slice or scenario.
  * method: dataset_size or tokens_processed.
  * n_models: models in the estimate, count. first_year, last_year: calendar years.
  * params_factor_per_year, data_factor_per_year, growth_factor_per_year: growth multiple per year, a ratio (2.0 means doubling each year).
  * params_share, data_share: share of the yearly growth in parameters times data, a fraction between 0 and 1, summing to 1.
  * share_lo, share_hi: 95% range for params_share, a fraction.
  * status: ok, not_both_rising or too_few.

## 9. data/processed/association_summary.csv (lens 4)

* **What it is:** Spearman rank correlation of training compute with parameters and with dataset size, for three scopes. It feeds the cross check on the opening screen.
* **Primary key:** scope_id.
* **Time coverage:** whatever each scope covers (all years, Language models, or release year 2020 or later).
* **Latest observation:** release year 2026.
* **Transform applied** (in `src/association.py`): keep models with all three numbers above zero, so both correlations use the same models. rho_params and rho_data are the Spearman correlations. diff is their gap. diff_lo and diff_hi are a 95% range from resampling the models 500 times with a fixed random seed.
* **Columns and units:** scope_id, scope_label (text); n_models (count); rho_params, rho_data, diff, diff_lo, diff_hi (correlation units between minus 1 and 1, differences between minus 2 and 2); status (ok or too_few).

## 10. data/processed/cleaning_steps.csv

* **What it is:** how many models remain after each cleaning step, shown in the "Data source and cleaning steps" panel.
* **Primary key:** table together with step.
* **Time coverage and latest observation:** same as the raw file it counts.
* **Transform applied:** row counts taken during `build_models()` and `build_compute_models()` in `src/data.py`.
* **Columns and units:** table (models or compute_models), step (text), models_remaining (count).

## 11. data/source.json

* **What it is:** citation, URL, licence and download date, shown on screen.
* **Primary key:** not applicable, a single record.
* **Time coverage:** not applicable. The download date is 18 November 2026.
* **Transform applied:** none, written by hand.

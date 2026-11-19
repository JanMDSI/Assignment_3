# Data dictionary

Every file the project reads or writes is listed here. All of it comes from one source, Epoch AI's "Data on AI Models".

* Source: Epoch AI, "Data on AI Models"
* URL: https://epoch.ai/data/ai-models
* Licence: CC BY 4.0 (credit Epoch AI)
* Downloaded: 5 November 2026

## 1. data/raw/epoch_ai_models.csv

* **What it is:** the full download, untouched. One row per model.
* **Primary key:** Model together with Publication date. The raw file has 3,708 rows and 57 columns.
* **Time coverage:** publication dates from 2 July 1950 to 28 October 2026. 13 rows have no usable date.
* **Latest observation:** latest publication date 28 October 2026. Latest record change 4 November 2026.
* **Transform applied:** none. This is the input to the pipeline.
* **Columns used by the app** (the other columns are kept in the file but not read):
  * Model: model name, text.
  * Publication date: date of the paper or announcement, YYYY-MM-DD.
  * Organization: comma separated list of organizations, text.
  * Organization categorization: Industry, Academia, Government, or a mix, text.
  * Domain: comma separated list such as Language or Vision, text.
  * Parameters: number of learnable parameters, count.
  * Training dataset size (total): training data size in datapoints, stored as text. Most values are plain or scientific numbers. A few hold two comma separated values.
  * Training compute (FLOP): training compute, floating point operations. Used only for a consistency check.
  * Link, Parameters notes, Dataset size notes, Last modified: source link, Epoch's notes, timestamp of Epoch's last edit.

## 2. data/samples/epoch_ai_models_sample.csv

* **What it is:** a small source sample with the same 57 columns, for quick pipeline checks.
* **Primary key:** Model together with Publication date.
* **Time coverage:** publication dates from 1 March 1955 to 28 October 2026. It has 124 rows.
* **Latest observation:** 28 October 2026.
* **Transform applied:** keeps every 30th row of the raw file (`python -m src.data --make-sample`). No values are changed.

## 3. data/processed/models.csv

* **What it is:** one row per model that reports both parameters and a usable dataset size. The app reads this file.
* **Primary key:** model_name (checked to be unique by the pipeline).
* **Time coverage:** release years 1950 to 2026, publication dates 2 July 1950 to 9 September 2026.
* **Latest observation:** latest publication date 9 September 2026. Latest record change 3 November 2026.
* **Transform applied** (in `src/data.py`): keep rows with parameters above zero, a dataset size that parses to one positive number, and a valid publication date. Equal pairs of dataset sizes are collapsed to one value. Conflicting pairs are dropped. Organizations and domains are tidied into lists separated by semicolons and commas. The release year and a single domain group are derived.
* **Columns and units:**
  * model_name: text.
  * release_year: calendar year, whole number.
  * publication_date: YYYY-MM-DD.
  * organization: organizations separated by "; ", text.
  * organization_sector: Industry, Academia, Government, Research collective, Mixed or collaboration, or Not specified.
  * domain: domains separated by ", ", text.
  * domain_group: the single domain, or "Multiple domains", or Not specified.
  * parameters: count of learnable parameters.
  * training_dataset_size: datapoints. Mostly tokens for language models, images or other items elsewhere.
  * training_compute_flops: floating point operations, blank when not reported.
  * source_link, parameters_notes, dataset_size_notes: text from Epoch, may be blank.
  * last_modified: timestamp of Epoch's last edit to the row.

## 4. data/processed/growth_by_slice.csv

* **What it is:** the answer table. One row per slice of the data, drawn as the bars in section 1 and summarised in the opening sentence.
* **Primary key:** slice_id.
* **Time coverage:** whatever each slice covers, shown in first_year and last_year.
* **Latest observation:** release year 2026 (the last year is partial).
* **Transform applied:** for each slice defined in `src/analysis.py`, ordinary least squares of log10(parameters) and of log10(dataset size) on release year. The two slopes are turned into yearly growth factors and into shares of their sum. Shares are left blank unless both slopes are positive and the slice has at least 10 models across at least 3 release years.
* **Columns and units:**
  * slice_id, slice_label: short code and readable name.
  * n_models: models in the slice, count.
  * first_year, last_year: calendar years.
  * params_factor_per_year, data_factor_per_year: growth multiple per year, a ratio (2.0 means doubling each year).
  * params_share, data_share: share of the yearly growth in parameters times data, a fraction between 0 and 1, summing to 1.
  * status: ok, not_both_rising, or too_few.

## 5. data/processed/cleaning_steps.csv

* **What it is:** how many models remain after each cleaning step, shown in the "Data source and cleaning steps" panel.
* **Primary key:** step.
* **Time coverage and latest observation:** same as the raw file it counts.
* **Transform applied:** row counts taken during `build_models()` in `src/data.py`.
* **Columns and units:** step (text), models_remaining (count).

## 6. data/source.json

* **What it is:** citation, URL, licence and download date, shown on screen.
* **Primary key:** not applicable, a single record.
* **Time coverage:** not applicable. The download date is 5 November 2026.
* **Transform applied:** none, written by hand.

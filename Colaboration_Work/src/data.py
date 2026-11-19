"""Data pipeline: raw Epoch AI file in, processed files out.

Run it with:

    python -m src.data                      # raw file to data/processed/
    python -m src.data --raw PATH --out DIR # any raw file, any output folder
    python -m src.data --make-sample        # rebuild the small source sample

Seven files are written to the output folder, and the app reads only these:

    models.csv                    models with parameters and a dataset size (lens 1)
    compute_models.csv            models with parameters and compute (lens 2)
    frontier_language_models.csv  frontier language models since 2019 (lens 3)
    coverage.csv                  every dated model, with which of the three numbers it reports (lens 4)
    answer_summary.csv            the growth split for every estimate in the app
    association_summary.csv       compute against parameters and dataset size, by scope (lens 4)
    cleaning_steps.csv            how many models are left after each cleaning step

Cleaning rules, in plain words:
    * Dataset size is stored as text in the raw file. Plain and scientific
      numbers are read as they are. A pair such as "5e9,5e9" is collapsed when
      both values agree and dropped when they disagree, because guessing would
      invent data.
    * Zero values are dropped, since they cannot sit on a log axis.
    * Rows without a usable publication date are dropped (the year filter needs it).
    * Line breaks inside model names are replaced by spaces.
    * Organisation and domain lists are tidied so a filter can match any one entry.
    * Every row keeps raw_row, its row number in the raw file as a spreadsheet
      shows it (the header is row 1), so any figure can be traced back.
Nothing is estimated or filled in. Every number in the app is read from, or
calculated from, a row of the raw file.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from src.analysis import DOMAIN_SEP, ORG_SEP
from src.association import association_summary
from src.attribution import answer_summary


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "all_ai_models.csv"
SAMPLE = ROOT / "data" / "samples" / "all_ai_models_sample.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
SOURCE_JSON = ROOT / "data" / "source.json"
MOE_CSV = ROOT / "data" / "curation" / "mixture_of_experts.csv"

SAMPLE_STEP = 30  # the sample keeps every 30th row of the raw file
FRONTIER_FIRST_YEAR = 2019

NOT_SPECIFIED = "Not specified"
NOT_RECORDED = "Not recorded"
MULTIPLE_DOMAINS = "Multiple domains"
MIXED_SECTOR = "Mixed or collaboration"

# Raw Epoch column mapped to the name used in the processed files
EPOCH_COLUMNS = {
    "Model": "model_name",
    "Publication date": "publication_date",
    "Organization": "organization",
    "Organization categorization": "organization_sector",
    "Domain": "domain",
    "Parameters": "parameters",
    "Training dataset size (total)": "training_dataset_size",
    "Training compute (FLOP)": "training_compute_flops",
    "Training compute estimation method": "compute_method",
    "Frontier model": "frontier_model",
    "Confidence": "confidence",
    "Base model": "base_model",
    "Link": "source_link",
    "Parameters notes": "parameters_notes",
    "Dataset size notes": "dataset_size_notes",
    "Last modified": "last_modified",
}
MODEL_COLUMNS = [
    "raw_row", "model_name", "release_year", "decimal_year", "publication_date", "organization",
    "organization_sector", "domain", "domain_group", "parameters", "training_dataset_size",
    "training_compute_flops", "compute_method", "frontier_model", "confidence", "base_model",
    "source_link", "parameters_notes", "dataset_size_notes", "last_modified",
]
COMPUTE_COLUMNS = [
    "raw_row", "model_name", "release_year", "decimal_year", "publication_date", "domain", "parameters",
    "training_compute_flops", "tokens_processed", "compute_method", "frontier_model", "source_link", "last_modified",
]
FRONTIER_COLUMNS = MODEL_COLUMNS + ["fresh_dense", "exclusion_reason"]
TEXT_COLUMNS = ("source_link", "parameters_notes", "dataset_size_notes", "last_modified", "base_model", "exclusion_reason")
COVERAGE_COLUMNS = ["raw_row", "model_name", "release_year", "domain", "has_parameters", "has_compute", "has_dataset_size"]
OUTPUT_FILES = (
    "models.csv", "compute_models.csv", "frontier_language_models.csv", "coverage.csv",
    "answer_summary.csv", "association_summary.csv", "cleaning_steps.csv",
)

_LEGAL_SUFFIX = re.compile(r"^(inc\.?|ltd\.?|llc|co\.?|corp\.?|limited|gmbh)$", re.IGNORECASE)
_SUFFIXES = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}


# ------------------------------------------------------------------- parsing


def _blank(value: object) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or not str(value).strip()


def parse_count(value: object) -> float:
    """Read '1.5e9', '1,200,000', '36T' or '5e9,5e9' as a number.

    Returns NaN for blanks and for comma separated values that disagree.
    """
    if _blank(value):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("\u2009", "")
    try:
        return float(text)
    except ValueError:
        pass
    parts = [part.strip() for part in text.split(",")]
    if len(parts) > 1:
        numbers = [parse_count(part) for part in parts]
        if not pd.isna(numbers[0]) and all(n == numbers[0] for n in numbers):
            return numbers[0]
        if re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", text):  # thousands separators
            return float(text.replace(",", ""))
        return float("nan")
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kmbt])", text, re.IGNORECASE)
    if match:
        return float(match.group(1)) * _SUFFIXES[match.group(2).lower()]
    return float("nan")


def split_organizations(text: object) -> list[str]:
    """Split Epoch's comma list of organisations, keeping 'Co., Ltd.' style names whole."""
    if _blank(text):
        return []
    parts: list[str] = []
    for token in (t.strip() for t in str(text).split(",")):
        if not token:
            continue
        if parts and _LEGAL_SUFFIX.match(token):
            parts[-1] = f"{parts[-1]}, {token}"
        else:
            parts.append(token)
    return list(dict.fromkeys(parts))


def split_domains(text: object) -> list[str]:
    if _blank(text):
        return []
    return sorted({t.strip() for t in str(text).replace(";", ",").split(",") if t.strip()})


def _sector(text: object) -> str:
    if _blank(text):
        return NOT_SPECIFIED
    tokens = {t.strip() for t in str(text).split(",") if t.strip()}
    return tokens.pop() if len(tokens) == 1 else MIXED_SECTOR


def _tidy_name(text: object) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _decimal_year(dates: pd.Series) -> pd.Series:
    """Year plus the fraction of the year that had passed, for example mid 2020 is about 2020.5."""
    return dates.dt.year + (dates.dt.dayofyear - 1) / 365.25


def _tidy_domains(frame: pd.DataFrame) -> pd.DataFrame:
    domain_lists = frame["domain"].map(split_domains)
    frame["domain"] = domain_lists.map(lambda d: DOMAIN_SEP.join(d) if d else NOT_SPECIFIED)
    frame["domain_group"] = domain_lists.map(lambda d: d[0] if len(d) == 1 else (MULTIPLE_DOMAINS if d else NOT_SPECIFIED))
    return frame


def _prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename columns, keep the raw row number and parse the shared fields."""
    missing = set(EPOCH_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"The raw file is missing columns: {sorted(missing)}")
    df = raw.rename(columns=EPOCH_COLUMNS)[list(EPOCH_COLUMNS.values())].copy()
    df.insert(0, "raw_row", df.index + 2)
    df["model_name"] = df["model_name"].map(_tidy_name)
    df["frontier_model"] = df["frontier_model"].astype(str).str.strip().str.lower().eq("true")
    df["confidence"] = df["confidence"].fillna(NOT_RECORDED)
    df["compute_method"] = df["compute_method"].fillna(NOT_RECORDED)
    df["base_model"] = df["base_model"].fillna("").astype(str)
    df["parameters"] = pd.to_numeric(df["parameters"], errors="coerce")
    df["training_compute_flops"] = pd.to_numeric(df["training_compute_flops"], errors="coerce")
    return df


# ----------------------------------------------------------------- the tables


def build_models(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, str, int]]]:
    """Lens 1 table: models with parameters and a usable dataset size."""
    df = _prepare(raw)
    steps = [("models", "Models in the Epoch AI snapshot", len(df))]

    df = df[df["parameters"] > 0]
    steps.append(("models", "With a reported parameter count", len(df)))
    df = df[df["training_dataset_size"].notna()]
    steps.append(("models", "With a reported training dataset size", len(df)))
    df["training_dataset_size"] = df["training_dataset_size"].map(parse_count)
    df = df[df["training_dataset_size"].notna()]
    steps.append(("models", "With one unambiguous dataset size", len(df)))
    df = df[df["training_dataset_size"] > 0]
    steps.append(("models", "With a positive size (needed for a log axis)", len(df)))
    dates = pd.to_datetime(df["publication_date"], errors="coerce")
    df = df[dates.notna()]
    dates = dates[dates.notna()]
    steps.append(("models", "With a publication date", len(df)))

    df["publication_date"] = dates.dt.strftime("%Y-%m-%d")
    df["release_year"] = dates.dt.year.astype("int64")
    df["decimal_year"] = _decimal_year(dates).round(4)
    df["organization_sector"] = df["organization_sector"].map(_sector)
    df = _tidy_domains(df)
    df["organization"] = df["organization"].map(split_organizations).map(lambda o: ORG_SEP.join(o) if o else NOT_SPECIFIED)
    for column in ("source_link", "parameters_notes", "dataset_size_notes", "last_modified"):
        df[column] = df[column].fillna("").astype(str)
    return df[MODEL_COLUMNS].sort_values(["release_year", "model_name", "raw_row"]).reset_index(drop=True), steps


def build_compute_models(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, str, int]]]:
    """Lens 2 table: models with parameters and training compute, dataset size not needed."""
    df = _prepare(raw)
    steps = [("compute_models", "Models in the Epoch AI snapshot", len(df))]
    dates = pd.to_datetime(df["publication_date"], errors="coerce")
    df = df[dates.notna()]
    dates = dates[dates.notna()]
    steps.append(("compute_models", "With a publication date", len(df)))
    df = df[df["training_compute_flops"] > 0]
    dates = dates.loc[df.index]
    steps.append(("compute_models", "With positive training compute", len(df)))
    df = df[df["parameters"] > 0]
    dates = dates.loc[df.index]
    steps.append(("compute_models", "With positive parameters as well", len(df)))

    df["publication_date"] = dates.dt.strftime("%Y-%m-%d")
    df["release_year"] = dates.dt.year.astype("int64")
    df["decimal_year"] = _decimal_year(dates).round(4)
    df = _tidy_domains(df)
    df["tokens_processed"] = df["training_compute_flops"] / (6 * df["parameters"])
    for column in ("source_link", "last_modified"):
        df[column] = df[column].fillna("").astype(str)
    return df[COMPUTE_COLUMNS].sort_values(["decimal_year", "raw_row"]).reset_index(drop=True), steps


def build_coverage(raw: pd.DataFrame) -> pd.DataFrame:
    """Lens 4 table: every model with a valid date and which of the three numbers it reports.

    A number counts as reported when it is above zero (and, for dataset size, when it
    parses to one value). Missing values are never treated as zero.
    """
    df = _prepare(raw)
    dates = pd.to_datetime(df["publication_date"], errors="coerce")
    df = df[dates.notna()].copy()
    df["release_year"] = dates[dates.notna()].dt.year.astype("int64")
    df = _tidy_domains(df)
    df["has_parameters"] = df["parameters"] > 0
    df["has_compute"] = df["training_compute_flops"] > 0
    df["has_dataset_size"] = df["training_dataset_size"].map(parse_count) > 0
    return df[COVERAGE_COLUMNS].sort_values(["release_year", "model_name", "raw_row"]).reset_index(drop=True)


def load_moe(path: Path = MOE_CSV) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["model_name", "evidence"])


def build_frontier(models: pd.DataFrame, moe: pd.DataFrame) -> pd.DataFrame:
    """Lens 3 table: frontier language models since 2019, with a flag for models that are not a fresh dense run.

    Epoch calls a model frontier when it was among the ten most compute heavy when released.
    A model is flagged when Epoch lists a base model (so it is a fine tune or derived model),
    or when it is in data/curation/mixture_of_experts.csv.
    """
    frontier = models[models["frontier_model"] & (models["domain"] == "Language") & (models["release_year"] >= FRONTIER_FIRST_YEAR)].copy()
    moe_notes = dict(zip(moe["model_name"], moe["evidence"]))
    derived = frontier["base_model"].str.len() > 0
    sparse = frontier["model_name"].isin(moe_notes)
    frontier["fresh_dense"] = ~(derived | sparse)
    reason = pd.Series("", index=frontier.index)
    reason = reason.mask(derived, "Derived from " + frontier["base_model"] + ". Its recorded training data describes the extra step, or the base model's data, rather than one full training run.")
    reason = reason.mask(sparse, "Mixture of experts: only part of the parameters is used for each token. " + frontier["model_name"].map(moe_notes).fillna(""))
    frontier["exclusion_reason"] = reason
    return frontier[FRONTIER_COLUMNS].reset_index(drop=True)


def validate(models: pd.DataFrame, compute_models: pd.DataFrame, frontier: pd.DataFrame, coverage: pd.DataFrame) -> None:
    """Stop with a clear message if a processed table breaks a documented rule."""
    for name, table in (("models", models), ("compute_models", compute_models)):
        if table.empty:
            raise ValueError(f"No rows survived cleaning for {name}")
        if table["raw_row"].duplicated().any():
            raise ValueError(f"{name}: raw_row must be unique")
        if table[["model_name", "release_year", "decimal_year"]].isna().any().any():
            raise ValueError(f"{name}: name and year can never be empty")
        if not (table["parameters"] > 0).all():
            raise ValueError(f"{name}: parameters must be positive")
        dates = pd.to_datetime(table["publication_date"], errors="raise")
        if dates.dt.year.ne(table["release_year"]).any():
            raise ValueError(f"{name}: release_year must match publication_date")
    if not (models["training_dataset_size"] > 0).all():
        raise ValueError("models: dataset size must be positive")
    if models[["organization", "domain"]].isna().any().any():
        raise ValueError("models: organisation and domain can never be empty")
    if not (compute_models["training_compute_flops"] > 0).all():
        raise ValueError("compute_models: compute must be positive")
    ratio = compute_models["tokens_processed"] * 6 * compute_models["parameters"] / compute_models["training_compute_flops"]
    if ((ratio - 1).abs() > 1e-9).any():
        raise ValueError("compute_models: tokens_processed must equal compute / (6 x parameters)")
    if not frontier.empty and not frontier["raw_row"].isin(models["raw_row"]).all():
        raise ValueError("frontier rows must come from models")
    if coverage["raw_row"].duplicated().any():
        raise ValueError("coverage: raw_row must be unique")
    if int((coverage["has_parameters"] & coverage["has_dataset_size"]).sum()) != len(models):
        raise ValueError("coverage and models disagree on how many models report parameters and dataset size")
    if int((coverage["has_parameters"] & coverage["has_compute"]).sum()) != len(compute_models):
        raise ValueError("coverage and compute_models disagree on how many models report parameters and compute")


def write_processed(raw_path: Path, out_dir: Path) -> dict:
    """Run the whole pipeline and write the seven processed files."""
    raw = pd.read_csv(raw_path, low_memory=False)
    models, steps_a = build_models(raw)
    compute_models, steps_b = build_compute_models(raw)
    frontier = build_frontier(models, load_moe())
    coverage = build_coverage(raw)
    validate(models, compute_models, frontier, coverage)
    summary = answer_summary(models, compute_models, frontier)
    associations = association_summary(models)
    cleaning = pd.DataFrame(steps_a + steps_b, columns=["table", "step", "models_remaining"])

    out_dir.mkdir(parents=True, exist_ok=True)
    models.to_csv(out_dir / "models.csv", index=False)
    compute_models.to_csv(out_dir / "compute_models.csv", index=False)
    frontier.to_csv(out_dir / "frontier_language_models.csv", index=False)
    coverage.to_csv(out_dir / "coverage.csv", index=False)
    summary.to_csv(out_dir / "answer_summary.csv", index=False)
    associations.to_csv(out_dir / "association_summary.csv", index=False)
    cleaning.to_csv(out_dir / "cleaning_steps.csv", index=False)
    return {"raw_rows": len(raw), "models": len(models), "compute_models": len(compute_models), "frontier": len(frontier), "estimates": len(summary), "coverage": len(coverage)}


def make_sample(raw_path: Path = RAW, out_path: Path = SAMPLE, step: int = SAMPLE_STEP) -> int:
    """Keep every `step`th raw row. The sample keeps the raw layout, so the same pipeline runs on it."""
    raw = pd.read_csv(raw_path, low_memory=False)
    sample = raw.iloc[::step]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out_path, index=False)
    return len(sample)


def load_processed(folder: Path = PROCESSED_DIR) -> dict[str, pd.DataFrame]:
    """Read the processed files the app needs, keyed by short name."""
    for name in OUTPUT_FILES:
        if not (folder / name).exists():
            raise FileNotFoundError(f"{folder / name} is missing. Run: python -m src.data")

    def read(name: str) -> pd.DataFrame:
        frame = pd.read_csv(folder / name, keep_default_na=False, na_values=[""])
        for column in TEXT_COLUMNS:
            if column in frame:
                frame[column] = frame[column].fillna("").astype(str)
        return frame

    return {
        "models": read("models.csv"),
        "compute": read("compute_models.csv"),
        "frontier": read("frontier_language_models.csv"),
        "coverage": read("coverage.csv"),
        "summary": read("answer_summary.csv"),
        "associations": read("association_summary.csv"),
        "cleaning": read("cleaning_steps.csv"),
    }


def load_source_info(path: Path = SOURCE_JSON) -> dict:
    """Citation details shown on screen, read with the standard library only."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=RAW, help="raw Epoch AI csv")
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR, help="folder for processed files")
    parser.add_argument("--make-sample", action="store_true", help="rebuild data/samples/ from the raw file and stop")
    args = parser.parse_args()
    if args.make_sample:
        print(f"Wrote {make_sample(args.raw)} sample rows to {SAMPLE}")
        return
    summary = write_processed(args.raw, args.out)
    print(
        f"Read {summary['raw_rows']:,} raw rows. Wrote {summary['models']:,} models, {summary['compute_models']:,} compute models, "
        f"{summary['frontier']} frontier language models, {summary['coverage']:,} coverage rows and {summary['estimates']} estimates to {args.out}"
    )


if __name__ == "__main__":
    main()

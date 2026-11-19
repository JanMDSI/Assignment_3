"""Data pipeline: raw Epoch AI file in, processed files out.

Run it with:

    python -m src.data                      # raw file to data/processed/
    python -m src.data --raw PATH --out DIR # any raw file, any output folder
    python -m src.data --make-sample        # rebuild the small source sample

Three files are written to the output folder, and the app reads only these:

    models.csv           one row per model that reports parameters and dataset size
    growth_by_slice.csv  the growth split for each slice defined in src/analysis.py
    cleaning_steps.csv   how many models are left after each cleaning step

Cleaning rules, in plain words:
    * Dataset size is stored as text in the raw file. Plain and scientific
      numbers are read as they are. A pair such as "5e9,5e9" is collapsed when
      both values agree and dropped when they disagree, because guessing would
      invent data.
    * Zero values are dropped, since they cannot sit on a log axis.
    * Rows without a usable publication date are dropped (the year filter needs it).
    * Organisation and domain lists are tidied so a filter can match any one entry.
Nothing is estimated or filled in. Every number in the app is read from, or
calculated from, a row of the raw file.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from src.analysis import DOMAIN_SEP, ORG_SEP, growth_by_slice


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "epoch_ai_models.csv"
SAMPLE = ROOT / "data" / "samples" / "epoch_ai_models_sample.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
SOURCE_JSON = ROOT / "data" / "source.json"

SAMPLE_STEP = 30  # the sample keeps every 30th row of the raw file

NOT_SPECIFIED = "Not specified"
MULTIPLE_DOMAINS = "Multiple domains"
MIXED_SECTOR = "Mixed or collaboration"

# Raw Epoch column mapped to the name used in the processed file
EPOCH_COLUMNS = {
    "Model": "model_name",
    "Publication date": "publication_date",
    "Organization": "organization",
    "Organization categorization": "organization_sector",
    "Domain": "domain",
    "Parameters": "parameters",
    "Training dataset size (total)": "training_dataset_size",
    "Training compute (FLOP)": "training_compute_flops",
    "Link": "source_link",
    "Parameters notes": "parameters_notes",
    "Dataset size notes": "dataset_size_notes",
    "Last modified": "last_modified",
}
MODEL_COLUMNS = [
    "model_name", "release_year", "publication_date", "organization", "organization_sector",
    "domain", "domain_group", "parameters", "training_dataset_size", "training_compute_flops",
    "source_link", "parameters_notes", "dataset_size_notes", "last_modified",
]

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


# ----------------------------------------------------------------- the pipeline


def build_models(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean the raw table. Returns the model table and the step by step counts."""
    missing = set(EPOCH_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"The raw file is missing columns: {sorted(missing)}")
    df = raw.rename(columns=EPOCH_COLUMNS)[list(EPOCH_COLUMNS.values())].copy()
    steps = [("Models in the Epoch AI snapshot", len(df))]

    df = df[pd.to_numeric(df["parameters"], errors="coerce") > 0]
    steps.append(("With a reported parameter count", len(df)))

    df = df[df["training_dataset_size"].notna()]
    steps.append(("With a reported training dataset size", len(df)))

    df["training_dataset_size"] = df["training_dataset_size"].map(parse_count)
    df = df[df["training_dataset_size"].notna()]
    steps.append(("With one unambiguous dataset size", len(df)))

    df = df[df["training_dataset_size"] > 0]
    steps.append(("With a positive size (needed for a log axis)", len(df)))

    df = df[pd.to_datetime(df["publication_date"], errors="coerce").notna()]
    steps.append(("With a publication date", len(df)))

    dates = pd.to_datetime(df["publication_date"])
    df["publication_date"] = dates.dt.strftime("%Y-%m-%d")
    df["release_year"] = dates.dt.year.astype("int64")
    df["organization_sector"] = df["organization_sector"].map(_sector)

    domain_lists = df["domain"].map(split_domains)
    df["domain"] = domain_lists.map(lambda d: DOMAIN_SEP.join(d) if d else NOT_SPECIFIED)
    df["domain_group"] = domain_lists.map(
        lambda d: d[0] if len(d) == 1 else (MULTIPLE_DOMAINS if d else NOT_SPECIFIED)
    )
    org_lists = df["organization"].map(split_organizations)
    df["organization"] = org_lists.map(lambda o: ORG_SEP.join(o) if o else NOT_SPECIFIED)

    for column in ("parameters", "training_dataset_size", "training_compute_flops"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in ("source_link", "parameters_notes", "dataset_size_notes", "last_modified"):
        df[column] = df[column].fillna("").astype(str)

    models = df[MODEL_COLUMNS].sort_values(["release_year", "model_name"]).reset_index(drop=True)
    cleaning = pd.DataFrame(steps, columns=["step", "models_remaining"])
    return models, cleaning


def validate(models: pd.DataFrame) -> None:
    """Stop with a clear message if the processed table breaks a documented rule."""
    if models.empty:
        raise ValueError("No models survived cleaning")
    if models["model_name"].duplicated().any():
        raise ValueError("model_name must be unique")
    if models[["model_name", "organization", "domain", "release_year"]].isna().any().any():
        raise ValueError("Name, organisation, domain and year can never be empty")
    if not ((models["parameters"] > 0).all() and (models["training_dataset_size"] > 0).all()):
        raise ValueError("Parameters and dataset size must be positive")
    if pd.to_datetime(models["publication_date"], errors="raise").dt.year.ne(models["release_year"]).any():
        raise ValueError("release_year must match publication_date")


def write_processed(raw_path: Path, out_dir: Path) -> dict:
    """Run the whole pipeline and write the three processed files."""
    raw = pd.read_csv(raw_path, low_memory=False)
    models, cleaning = build_models(raw)
    validate(models)
    slices = growth_by_slice(models)
    out_dir.mkdir(parents=True, exist_ok=True)
    models.to_csv(out_dir / "models.csv", index=False)
    slices.to_csv(out_dir / "growth_by_slice.csv", index=False)
    cleaning.to_csv(out_dir / "cleaning_steps.csv", index=False)
    return {"models": len(models), "slices": len(slices), "raw_rows": len(raw)}


def make_sample(raw_path: Path = RAW, out_path: Path = SAMPLE, step: int = SAMPLE_STEP) -> int:
    """Keep every `step`th raw row. The sample keeps the raw layout, so the same pipeline runs on it."""
    raw = pd.read_csv(raw_path, low_memory=False)
    sample = raw.iloc[::step]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out_path, index=False)
    return len(sample)


def load_processed(folder: Path = PROCESSED_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the processed files the app needs: models, slices, cleaning steps."""
    for name in ("models.csv", "growth_by_slice.csv", "cleaning_steps.csv"):
        if not (folder / name).exists():
            raise FileNotFoundError(f"{folder / name} is missing. Run: python -m src.data")
    models = pd.read_csv(folder / "models.csv", keep_default_na=False, na_values=[""])
    for column in ("parameters_notes", "dataset_size_notes", "source_link", "last_modified"):
        models[column] = models[column].fillna("").astype(str)
    slices = pd.read_csv(folder / "growth_by_slice.csv")
    cleaning = pd.read_csv(folder / "cleaning_steps.csv")
    return models, slices, cleaning


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
    print(f"Read {summary['raw_rows']:,} raw rows, wrote {summary['models']:,} models and {summary['slices']} slices to {args.out}")


if __name__ == "__main__":
    main()

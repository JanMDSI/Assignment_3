"""
Member 3 transform: Epoch AI raw file -> member3_language_models_cleaned.csv

Source : data/source/all_ai_models.csv
         (Epoch AI, "Data on AI models", https://epoch.ai/data/ai-models, CC BY 4.0)
Output : data/processed/member3_language_models_cleaned.csv

Steps
 1. Keep frontier models (column "Frontier model" == True).
    Epoch defines these as models in the top 10 by training compute when released.
 2. Keep models whose Domain is exactly "Language" (excludes multimodal models).
 3. Keep models published in 2019 or later.
 4. Keep models with both a parameter count and a training-dataset size recorded.
 5. Keep 9 columns, rename them, and turn the publication date into a release year.
 6. Sort by release year, then model name.

Run from the repository root:
    python scripts/clean_member3.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "data" / "source" / "all_ai_models.csv"
OUTPUT = ROOT / "data" / "processed" / "member3_language_models_cleaned.csv"

FIRST_YEAR = 2019

RENAME = {
    "Model": "Model",
    "Release year": "Release year",
    "Parameters": "Parameter count (as recorded)",
    "Training dataset size (total)": "Training data size (as recorded)",
    "Confidence": "Source confidence",
    "Training compute (FLOP)": "Training compute (FLOP, as recorded)",
    "Link": "Source URL",
    "Parameters notes": "Parameter-count notes",
    "Dataset size notes": "Dataset-size notes",
}


def main():
    raw = pd.read_csv(SOURCE, low_memory=False)
    print(f"Read {len(raw)} models from {SOURCE.name}")

    df = raw.copy()
    df["Release year"] = pd.to_datetime(
        df["Publication date"], errors="coerce"
    ).dt.year

    frontier = df["Frontier model"].astype(str).str.strip().str.lower() == "true"
    df = df[frontier]
    print(f"  frontier models:                     {len(df)}")

    df = df[df["Domain"] == "Language"]
    print(f"  ... with Domain == 'Language':       {len(df)}")

    df = df[df["Release year"] >= FIRST_YEAR]
    print(f"  ... released {FIRST_YEAR} or later:          {len(df)}")

    df = df.dropna(subset=["Parameters", "Training dataset size (total)"])
    print(f"  ... with parameters and dataset size: {len(df)}")

    out = df[list(RENAME)].rename(columns=RENAME)
    out["Release year"] = out["Release year"].astype(int)
    out = out.sort_values(["Release year", "Model"]).reset_index(drop=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(out)} rows to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
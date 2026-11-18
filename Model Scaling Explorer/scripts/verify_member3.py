import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/member3_language_models_cleaned.csv")

# The 6 fine-tuned / mixture-of-experts models the app excludes
drop = ["Flan-PaLM 540B", "U-PaLM (540B)", "Minerva (540B)", "FLAN 137B",
        "Llama Nemotron Ultra 253B", "Switch"]
d = df[~df["Model"].isin(drop)]

def slope(col):
    # trend of log10(value) against release year
    return np.polyfit(d["Release year"], np.log10(d[col]), 1)[0]

p = slope("Parameter count (as recorded)")
t = slope("Training data size (as recorded)")
c = slope("Training compute (FLOP, as recorded)")

print(len(d), "models")
print("Parameters grew", round(10**p, 1), "x per year")
print("Compute grew", round(10**c, 1), "x per year")
print("Tokens-processed method:", round(p / c * 100), "% from bigger models")
print("Dataset-size method:", round(p / (p + t) * 100), "% from bigger models")
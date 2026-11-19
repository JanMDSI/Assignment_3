"""Which moves more closely with training compute: parameters or dataset size?

This is lens 4. It does not split growth. It asks, across models, how closely
compute rises with each of the two, using Spearman rank correlation (it only
cares about order, so the huge range of the numbers does not matter).

Two choices keep the comparison fair:
  * Both correlations use the same models, the ones with all three numbers.
    Comparing a correlation from one set of models with a correlation from a
    different set would mix up sample differences with real differences.
  * The gap between the two correlations gets a 95% range from resampling
    the models 500 times, so a small gap can be called small.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis import has_domain


N_BOOT = 500
SEED = 0
MIN_MODELS = 10

SCOPES = [
    {"id": "all", "label": "All models with all three numbers", "domain": None, "start": None},
    {"id": "language", "label": "Language models", "domain": "Language", "start": None},
    {"id": "recent", "label": "Released 2020 or later", "domain": None, "start": 2020},
]


def with_all_three(models: pd.DataFrame) -> pd.DataFrame:
    """Models that report parameters, dataset size and compute, all positive."""
    rows = models.dropna(subset=["parameters", "training_dataset_size", "training_compute_flops"])
    return rows[(rows["parameters"] > 0) & (rows["training_dataset_size"] > 0) & (rows["training_compute_flops"] > 0)]


def spearman(x: pd.Series, y: pd.Series) -> float:
    """Pearson correlation of the ranks. Tied values share their average rank."""
    return float(pd.Series(np.asarray(x)).rank().corr(pd.Series(np.asarray(y)).rank()))


def compare_associations(frame: pd.DataFrame, n_boot: int = N_BOOT, min_models: int = MIN_MODELS) -> dict:
    """Spearman of compute with parameters and with dataset size on the same models, and their gap."""
    rows = with_all_three(frame)
    nan = float("nan")
    result = {"n_models": int(len(rows)), "rho_params": nan, "rho_data": nan, "diff": nan, "diff_lo": nan, "diff_hi": nan, "status": "too_few"}
    if len(rows) < min_models:
        return result
    compute = rows["training_compute_flops"].to_numpy(float)
    params = rows["parameters"].to_numpy(float)
    data = rows["training_dataset_size"].to_numpy(float)
    result["rho_params"] = spearman(compute, params)
    result["rho_data"] = spearman(compute, data)
    result["diff"] = result["rho_params"] - result["rho_data"]
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(rows), len(rows))
        ranks = pd.DataFrame({"c": compute[pick], "p": params[pick], "d": data[pick]}).rank()
        corr = np.corrcoef(ranks.to_numpy().T)
        if np.isfinite(corr).all():
            diffs.append(corr[0, 1] - corr[0, 2])
    if diffs:
        result["diff_lo"], result["diff_hi"] = (float(v) for v in np.percentile(diffs, [2.5, 97.5]))
    result["status"] = "ok"
    return result


def association_summary(models: pd.DataFrame) -> pd.DataFrame:
    """One row per scope. Saved as data/processed/association_summary.csv."""
    rows = []
    for spec in SCOPES:
        frame = models
        if spec["domain"]:
            frame = frame[has_domain(frame, spec["domain"])]
        if spec["start"] is not None:
            frame = frame[frame["release_year"] >= spec["start"]]
        result = compare_associations(frame)
        rows.append({"scope_id": spec["id"], "scope_label": spec["label"], **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.items()}})
    return pd.DataFrame(rows)


def association_text(result: dict) -> str:
    """One plain sentence for one comparison."""
    if result["status"] != "ok":
        return f"Too few models with parameters, dataset size and compute together (needs {MIN_MODELS})."
    p, d = result["rho_params"], result["rho_data"]
    stronger = "parameters" if p > d else "dataset size"
    base = (
        f"Across {result['n_models']:,} models with all three numbers, training compute rises with parameters at Spearman {p:.2f} "
        f"and with dataset size at {d:.2f}"
    )
    if result["diff_lo"] <= 0 <= result["diff_hi"]:
        return base + f". The gap ({result['diff']:+.2f}, 95% range {result['diff_lo']:+.2f} to {result['diff_hi']:+.2f}) is within sampling noise, so neither clearly moves more closely with compute."
    return base + f". Compute moves more closely with {stronger}: the gap is {abs(result['diff']):.2f} (95% range {result['diff_lo']:+.2f} to {result['diff_hi']:+.2f})."

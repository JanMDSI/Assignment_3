"""Split the growth in AI training compute into 'larger models' and 'more training data'.

Source: Epoch AI, AI models dataset, https://epoch.ai/data/all_ai_models.csv (CC BY 4.0).
Training compute C is about 6 x N x D (N = parameters, D = training data), so D is inferred as C / (6 N).
Then log C = log 6 + log N + log D exactly, so growth in log C splits into a part from N and a part from D.
"""
import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).parent / "data" / "raw" / "all_ai_models.csv"
URL = "https://epoch.ai/data/all_ai_models.csv"
COLUMNS = {
    "model": "Model",
    "date": "Publication date",
    "compute": "Training compute (FLOP)",
    "params": "Parameters",
    "domain": "Domain",
    "method": "Training compute estimation method",
}


def fetch_if_missing():
    if RAW.exists():
        return True
    try:
        RAW.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0 (course project)"})
        RAW.write_bytes(urllib.request.urlopen(req, timeout=60).read())
        return True
    except Exception:
        return False


def load():
    """Return (usable models, lineage). Usable = date, compute and parameters all present."""
    raw = pd.read_csv(RAW, engine="python", on_bad_lines="warn")
    lookup = {c.strip().lstrip("\ufeff").lower(): c for c in raw.columns}
    missing = [v for v in COLUMNS.values() if v.lower() not in lookup]
    if missing:
        raise ValueError(f"Columns not found: {missing}. Columns in file: {list(raw.columns)}")
    d = pd.DataFrame({k: raw[lookup[v.lower()]] for k, v in COLUMNS.items()})
    d.insert(0, "raw_idx", raw.index)
    d.insert(1, "raw_row", raw.index + 2)  # spreadsheet row number, header is row 1
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["compute"] = pd.to_numeric(d["compute"], errors="coerce")
    d["params"] = pd.to_numeric(d["params"], errors="coerce")

    u = d[d["date"].notna() & (d["compute"] > 0) & (d["params"] > 0)].copy()
    u["year"] = u["date"].dt.year + (u["date"].dt.dayofyear - 1) / 365.25
    u["tokens"] = u["compute"] / (6 * u["params"])
    u["log_compute"] = np.log10(u["compute"])
    u["log_params"] = np.log10(u["params"])
    u["log_tokens"] = np.log10(u["tokens"])
    info = {
        "sha256": hashlib.sha256(RAW.read_bytes()).hexdigest(),
        "rows_in_file": len(d),
        "rows_with_compute": int((d["compute"] > 0).sum()),
        "rows_with_params": int((d["params"] > 0).sum()),
        "rows_usable": len(u),
        "latest_date": str(d["date"].max().date()),
    }
    return u.reset_index(drop=True), info


def subset(u, domain=None, independent_only=False):
    if domain:
        u = u[u["domain"].fillna("").str.contains(domain, case=False, regex=False)]
    if independent_only:  # compute reported by developers or estimated from hardware, not from N x D
        m = u["method"].fillna("").str.lower()
        u = u[m.str.contains("reported") | m.str.contains("hardware")]
    return u


def _slopes(x, y):
    """Least-squares slope of y on x. Works on 1-D arrays or on rows of 2-D arrays."""
    xc = x - x.mean(axis=-1, keepdims=True)
    return (xc * (y - y.mean(axis=-1, keepdims=True))).sum(axis=-1) / (xc ** 2).sum(axis=-1)


def decompose(u, start, end, n_boot=1000, seed=0, min_n=30):
    """Growth per year (orders of magnitude) of compute, parameters and data, and the share due to parameters.

    The window is whole calendar years start..end. The interval resamples models with replacement.
    Returns None if there are too few models or compute did not grow.
    """
    w = u[(u["year"] >= start) & (u["year"] < end + 1)]
    n = len(w)
    if n < min_n:
        return None
    x = w["year"].to_numpy()
    c, p, t = (w[k].to_numpy() for k in ("log_compute", "log_params", "log_tokens"))
    s_c, s_p, s_t = _slopes(x, c), _slopes(x, p), _slopes(x, t)
    if s_c <= 0:
        return None
    idx = np.random.default_rng(seed).integers(0, n, (n_boot, n))
    bc, bp = _slopes(x[idx], c[idx]), _slopes(x[idx], p[idx])
    shares = (bp / bc)[bc > 0]
    if shares.size == 0:
        return None
    return {
        "n": n, "slope_compute": s_c, "slope_params": s_p, "slope_tokens": s_t,
        "growth_per_year": 10 ** s_c, "share_params": s_p / s_c, "share_tokens": s_t / s_c,
        "lo": float(np.percentile(shares, 2.5)), "hi": float(np.percentile(shares, 97.5)),
    }


def scenarios(u, start, end):
    """The same share under other sample choices, to show whether the answer depends on them."""
    options = [
        ("All models", u, start, end),
        ("Developer-reported or hardware compute only", subset(u, independent_only=True), start, end),
        ("Language models only", subset(u, "Language"), start, end),
        ("Last five years only", u, end - 4, end),
    ]
    rows = []
    for label, d, a, b in options:
        r = decompose(d, a, b)
        if r:
            rows.append({"scenario": label, "share": r["share_params"], "lo": r["lo"], "hi": r["hi"], "n": r["n"]})
    return pd.DataFrame(rows, columns=["scenario", "share", "lo", "hi", "n"])
"""Colour vision deficiency check for the chart palette.

Simulates protanopia, deuteranopia and tritanopia (severity 1.0) and reports the
smallest CIELAB colour distance between any two colours that are drawn together
in a chart. A pair below THRESHOLD fails.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.views_common import BLACK, BLUE, METHOD_STYLE, ORANGE, PALETTE  # noqa: E402

THRESHOLD = 15.0  # CIELAB distance; below this two colours are hard to tell apart

# Severity 1.0, applied to linear RGB.
MATRICES = {
    "protanopia": np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]]),
    "deuteranopia": np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]]),
    "tritanopia": np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]]),
}

# Sets of colours that appear in the same chart.
GROUPS = {
    "Bars and lines (blue, orange)": [BLUE, ORANGE],
    "Interval chart (blue circle, red diamond)": [METHOD_STYLE["dataset_size"]["color"], METHOD_STYLE["tokens_processed"]["color"]],
    "Lens 1 scatter (palette plus black)": PALETTE + [BLACK],
    "Lens 3 scatter (blue, black, orange star)": [BLUE, BLACK, ORANGE],
}


def to_rgb(hex_colour: str) -> np.ndarray:
    return np.array([int(hex_colour[i : i + 2], 16) for i in (1, 3, 5)]) / 255


def to_linear(rgb: np.ndarray) -> np.ndarray:
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def to_lab(linear: np.ndarray) -> np.ndarray:
    xyz_matrix = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = xyz_matrix @ linear / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def simulate(hex_colour: str, matrix: np.ndarray | None) -> np.ndarray:
    linear = to_linear(to_rgb(hex_colour))
    if matrix is not None:
        linear = np.clip(matrix @ linear, 0, 1)
    return to_lab(linear)


def main() -> int:
    failures = 0
    for group, colours in GROUPS.items():
        print(f"\n{group}")
        for condition, matrix in {"normal vision": None, **MATRICES}.items():
            labs = {c: simulate(c, matrix) for c in colours}
            pairs = [(float(np.linalg.norm(labs[a] - labs[b])), a, b) for a, b in itertools.combinations(colours, 2)]
            worst = min(pairs)
            status = "pass" if worst[0] >= THRESHOLD else "FAIL"
            failures += status == "FAIL"
            print(f"  {condition:<14} closest pair {worst[1]} / {worst[2]}  distance {worst[0]:5.1f}  {status}")
    print(f"\n{failures} pair(s) below {THRESHOLD}. Marker shapes also differ, so colour is not the only cue.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

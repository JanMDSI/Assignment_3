"""Chart audit: every chart the app draws has labelled axes, and log axes say so."""

from pathlib import Path

import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def _draw_all_charts(monkeypatch) -> list:
    figures: list = []
    monkeypatch.setattr(st, "plotly_chart", lambda fig, *args, **kwargs: figures.append(fig))
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    assert not app.exception
    return figures


def test_app_draws_at_least_four_charts(monkeypatch) -> None:
    assert len(_draw_all_charts(monkeypatch)) >= 4


def test_every_axis_has_a_title_and_log_axes_say_so(monkeypatch) -> None:
    for number, fig in enumerate(_draw_all_charts(monkeypatch), start=1):
        for name in ("xaxis", "yaxis"):
            axis = fig.layout[name]
            title = (axis.title.text or "").strip()
            assert title, f"chart {number}: {name} has no title"
            if axis.type == "log":
                assert "log scale" in title, f"chart {number}: log {name} does not say 'log scale': {title!r}"
                

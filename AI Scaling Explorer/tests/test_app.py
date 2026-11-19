from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def _app() -> AppTest:
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()


def test_opening_screen_shows_question_and_answer() -> None:
    app = _app()
    assert not app.exception
    assert app.title[0].value == "The Rise of Artificial Intelligence: Data Narrative Studio"
    assert any("Question:" in item.value for item in app.markdown)
    assert len(app.success) == 1 and "training data" in app.success[0].value
    assert len(app.metric) >= 8


def test_filters_change_what_is_shown() -> None:
    app = _app()
    before = app.info[0].value
    app.multiselect(key="domains").set_value(["Vision"]).run()
    assert not app.exception and app.info[0].value != before


def test_empty_filter_shows_a_helpful_message() -> None:
    app = _app()
    app.multiselect(key="organizations").set_value(["OpenAI"]).run()
    app.multiselect(key="domains").set_value(["Biology"]).run()
    assert not app.exception
    assert len(app.warning) == 1 and "widening" in app.warning[0].value

from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def _app() -> AppTest:
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()


def test_opening_screen_shows_question_and_answer() -> None:
    app = _app()
    assert not app.exception
    assert app.title[0].value == "AI Scaling Explorer"
    assert any("Question:" in item.value for item in app.markdown)
    assert len(app.success) == 1 and "Counting data as" in app.success[0].value
    assert len(app.metric) >= 5
    assert any("Cross check from lens 4" in item.value for item in app.markdown)


def test_lens_one_filters_change_what_is_shown() -> None:
    app = _app()
    before = app.info[0].value
    app.multiselect(key="l1_domains").set_value(["Vision"]).run()
    assert not app.exception and app.info[0].value != before


def test_lens_one_empty_filter_shows_a_helpful_message() -> None:
    app = _app()
    app.multiselect(key="l1_orgs").set_value(["OpenAI"]).run()
    app.multiselect(key="l1_domains").set_value(["Biology"]).run()
    assert not app.exception
    assert any("widening" in item.value for item in app.warning)


def test_lens_three_toggle_changes_the_estimate() -> None:
    app = _app()
    app.checkbox(key="l3_exclude").set_value(False).run()
    assert not app.exception


def test_lens_four_search_and_empty_state() -> None:
    app = _app()
    app.text_input(key="l4_search").set_value("gpt").run()
    assert not app.exception
    app.text_input(key="l4_search").set_value("zzzzzz").run()
    assert not app.exception and any("widening" in item.value for item in app.warning)

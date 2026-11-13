"""Exercise real Streamlit reruns without making paid API requests."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.database import get_evaluations, save_evaluation
from src.evaluation import RUBRIC, make_evaluation
from src.openai_client import GenerationError

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def configured_app(monkeypatch, tmp_path):
    values = {
        "OPENAI_API_KEY": "test-key-not-real",
        "OPENAI_MODEL": "gpt-4.1-mini",
        "OPENAI_USE_TEMPERATURE": "true",
        "OPENAI_TEMPERATURE_A": "0.4",
        "OPENAI_TEMPERATURE_B": "0.9",
        "OPENAI_MAX_OUTPUT_TOKENS": "1024",
        "EVALUATOR_DB_PATH": str(tmp_path / "evaluations.db"),
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr("src.openai_client.generate_pair", lambda *args, **kwargs: ("Answer A preserves the deadline.", "Answer B omits the deadline."))
    return AppTest.from_file(ROOT / "app.py", default_timeout=30), tmp_path / "evaluations.db"


def button(app, label):
    return next(item for item in app.button if item.label == label)


def test_all_pages_open_without_key_or_data(configured_app, monkeypatch):
    app, db = configured_app
    monkeypatch.setenv("OPENAI_API_KEY", "")
    app.run()
    assert not app.exception
    for page in sorted((ROOT / "pages").glob("*.py")):
        app.switch_page(f"pages/{page.name}").run()
        assert not app.exception, page.name
        if page.name == "1_Compare_Responses.py":
            assert button(app, "Generate Responses").disabled
            assert any("OPENAI_API_KEY" in message.value for message in app.info)
    assert db.is_file()


def test_compare_validate_save_and_inspect(configured_app):
    app, db = configured_app
    app.run().switch_page("pages/1_Compare_Responses.py").run()
    app.text_area(key="prompt_input").input("Rewrite the request politely.")
    app.text_area(key="constraints_input").input("Keep today's deadline.")
    button(app, "Generate Responses").click().run()
    assert not app.exception
    pair_id = app.session_state["comparison"]["id"]
    button(app, "Save Evaluation").click().run()
    assert app.error
    assert get_evaluations(db).empty
    app.radio(key=f"{pair_id}_preference").set_value("A")
    app.text_area(key=f"{pair_id}_rationale").input("A keeps the explicit deadline; B loses it.")
    app.slider(key=f"{pair_id}_a_instruction_following").set_value(5)
    app.slider(key=f"{pair_id}_b_instruction_following").set_value(1)
    button(app, "Save Evaluation").click().run()
    assert not app.exception
    assert app.success
    assert button(app, "Save Evaluation").disabled
    rows = get_evaluations(db)
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["preferred_response"] == "A"
    assert row["A_instruction_following"] == 5
    assert row["B_instruction_following"] == 1
    assert row["constraints"] == "Keep today's deadline."
    assert row["rationale"] == "A keeps the explicit deadline; B loses it."
    app.run()
    assert len(get_evaluations(db)) == 1
    app.switch_page("pages/2_Evaluation_History.py").run()
    assert not app.exception
    assert app.dataframe[0].value.shape[0] == 1
    app.text_input[0].input("no match").run()
    assert app.dataframe[0].value.empty
    assert not app.exception


def test_navigation_preserves_judgment_and_changed_prompt_blocks_save(configured_app):
    app, db = configured_app
    app.run().switch_page("pages/1_Compare_Responses.py").run()
    app.text_area(key="prompt_input").input("Original prompt")
    button(app, "Generate Responses").click().run()
    pair_id = app.session_state["comparison"]["id"]
    app.slider(key=f"{pair_id}_a_clarity").set_value(4)
    app.radio(key=f"{pair_id}_preference").set_value("Tie")
    app.text_area(key=f"{pair_id}_rationale").input("Equally useful overall.").run()
    app.switch_page("pages/5_Concepts.py").run()
    app.switch_page("pages/1_Compare_Responses.py").run()
    assert app.slider(key=f"{pair_id}_a_clarity").value == 4
    assert app.radio(key=f"{pair_id}_preference").value == "Tie"
    assert app.text_area(key=f"{pair_id}_rationale").value == "Equally useful overall."
    app.text_area(key="prompt_input").input("Different prompt").run()
    assert button(app, "Save Evaluation").disabled
    assert app.warning
    assert get_evaluations(db).empty


def test_generation_failure_keeps_previous_pair(configured_app, monkeypatch):
    app, _ = configured_app
    app.run().switch_page("pages/1_Compare_Responses.py").run()
    app.text_area(key="prompt_input").input("A prompt")
    button(app, "Generate Responses").click().run()
    original_id = app.session_state["comparison"]["id"]

    def fail(*args, **kwargs):
        raise GenerationError("The service is temporarily unavailable.")

    monkeypatch.setattr("src.openai_client.generate_pair", fail)
    button(app, "Generate Responses").click().run()
    assert app.error[0].value == "The service is temporarily unavailable."
    assert app.session_state["comparison"]["id"] == original_id
    assert not app.exception


def test_sample_populates_compare_and_concepts_slider(configured_app):
    app, _ = configured_app
    app.run().switch_page("pages/6_Sample_Prompts.py").run()
    app.button(key="sample_0").click().run()
    assert not app.exception
    assert "Explain gravity" in app.text_area(key="prompt_input").value
    assert "3 sentences" in app.text_area(key="constraints_input").value
    app.switch_page("pages/5_Concepts.py").run()
    assert app.warning
    app.slider[0].set_value(50).run()
    assert app.success


def test_populated_analytics_and_reward_model(configured_app):
    app, db = configured_app
    for index in range(12):
        prefer_a = index % 2 == 0
        record = make_evaluation(
            prompt=f"Task {index}", constraints="", response_a="A", response_b="B",
            preferred_response="A" if prefer_a else "B", rationale="Test fixture judgment.",
            scores_a=dict.fromkeys(RUBRIC, 5 if prefer_a else 2),
            scores_b=dict.fromkeys(RUBRIC, 2 if prefer_a else 5),
        )
        save_evaluation(record, db)
    app.run().switch_page("pages/3_Analytics.py").run()
    assert not app.exception
    assert app.metric[0].value == "12"
    app.switch_page("pages/4_Reward_Model.py").run()
    assert not app.exception
    assert app.metric[0].value == "9"
    assert app.metric[1].value == "3"
    assert app.selectbox

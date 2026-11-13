"""Generate an independent pair and record one deliberate human judgment."""

import sqlite3
from uuid import uuid4

import streamlit as st

from src.database import init_db, save_evaluation
from src.evaluation import RUBRIC, ValidationError, calculate_average, make_evaluation
from src.openai_client import GenerationError, generate_pair
from src.ui import remember_input, settings_or_stop


def remember_rating(pair_id: str, side: str, category: str, widget_key: str) -> None:
    """Preserve unfinished ratings across page navigation."""
    pair = st.session_state.get("comparison")
    if pair and pair["id"] == pair_id:
        pair[f"scores_{side}"][category] = st.session_state[widget_key]


def remember_judgment(pair_id: str, field: str, widget_key: str) -> None:
    """Preserve preference and rationale with their original response pair."""
    pair = st.session_state.get("comparison")
    if pair and pair["id"] == pair_id:
        pair[field] = st.session_state[widget_key]


st.title("Compare Responses")
st.write("Read both answers, score each against the same rubric, then explain your preference.")
settings = settings_or_stop()
st.session_state.setdefault("prompt_input", st.session_state.get("draft_prompt", ""))
st.session_state.setdefault("constraints_input", st.session_state.get("draft_constraints", ""))
prompt = st.text_area("Prompt", key="prompt_input", height=130, max_chars=20000, placeholder="What would you like the model to do?", on_change=remember_input, args=("prompt_input", "draft_prompt"))
constraints = st.text_area("Optional constraints", key="constraints_input", height=80, max_chars=5000, placeholder="For example: exactly three sentences, simple language, keep the deadline.", on_change=remember_input, args=("constraints_input", "draft_constraints"))
st.page_link("pages/6_Sample_Prompts.py", label="Need an idea? Browse sample prompts", icon="💡")
with st.expander("Generation settings"):
    st.write(f"Configured model: **{settings.model}**")
    st.caption("Change OPENAI_MODEL in .env to use another model. Regenerate to apply different settings.")
    use_temperature = st.checkbox("Send temperature to the model", value=settings.use_temperature, help="Turn this off if your chosen model does not support temperature. The two independent writing approaches still provide variation.")
    left, right = st.columns(2)
    with left:
        temperature_a = st.slider("Response A temperature", 0.0, 2.0, settings.temperature_a, 0.1, disabled=not use_temperature)
    with right:
        temperature_b = st.slider("Response B temperature", 0.0, 2.0, settings.temperature_b, 0.1, disabled=not use_temperature)
    st.caption("A uses a direct writing approach; B uses an alternative structure when compatible with your request. Temperature changes randomness, not quality. A/B are candidates from the same model, not different models.")

has_key = bool(settings.api_key and settings.api_key != "your_api_key_here")
if not has_key:
    st.info("To generate responses, copy .env.example to .env and set OPENAI_API_KEY to your own API key. The other learning pages work without a key.")
st.caption("Generation sends your prompt and constraints to OpenAI. A pair uses two API calls; an identical pair may trigger one additional attempt. API usage can incur charges.")
if st.button("Generate Responses", type="primary", disabled=not has_key):
    if not prompt.strip():
        st.error("Enter a prompt before generating responses.")
    else:
        try:
            with st.spinner("Generating two independent responses…"):
                response_a, response_b = generate_pair(prompt.strip(), constraints.strip(), settings=settings, temperature_a=temperature_a, temperature_b=temperature_b, use_temperature=use_temperature)
            st.session_state["comparison"] = {
                "id": str(uuid4()), "prompt": prompt.strip(), "constraints": constraints.strip(),
                "response_a": response_a, "response_b": response_b, "model": settings.model,
                "scores_a": dict.fromkeys(RUBRIC, 3), "scores_b": dict.fromkeys(RUBRIC, 3),
                "preferred_response": None, "rationale": "", "saved": False,
            }
        except GenerationError as error:
            st.error(str(error))

pair = st.session_state.get("comparison")
if not pair:
    st.info("Your response pair will appear here after generation. No data is saved until you choose Save Evaluation.")
    st.stop()

st.divider()
st.subheader("Your comparison")
st.caption(f"Generated with {pair['model']} · Sliders start at 3; review every category before saving.")
with st.expander("Prompt and constraints used for this pair"):
    st.text(pair["prompt"])
    st.text(pair["constraints"] or "No additional constraints.")
stale = prompt.strip() != pair["prompt"] or constraints.strip() != pair["constraints"]
if stale:
    st.warning("The input has changed since these responses were generated. Generate a new pair, or restore the original input, before saving this evaluation.")

for side, column in zip(("a", "b"), st.columns(2, gap="large")):
    with column:
        st.subheader(f"Response {side.upper()}")
        with st.container(border=True):
            st.text(pair[f"response_{side}"])
        for category, details in RUBRIC.items():
            widget_key = f"{pair['id']}_{side}_{category}"
            st.slider(details["label"], 1, 5, pair[f"scores_{side}"][category], key=widget_key, help=details["description"], disabled=pair["saved"], on_change=remember_rating, args=(pair["id"], side, category, widget_key))
        st.metric(f"Response {side.upper()} average", f"{calculate_average(pair[f'scores_{side}']):.2f} / 5")

st.caption("1 = poor · 2 = needs work · 3 = adequate · 4 = good · 5 = excellent. Your preference may differ from the average; explain the tradeoff.")
preference_key = f"{pair['id']}_preference"
options = ["A", "B", "Tie"]
current_preference = pair["preferred_response"]
st.radio("Which response is better?", options, index=options.index(current_preference) if current_preference in options else None, format_func=lambda value: f"Response {value}" if value != "Tie" else "Tie", key=preference_key, horizontal=True, disabled=pair["saved"], on_change=remember_judgment, args=(pair["id"], "preferred_response", preference_key))
rationale_key = f"{pair['id']}_rationale"
st.text_area("Why? Give a short rationale", value=pair["rationale"], key=rationale_key, max_chars=5000, placeholder="Mention a requirement, factual issue, or tradeoff that informed your choice.", disabled=pair["saved"], on_change=remember_judgment, args=(pair["id"], "rationale", rationale_key))
if st.button("Save Evaluation", type="primary", disabled=pair["saved"] or stale):
    try:
        record = make_evaluation(
            prompt=pair["prompt"], constraints=pair["constraints"],
            response_a=pair["response_a"], response_b=pair["response_b"],
            preferred_response=pair["preferred_response"], rationale=pair["rationale"],
            scores_a=pair["scores_a"], scores_b=pair["scores_b"], evaluation_id=pair["id"],
        )
        init_db(settings.db_path)
        save_evaluation(record, settings.db_path)
        pair["saved"] = True
        st.rerun()
    except (ValidationError, ValueError) as error:
        st.error(str(error))
    except (sqlite3.Error, OSError):
        st.error("Could not save this evaluation. Check the database folder permissions and try again. Your comparison is still available here.")
if pair["saved"]:
    st.success("Evaluation saved. You can find it in Evaluation History. Generate another pair to continue.")
    st.page_link("pages/2_Evaluation_History.py", label="View saved evaluations", icon="🗂️")

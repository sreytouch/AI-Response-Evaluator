"""Small UI helpers shared by Streamlit pages."""

import sqlite3

import pandas as pd
import streamlit as st

from src.config import ConfigurationError, Settings, get_settings
from src.database import get_evaluations, init_db


def settings_or_stop() -> Settings:
    """Report configuration errors without exposing environment secrets."""
    try:
        return get_settings()
    except ConfigurationError as error:
        st.error(str(error))
        st.stop()


def load_evaluations() -> pd.DataFrame:
    """Initialize storage and return fresh data on every page visit."""
    settings = settings_or_stop()
    try:
        init_db(settings.db_path)
        return get_evaluations(settings.db_path)
    except (sqlite3.Error, OSError):
        st.error("Could not open the evaluation database. Check that the data folder is writable and EVALUATOR_DB_PATH points to a SQLite file.")
        st.stop()


def remember_input(widget_key: str, draft_key: str) -> None:
    """Keep unfinished prompt inputs when navigating between pages."""
    st.session_state[draft_key] = st.session_state[widget_key]


def choose_sample(prompt: str, constraints: str = "") -> None:
    """Populate the comparison inputs before switching pages."""
    st.session_state["draft_prompt"] = prompt
    st.session_state["draft_constraints"] = constraints
    st.session_state["prompt_input"] = prompt
    st.session_state["constraints_input"] = constraints
    st.switch_page("pages/1_Compare_Responses.py")

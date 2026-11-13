"""Filter, inspect, and export collected human preference data."""

import pandas as pd
import streamlit as st

from src.evaluation import RUBRIC, export_csv, filter_evaluations
from src.ui import load_evaluations

st.title("Evaluation History")
st.write("Every saved judgment includes its original prompt, both answers, all scores, and your explanation.")
frame = load_evaluations()
if frame.empty:
    st.info("No evaluations yet. Compare two responses and save your first judgment.")
    st.page_link("pages/1_Compare_Responses.py", label="Start a comparison", icon="⚖️")
    st.stop()

left, right = st.columns([1, 2])
with left:
    preference = st.selectbox("Preferred response", ["All", "A", "B", "Tie"])
with right:
    search = st.text_input("Search prompt, constraints, responses, or rationale", placeholder="Enter a word or phrase")
start_date = end_date = None
if st.checkbox("Filter by date (UTC)"):
    dates = pd.to_datetime(frame["timestamp"], utc=True).dt.date
    date_range = st.date_input("Date range", value=(dates.min(), dates.max()), min_value=dates.min(), max_value=dates.max())
    if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
        st.info("Select both a start and an end date.")
        st.stop()
    start_date, end_date = date_range
filtered = filter_evaluations(frame, preferred_response=None if preference == "All" else preference, start_date=start_date, end_date=end_date, search=search)
st.caption(f"Showing {len(filtered)} of {len(frame)} evaluations. Dates and timestamps use UTC.")
st.dataframe(filtered, hide_index=True, width="stretch")
st.download_button("Download as CSV", data=export_csv(filtered), file_name="human_preferences.csv", mime="text/csv", disabled=filtered.empty)
st.caption("The export includes all columns for the filtered rows. Spreadsheet-formula-like text is prefixed with an apostrophe; database text is unchanged.")
if filtered.empty:
    st.info("No evaluations match these filters.")
    st.stop()

st.subheader("Inspect an evaluation")
rows = {row["evaluation_id"]: row for row in filtered.to_dict("records")}
evaluation_id = st.selectbox("Evaluation", list(rows), format_func=lambda key: f"{rows[key]['timestamp'][:19]} · {rows[key]['prompt'][:65]} · {key[:8]}")
record = rows[evaluation_id]
st.caption(f"ID: {evaluation_id}")
st.markdown("**Prompt**")
st.text(record["prompt"])
st.markdown("**Constraints**")
st.text(record["constraints"] or "None")
for side, column in zip(("a", "b"), st.columns(2, gap="large")):
    with column:
        st.subheader(f"Response {side.upper()}")
        with st.container(border=True):
            st.text(record[f"response_{side}"])
        st.metric("Overall average", f"{record[f'{side.upper()}_average_score']:.2f} / 5")
scores = pd.DataFrame({"Category": [value["label"] for value in RUBRIC.values()], "Response A": [record[f"A_{key}"] for key in RUBRIC], "Response B": [record[f"B_{key}"] for key in RUBRIC]})
st.dataframe(scores, hide_index=True, width="stretch")
st.markdown(f"**Preference:** {'Tie' if record['preferred_response'] == 'Tie' else 'Response ' + record['preferred_response']}")
st.markdown("**Rationale**")
st.text(record["rationale"])

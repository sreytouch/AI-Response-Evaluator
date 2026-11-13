"""Clickable sample tasks for consistent evaluation practice."""

import streamlit as st

from src.sample_prompts import SAMPLE_PROMPTS
from src.ui import choose_sample

st.title("Sample Prompts")
st.write("Choose a task to populate the comparison screen. Look for specific requirements that an answer could miss.")
columns = st.columns(2, gap="large")
for index, sample in enumerate(SAMPLE_PROMPTS):
    with columns[index % 2].container(border=True):
        st.subheader(sample["title"])
        st.write(sample["prompt"])
        st.caption(sample["constraints"])
        if st.button("Use this prompt", key=f"sample_{index}", width="stretch"):
            choose_sample(sample["prompt"], sample["constraints"])

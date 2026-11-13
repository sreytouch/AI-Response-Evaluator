"""Explore saved human judgments without implying model training occurred."""

import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.ticker import MaxNLocator

from src.analytics import dashboard_summary, preference_counts, rubric_averages, score_trend
from src.ui import load_evaluations

st.title("Analytics dashboard")
st.write("Look for patterns in your saved judgments. These scores describe your evaluations, not a universal model benchmark.")
frame = load_evaluations()
summary = dashboard_summary(frame)
columns = st.columns(4)
for column, label, value in zip(
    columns,
    ["Evaluations", "Prefer A", "Prefer B", "Ties"],
    [str(summary["total"]), f"{summary['percent_A']:.1f}%", f"{summary['percent_B']:.1f}%", f"{summary['percent_Tie']:.1f}%"],
):
    column.metric(label, value)

if frame.empty:
    st.info("Save your first evaluation on Compare Responses to start building this dashboard.")
    st.stop()

left, right = st.columns(2)
left.metric("Average score · Response A", f"{summary['average_A']:.2f} / 5")
right.metric("Average score · Response B", f"{summary['average_B']:.2f} / 5")

left, right = st.columns(2)
with left:
    st.subheader("Preference counts")
    counts = preference_counts(frame)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(["Response A", "Response B", "Tie"], counts.values, color=["#3665C2", "#218D88", "#9AA5B1"])
    ax.set_ylabel("Evaluations")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    for index, count in enumerate(counts.values):
        ax.annotate(str(count), (index, count), xytext=(0, 5), textcoords="offset points", ha="center")
    ax.set_ylim(0, max(int(counts.max()) * 1.25, 1.5))
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.subheader("Average rubric scores")
    averages = rubric_averages(frame)
    fig, ax = plt.subplots(figsize=(6, 4))
    averages.plot.barh(ax=ax, color=["#3665C2", "#218D88"])
    ax.set_xlim(0, 5)
    ax.set_xlabel("Mean human rating (1–5)")
    ax.legend(title="Response")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with st.expander("View chart data"):
    st.dataframe(averages.round(2), width="stretch")
    st.caption("Rubric averages and overall averages include ties.")

st.subheader("Scores over time")
trend = score_trend(frame)
if len(trend) < 2:
    st.info("Save evaluations at two or more different times to see the trend.")
else:
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for label, color in (("A", "#3665C2"), ("B", "#218D88")):
        ax.plot(trend.index, trend[f"{label}_average_score"], marker="o", markersize=4, label=f"Response {label}", color=color)
    ax.set_ylim(1, 5)
    ax.set_ylabel("Mean rubric score")
    ax.set_xlabel("Evaluation time (UTC)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Evaluations with the same timestamp are averaged. Changes can reflect different prompts or evaluators; they do not prove the underlying AI improved.")

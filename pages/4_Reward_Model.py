"""Fit and inspect a small rubric-based preference classifier."""

import streamlit as st

from src.reward_model import predict_preference, train_reward_model, training_requirement
from src.ui import load_evaluations

st.title("Reward model demo")
st.write("Can six human ratings help predict which response a person prefers?")
st.info("This is a toy approximation: logistic regression learns from rubric-score differences. It does not read the answers, train an LLM, or run RLHF/PPO.")
frame = load_evaluations()
decisive_count = int(frame["preferred_response"].isin(["A", "B"]).sum()) if not frame.empty else 0
tie_count = len(frame) - decisive_count
st.caption(f"Available: {decisive_count} non-tied evaluations · {tie_count} ties excluded from training")

with st.expander("How this demonstration works", expanded=True):
    st.markdown(
        "1. For each category, subtract **B's score from A's score**.\n"
        "2. Use the human preference as the label: **A = 1, B = 0**.\n"
        "3. Set aside 25% of the examples, preserving the preference balance as closely as possible.\n"
        "4. Fit logistic regression on the remaining 75%, then check predictions on the held-out examples."
    )
    st.caption("The split uses a fixed random seed for reproducibility. The displayed model remains trained only on the training subset; it is not refit on held-out examples.")

reason = training_requirement(frame)
if reason:
    st.warning(reason)
    st.stop()

try:
    result = train_reward_model(frame)
except ValueError as error:
    st.error(str(error))
    st.stop()

columns = st.columns(4)
columns[0].metric("Training examples", len(result.training_ids))
columns[1].metric("Held-out examples", len(result.held_out_ids))
columns[2].metric("Held-out accuracy", f"{result.accuracy:.0%}")
columns[3].metric("Majority baseline", f"{result.baseline_accuracy:.0%}")
st.caption("The baseline always picks the most common training preference. Small datasets make accuracy unstable. Repeated or very similar prompts can make a random split look better than performance on genuinely new tasks.")

st.subheader("What did the model learn?")
st.dataframe(result.coefficients, hide_index=True, width="stretch")
st.write("A positive coefficient makes A more likely when A's rating exceeds B's in that category. A negative coefficient goes the other way. Coefficients show associations in this dataset, not objective importance or causation.")
st.caption(f"Intercept: {result.model.intercept_[0]:.3f}. This allows the classifier to learn a general preference for A or B, including possible position bias.")

st.subheader("Predict a saved evaluation")
records = {str(row["evaluation_id"]): row.to_dict() for _, row in frame.iterrows()}
selected = st.selectbox(
    "Evaluation",
    options=list(records),
    format_func=lambda key: f"{records[key]['timestamp'][:16]} · {records[key]['prompt'][:70]} · {key[:8]}",
)
record = records[selected]
prediction = predict_preference(result, record)
st.write(f"**{prediction['membership']}** · Human preference: **{record['preferred_response']}**")
st.write(f"Model prediction: **Response {prediction['preferred_response']}**")
left, right = st.columns(2)
left.metric("Estimated P(A preferred)", f"{prediction['probability_a']:.1%}")
right.metric("Estimated P(B preferred)", f"{prediction['probability_b']:.1%}")
st.caption("These are uncalibrated model estimates, not confidence in factual correctness. This binary model cannot predict a tie.")
if prediction["membership"] == "Training example":
    st.caption("This example helped fit the model; its prediction is not an independent test.")
with st.expander("Inspect the example"):
    st.write("**Prompt**")
    st.text(record["prompt"])
    if record["constraints"]:
        st.write("**Constraints**")
        st.text(record["constraints"])
    left, right = st.columns(2)
    left.write("**Response A**")
    left.text(record["response_a"])
    right.write("**Response B**")
    right.text(record["response_b"])
    st.write("**Human rationale**")
    st.text(record["rationale"])

st.subheader("What would real reward modeling add?")
st.write("A real text reward model can learn from prompts and responses rather than requiring a person to score every new answer first. This demo only learns a relationship among ratings and choices. Improving its classifier does not update the response-generating model.")

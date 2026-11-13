"""Beginner explanations and an interactive example of a misspecified reward."""

import pandas as pd
import streamlit as st

st.title("Concepts, without the jargon")
st.write("Use this glossary to connect each part of the app to the larger model-improvement workflow.")

concepts = {
    "Pretraining": "A model first learns broad language patterns from large datasets. The OpenAI model already went through this stage before this app calls it.",
    "Post-training": "Additional work after pretraining makes a model more useful for particular goals. Instruction tuning and preference-based learning are examples. This app explores the data and evaluation side.",
    "Supervised Fine-Tuning (SFT)": "Train a model using examples of a prompt and a desired answer. A carefully reviewed winning answer could become an SFT example; this app collects comparisons but does not fine-tune a model.",
    "Human Feedback": "A person's judgment about an answer. Here you provide six scores, a preference, and a rationale explaining your decision.",
    "Preference Data": "Examples of which answer someone prefers for the same request. The SQLite database stores the prompt, both answers, ratings, choice, and explanation. A tie is useful feedback too.",
    "Reward Model": "A model that estimates how desirable an output is. Our demo uses rating differences to predict A versus B; production text reward models may use the prompt and answer directly.",
    "Reward Signal": "A number used to guide an optimization process. An estimated preference can be one ingredient, but a high score is only as useful as the goal it measures.",
    "Reinforcement Learning": "An agent changes its behavior based on rewards from its actions. This app does not run a reinforcement-learning algorithm or update the AI's weights.",
    "RLHF": "Reinforcement Learning from Human Feedback: people provide judgments that can help define a reward, which can then guide model optimization. We demonstrate collection and a toy preference model, not the full training pipeline.",
    "Reward Hacking": "An optimizer can exploit a scoring rule in a way that earns a high score but misses the real goal. The interactive example below shows a misaligned reward; no optimizer is actually trained here.",
    "Evaluation": "Check performance against explicit goals. The six-category rubric separates factual correctness from following constraints, relevance, clarity, completeness, and style.",
    "Hill Climbing / Iterative Improvement": "Make a small change, evaluate it, and retain improvements. Try clearer prompt instructions, collect fresh feedback, and compare the results. Avoid choosing only easy examples or reusing a test set to tune repeatedly.",
}
for title, explanation in concepts.items():
    with st.expander(title):
        st.write(explanation)

st.divider()
st.subheader("When a reward measures the wrong thing")
st.write("**Request:** Rewrite ‘I need your answer today’ to sound more polite without changing the deadline.")
left, right = st.columns(2)
with left:
    st.markdown("#### Response 1")
    st.write("Could you please provide your answer by the end of today?")
    st.caption("Polite, and keeps the deadline.")
with right:
    st.markdown("#### Response 2")
    st.write("Whenever you have a chance, I would really appreciate hearing from you.")
    st.caption("Sounds very polite, but removes the deadline.")

st.write("Imagine the following **invented ratings**, used only to demonstrate a scoring rule:")
ratings = pd.DataFrame({"Response": ["Response 1", "Response 2"], "Politeness": [4, 5], "Preserves deadline": [5, 1]})
st.dataframe(ratings, hide_index=True, width="stretch")
weight = st.slider("Weight placed on politeness (%)", min_value=0, max_value=100, value=90, step=5)
politeness_weight = weight / 100
score_1 = politeness_weight * 4 + (1 - politeness_weight) * 5
score_2 = politeness_weight * 5 + (1 - politeness_weight) * 1
st.caption(f"Score = {politeness_weight:.2f} × politeness + {1 - politeness_weight:.2f} × deadline preservation")
left, right = st.columns(2)
left.metric("Response 1 reward", f"{score_1:.2f} / 5")
right.metric("Response 2 reward", f"{score_2:.2f} / 5")
if abs(score_1 - score_2) < 0.0001:
    st.warning("The scoring rule gives a tie, even though Response 2 loses an explicit requirement.")
elif score_2 > score_1:
    st.warning("The scoring rule prefers Response 2, but the user required today's deadline. It rewards politeness at the expense of instruction following.")
else:
    st.success("This weighting prefers Response 1, which keeps the requested deadline.")
st.write("This is reward misalignment. If a model learned to remove deadlines to maximize this rule, that would be a reward-hacking behavior. A better evaluation checks all important requirements; a hard constraint such as keeping the deadline may need to be a pass/fail condition, not a tradeable style point.")
st.caption("Moving this slider changes the illustration only. It does not modify your saved evaluations or train a model.")

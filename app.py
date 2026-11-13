"""Streamlit entry point: navigation and a short introduction."""

import streamlit as st

st.set_page_config(page_title="AI Response Evaluator", page_icon="⚖️", layout="wide")


def home() -> None:
    """Explain the project before inviting a first evaluation."""
    st.caption("A HUMAN FEEDBACK LAB")
    st.title("AI Response Evaluator")
    st.markdown("### Two answers. One thoughtful comparison.")
    st.write(
        "Generate two responses to the same prompt, judge each with a clear rubric, "
        "and record which one you prefer. Turn individual judgments into a small, "
        "inspectable dataset."
    )
    left, right = st.columns([2, 1], gap="large")
    with left:
        st.subheader("The workflow")
        st.code(
            "Prompt\n  ↓\nGenerate two responses\n  ↓\nHuman evaluation\n  ↓\nPreference data\n  ↓\nAnalyze results\n  ↓\nImprove instructions and evaluate again",
            language=None,
        )
        st.page_link("pages/1_Compare_Responses.py", label="Start a comparison", icon="⚖️")
        st.page_link("pages/6_Sample_Prompts.py", label="Choose a sample prompt", icon="💡")
    with right:
        with st.container(border=True):
            st.markdown("**What you will collect**")
            st.write("A prompt, two answers, 12 rubric scores, a preference, and your rationale.")
            st.markdown("**What you will learn**")
            st.write("How human feedback can help define and measure better model behavior.")
            st.markdown("**What runs locally**")
            st.write("The interface, SQLite database, charts, and toy reward model.")
    st.subheader("The ideas behind the app")
    concepts = [
        ("Human preference data", "A record of which answer a person prefers for the same request, with an explanation of why."),
        ("Supervised fine-tuning (SFT)", "Teaching a pretrained model using examples of prompts and desired answers. This app collects evaluations; it does not fine-tune a language model."),
        ("RLHF", "Reinforcement learning from human feedback: human judgments help define a reward, which can guide model training. Here you explore the feedback stage, not the training loop."),
        ("Reward model", "A model that estimates what people will prefer. Our small demo predicts preferences from the scores you supply, rather than reading answer text."),
        ("Evaluation", "Checking a response against explicit criteria. Repeating these checks helps reveal strengths, failures, and tradeoffs."),
        ("Iterative improvement", "Look at failures, revise instructions, and compare new outputs. Improvement needs measurement; it is not guaranteed."),
    ]
    columns = st.columns(3)
    for index, (title, explanation) in enumerate(concepts):
        with columns[index % 3].container(border=True):
            st.markdown(f"**{title}**")
            st.write(explanation)
    st.info("Educational project: no PPO, language-model weight updates, or production RLHF training takes place.")
    st.caption("Generating responses sends the prompt and constraints to OpenAI and may incur API charges. Saved evaluations stay in your local database.")


navigation = st.navigation([
    st.Page(home, title="Home", icon="🏠", default=True),
    st.Page("pages/1_Compare_Responses.py", title="Compare Responses", icon="⚖️"),
    st.Page("pages/2_Evaluation_History.py", title="Evaluation History", icon="🗂️"),
    st.Page("pages/3_Analytics.py", title="Analytics Dashboard", icon="📊"),
    st.Page("pages/4_Reward_Model.py", title="Reward Model Demo", icon="🧪"),
    st.Page("pages/5_Concepts.py", title="Concepts", icon="📖"),
    st.Page("pages/6_Sample_Prompts.py", title="Sample Prompts", icon="💡"),
])
with st.sidebar:
    st.caption("Local learning project · Human judgments first")
navigation.run()

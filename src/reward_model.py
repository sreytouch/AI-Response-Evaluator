"""An educational preference classifier using human-provided rubric scores.

This classifier does not read response text, update an LLM, or perform RLHF.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.evaluation import RUBRIC

MIN_EXAMPLES = 10
MIN_PER_CLASS = 4


@dataclass
class RewardModelResult:
    """Model and the exact split used to report held-out performance."""

    model: LogisticRegression
    accuracy: float
    baseline_accuracy: float
    training_ids: tuple[str, ...]
    held_out_ids: tuple[str, ...]
    coefficients: pd.DataFrame
    decisive_count: int
    tie_count: int


def rubric_differences(frame: pd.DataFrame) -> pd.DataFrame:
    """Use A minus B per category; never use the preference label as a feature."""
    required = [f"{side}_{key}" for side in ("A", "B") for key in RUBRIC]
    if not set(required).issubset(frame.columns):
        raise ValueError("Each example needs all six rubric scores for both responses.")
    scores = frame[required].apply(pd.to_numeric, errors="coerce")
    values = scores.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 1).any() or (values > 5).any() or (values % 1 != 0).any():
        raise ValueError("Rubric scores must be whole numbers from 1 to 5.")
    return pd.DataFrame(
        {key: scores[f"A_{key}"] - scores[f"B_{key}"] for key in RUBRIC}, index=frame.index
    )


def training_requirement(frame: pd.DataFrame) -> str | None:
    """Explain why training cannot run, or return None when it can."""
    if frame.empty or "preferred_response" not in frame:
        return "Collect at least 10 non-tied evaluations, including at least 4 preferences for A and 4 for B."
    decisive = frame.loc[frame["preferred_response"].isin(["A", "B"])]
    counts = decisive["preferred_response"].value_counts()
    if len(decisive) < MIN_EXAMPLES or any(counts.get(label, 0) < MIN_PER_CLASS for label in ("A", "B")):
        return (
            f"Collected {len(decisive)} non-tied evaluations: {counts.get('A', 0)} prefer A and "
            f"{counts.get('B', 0)} prefer B. Collect at least 10 in total and at least 4 of each "
            "preference so both the training and held-out sets can include both choices. "
            "Keep your ratings honest; do not choose a response just to balance the data."
        )
    return None


def train_reward_model(frame: pd.DataFrame) -> RewardModelResult:
    """Train on 75% of decisive examples and score on the untouched 25%."""
    reason = training_requirement(frame)
    if reason:
        raise ValueError(reason)
    decisive = frame.loc[frame["preferred_response"].isin(["A", "B"])].copy()
    decisive = decisive.sort_values("evaluation_id").reset_index(drop=True)
    if decisive["evaluation_id"].duplicated().any():
        raise ValueError("Evaluation IDs must be unique before splitting the dataset.")
    features = rubric_differences(decisive)
    labels = (decisive["preferred_response"] == "A").astype(int)
    train_indices, test_indices = train_test_split(
        decisive.index.to_numpy(), test_size=0.25, random_state=42, stratify=labels
    )
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(features.loc[train_indices], labels.loc[train_indices])
    predictions = model.predict(features.loc[test_indices])
    majority_label = int(labels.loc[train_indices].mode().iloc[0])
    baseline = [majority_label] * len(test_indices)
    coefficients = pd.DataFrame(
        {
            "Category": [details["label"] for details in RUBRIC.values()],
            "Coefficient": model.coef_[0],
        }
    )
    return RewardModelResult(
        model=model,
        accuracy=float(accuracy_score(labels.loc[test_indices], predictions)),
        baseline_accuracy=float(accuracy_score(labels.loc[test_indices], baseline)),
        training_ids=tuple(decisive.loc[train_indices, "evaluation_id"].astype(str)),
        held_out_ids=tuple(decisive.loc[test_indices, "evaluation_id"].astype(str)),
        coefficients=coefficients,
        decisive_count=len(decisive),
        tie_count=int((frame["preferred_response"] == "Tie").sum()),
    )


def predict_preference(result: RewardModelResult, evaluation: dict[str, Any]) -> dict[str, Any]:
    """Predict from already-entered scores and identify in-sample examples."""
    features = rubric_differences(pd.DataFrame([evaluation]))
    probability_a = float(result.model.predict_proba(features)[0, list(result.model.classes_).index(1)])
    evaluation_id = str(evaluation["evaluation_id"])
    membership = (
        "Training example" if evaluation_id in result.training_ids
        else "Held-out example" if evaluation_id in result.held_out_ids
        else "Excluded tie / unseen example"
    )
    return {
        "preferred_response": "A" if probability_a >= 0.5 else "B",
        "probability_a": probability_a,
        "probability_b": 1.0 - probability_a,
        "membership": membership,
    }

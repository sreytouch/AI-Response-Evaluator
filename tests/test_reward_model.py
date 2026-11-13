"""Tests exercise tie exclusion, data sufficiency, and holdout separation."""

import pandas as pd
import pytest

from src.evaluation import RUBRIC
from src.reward_model import (
    predict_preference,
    rubric_differences,
    train_reward_model,
    training_requirement,
)


def dataset(count: int = 20) -> pd.DataFrame:
    records = []
    for index in range(count):
        choice = "A" if index % 2 == 0 else "B"
        record = {"evaluation_id": f"evaluation-{index:03}", "preferred_response": choice}
        for key in RUBRIC:
            record[f"A_{key}"] = 5 if choice == "A" else 2
            record[f"B_{key}"] = 2 if choice == "A" else 5
        records.append(record)
    return pd.DataFrame(records)


def test_differences_use_scores_not_the_label():
    frame = dataset(2)
    features = rubric_differences(frame)
    assert list(features.columns) == list(RUBRIC)
    assert list(features.iloc[0]) == [3] * 6
    frame["preferred_response"] = "Tie"
    pd.testing.assert_frame_equal(features, rubric_differences(frame))


@pytest.mark.parametrize("bad_value", [0, 6, 3.5, float("nan"), float("inf"), "bad"])
def test_invalid_scores_are_rejected(bad_value):
    frame = dataset(2).astype({"A_correctness": object})
    frame.loc[0, "A_correctness"] = bad_value
    with pytest.raises(ValueError, match="whole numbers"):
        rubric_differences(frame)


def test_not_enough_data_or_one_preference_is_explained():
    assert training_requirement(pd.DataFrame()) is not None
    assert training_requirement(dataset(4)) is not None
    frame = dataset()
    frame["preferred_response"] = "A"
    assert training_requirement(frame) is not None
    with pytest.raises(ValueError, match="non-tied"):
        train_reward_model(frame)


def test_ties_are_excluded_and_holdout_is_disjoint():
    frame = dataset()
    tie = frame.iloc[0].copy()
    tie["evaluation_id"] = "tie-only"
    tie["preferred_response"] = "Tie"
    frame = pd.concat([frame, pd.DataFrame([tie])], ignore_index=True)
    result = train_reward_model(frame)
    assert result.decisive_count == 20
    assert result.tie_count == 1
    assert len(result.training_ids) == 15
    assert len(result.held_out_ids) == 5
    assert set(result.training_ids).isdisjoint(result.held_out_ids)
    assert "tie-only" not in result.training_ids + result.held_out_ids
    assert result.accuracy == 1.0
    assert 0 <= result.baseline_accuracy <= 1
    assert len(result.coefficients) == 6
    assert result.model.n_features_in_ == 6
    for evaluation_id, expected_membership in [
        (result.training_ids[0], "Training example"),
        (result.held_out_ids[0], "Held-out example"),
        ("tie-only", "Excluded tie / unseen example"),
    ]:
        record = frame.loc[frame["evaluation_id"] == evaluation_id].iloc[0].to_dict()
        prediction = predict_preference(result, record)
        assert prediction["membership"] == expected_membership
        assert prediction["probability_a"] + prediction["probability_b"] == pytest.approx(1)


def test_split_is_reproducible_independent_of_row_order():
    frame = dataset()
    one = train_reward_model(frame)
    two = train_reward_model(frame.sample(frac=1, random_state=7))
    assert one.training_ids == two.training_ids
    assert one.held_out_ids == two.held_out_ids
    assert one.accuracy == two.accuracy


def test_duplicate_ids_cannot_leak_between_train_and_test():
    frame = dataset()
    frame.loc[1, "evaluation_id"] = frame.loc[0, "evaluation_id"]
    with pytest.raises(ValueError, match="unique"):
        train_reward_model(frame)

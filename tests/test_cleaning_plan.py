
import pandas as pd
import pytest

from data_cleaner import (
    validate_cleaning_plan,
    preview_cleaning_plan,
    apply_cleaning_plan,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "product": [" Laptop ", "Mouse", "Mouse"],
        "price": [1500.0, None, 100.0],
        "quantity": [2, 5, 5],
    })


def test_valid_plan(sample_df):
    plan = {
        "operations": [
            {"action": "fill_missing", "column": "price", "method": "median"}
        ]
    }

    result = validate_cleaning_plan(plan, sample_df)

    assert result[0]["requires_approval"] is True


def test_unknown_action(sample_df):
    plan = {
        "operations": [{"action": "execute_python", "code": "print(1)"}]
    }

    with pytest.raises(ValueError):
        validate_cleaning_plan(plan, sample_df)


def test_invalid_column(sample_df):
    plan = {
        "operations": [
            {"action": "fill_missing", "column": "unknown", "method": "median"}
        ]
    }

    with pytest.raises(ValueError):
        validate_cleaning_plan(plan, sample_df)


def test_approval_required(sample_df):
    plan = {
        "operations": [{"action": "remove_duplicates"}]
    }

    with pytest.raises(PermissionError):
        apply_cleaning_plan(sample_df, plan)


def test_original_data_unchanged(sample_df):
    original = sample_df.copy(deep=True)

    plan = {
        "operations": [
            {"action": "trim_whitespace", "column": "product"},
            {"action": "fill_missing", "column": "price", "method": "median"},
        ]
    }

    cleaned, log = apply_cleaning_plan(
        sample_df, plan, approved=True
    )

    pd.testing.assert_frame_equal(sample_df, original)
    assert cleaned["price"].isna().sum() == 0
    assert cleaned.loc[0, "product"] == "Laptop"
    assert len(log) == 2


def test_preview(sample_df):
    plan = {
        "operations": [
            {"action": "fill_missing", "column": "price", "method": "median"}
        ]
    }

    preview = preview_cleaning_plan(sample_df, plan)

    assert preview["requires_approval"] is True
    assert preview["preview_data"]["price"].isna().sum() == 0
    assert sample_df["price"].isna().sum() == 1

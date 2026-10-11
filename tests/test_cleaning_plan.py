
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
    assert preview["after_report"]["score"] > preview["before_report"]["score"]


@pytest.mark.parametrize("action", [None, [], {}, 42, True, "execute_python"])
def test_malformed_action_is_rejected_as_value_error(sample_df, action):
    with pytest.raises(ValueError):
        validate_cleaning_plan({"operations": [{"action": action}]}, sample_df)


@pytest.mark.parametrize("plan", [
    [], {"operations": {}}, {"operations": [None]},
    {"operations": [{"action": "trim_whitespace", "column": []}]},
    {"operations": [{"action": "trim_whitespace", "column": "price"}]},
    {"operations": [{"action": "fill_missing", "column": "price", "method": []}]},
    {"operations": [{"action": "remove_duplicates", "code": "print(1)"}]},
    {"operations": [{"action": "remove_duplicates"}] * 21},
    {"operations": [], "code": "print(1)"},
])
def test_malformed_plans_are_rejected(sample_df, plan):
    with pytest.raises(ValueError):
        validate_cleaning_plan(plan, sample_df)


@pytest.mark.parametrize("approved", [False, None, 1, "yes", "true"])
def test_trimming_requires_explicit_boolean_approval(sample_df, approved):
    plan = {"operations": [{"action": "trim_whitespace", "column": "product"}]}
    with pytest.raises(PermissionError):
        apply_cleaning_plan(sample_df, plan, approved=approved)
    assert sample_df.loc[0, "product"] == " Laptop "
    assert validate_cleaning_plan(plan, sample_df)[0]["requires_approval"] is True


def test_fractional_median_promotes_nullable_integer_copy():
    frame = pd.DataFrame({"amount": pd.Series([1, 2, pd.NA], dtype="Int64")})
    original = frame.copy(deep=True)
    plan = {"operations": [{"action": "fill_missing", "column": "amount", "method": "median"}]}
    result, _ = apply_cleaning_plan(frame, plan, approved=True)
    assert result["amount"].tolist() == [1.0, 2.0, 1.5]
    assert pd.api.types.is_float_dtype(result["amount"])
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("values,dtype", [
    ([pd.NA, pd.NA], "Float64"), ([True, False], "boolean"),
    ([1 + 2j, 3 + 4j], "complex128"), ([float("inf"), float("inf")], "float64"),
])
def test_invalid_median_columns_are_rejected(values, dtype):
    frame = pd.DataFrame({"value": pd.Series(values, dtype=dtype)})
    with pytest.raises(ValueError):
        validate_cleaning_plan({"operations": [{"action": "fill_missing", "column": "value", "method": "median"}]}, frame)


def test_preview_does_not_authorize_application(sample_df):
    plan = {"operations": [{"action": "trim_whitespace", "column": "product"}]}
    preview = preview_cleaning_plan(sample_df, plan)
    assert preview["requires_approval"] is True
    assert preview["preview_data"].loc[0, "product"] == "Laptop"
    assert sample_df.loc[0, "product"] == " Laptop "
    with pytest.raises(PermissionError):
        apply_cleaning_plan(sample_df, plan)


def test_duplicate_columns_are_rejected():
    frame = pd.DataFrame([[1, 2]], columns=["value", "value"])
    with pytest.raises(ValueError):
        validate_cleaning_plan({"operations": []}, frame)

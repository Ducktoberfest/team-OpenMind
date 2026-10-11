
import pandas as pd
import pytest

from data_analyzer import (
    summary_stats,
    validate_op,
    run_operation,
    run_operations,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Product": ["A", "B", "A", "B"],
        "Region": ["Baku", "Ganja", "Baku", "Baku"],
        "Revenue": [100, 200, 300, 400],
        "Quantity": [1, 2, 3, 4],
        "Order_Date": [
            "2024-01-10",
            "2024-01-20",
            "2024-02-10",
            "2024-02-20",
        ],
    })


def test_summary_stats(sample_df):
    report = summary_stats(sample_df)

    assert report["rows"] == 4
    assert report["cols"] == 5
    assert report["numeric"]["Revenue"]["sum"] == 1000


def test_sum(sample_df):
    result = run_operation(
        sample_df,
        {
            "op": "sum",
            "metric": "Revenue",
        },
    )

    assert result["value"] == 1000


def test_mean(sample_df):
    result = run_operation(
        sample_df,
        {
            "op": "mean",
            "metric": "Revenue",
        },
    )

    assert result["value"] == 250


def test_count(sample_df):
    result = run_operation(
        sample_df,
        {"op": "count"},
    )

    assert result["value"] == 4


def test_group_sum(sample_df):
    result = run_operation(
        sample_df,
        {
            "op": "group_sum",
            "metric": "Revenue",
            "group_by": "Product",
        },
    )

    assert result["table"][0]["Revenue"] == 600


def test_group_count(sample_df):
    result = run_operation(
        sample_df,
        {
            "op": "group_count",
            "group_by": "Product",
        },
    )

    assert len(result["table"]) == 2


def test_filter(sample_df):
    result = run_operation(
        sample_df,
        {
            "op": "sum",
            "metric": "Revenue",
            "filters": [
                {
                    "column": "Region",
                    "operator": "==",
                    "value": "Baku",
                }
            ],
        },
    )

    assert result["value"] == 800


def test_contains_literal(sample_df):
    result = run_operation(
        sample_df,
        {
            "op": "count",
            "filters": [
                {
                    "column": "Product",
                    "operator": "contains",
                    "value": ".",
                }
            ],
        },
    )

    assert result["value"] == 0


def test_trend(sample_df):
    result = run_operation(
        sample_df,
        {
            "op": "trend",
            "metric": "Revenue",
            "date_col": "Order_Date",
            "freq": "M",
        },
    )

    assert len(result["table"]) == 2
    assert result["table"][0]["Revenue"] == 300
    assert result["table"][1]["Revenue"] == 700


def test_unknown_operation(sample_df):
    with pytest.raises(ValueError):
        run_operation(
            sample_df,
            {"op": "execute_python"},
        )


def test_invalid_metric(sample_df):
    with pytest.raises(ValueError):
        run_operation(
            sample_df,
            {
                "op": "sum",
                "metric": "Unknown",
            },
        )


def test_invalid_top_n(sample_df):
    with pytest.raises(ValueError):
        run_operation(
            sample_df,
            {
                "op": "group_sum",
                "metric": "Revenue",
                "group_by": "Product",
                "top_n": -1,
            },
        )


def test_invalid_filter(sample_df):
    with pytest.raises(ValueError):
        run_operation(
            sample_df,
            {
                "op": "count",
                "filters": [
                    {
                        "column": "Region",
                        "operator": "DROP",
                        "value": "Baku",
                    }
                ],
            },
        )


def test_original_unchanged(sample_df):
    original = sample_df.copy(deep=True)

    run_operation(
        sample_df,
        {
            "op": "group_sum",
            "metric": "Revenue",
            "group_by": "Product",
        },
    )

    pd.testing.assert_frame_equal(
        sample_df,
        original,
    )


def test_multiple_operations(sample_df):
    results = run_operations(
        sample_df,
        [
            {"op": "count"},
            {
                "op": "sum",
                "metric": "Revenue",
            },
        ],
    )

    assert len(results) == 2
    assert results[0]["value"] == 4
    assert results[1]["value"] == 1000

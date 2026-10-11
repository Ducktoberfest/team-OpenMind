
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


@pytest.mark.parametrize("operation", [
    {"op": []},
    {"op": {"execute": "python"}},
    {"op": "sum", "metric": []},
    {"op": "count", "filters": [{"column": [], "operator": "==", "value": 1}]},
    {"op": "count", "filters": [{"column": "Product", "operator": [], "value": 1}]},
    {"op": "count", "filters": [{"column": "Product", "operator": "==", "value": {}}]},
    {"op": "count", "filters": [{"column": "Revenue", "operator": "==", "value": float("inf")}]},
    {"op": "count", "code": "__import__('os').system('echo unsafe')"},
])
def test_malformed_ai_operations_raise_value_error(sample_df, operation):
    original = sample_df.copy(deep=True)
    with pytest.raises(ValueError):
        run_operation(sample_df, operation)
    pd.testing.assert_frame_equal(sample_df, original)


@pytest.mark.parametrize("columns", [[[]], [1], ["Revenue", "Revenue"]])
def test_invalid_columns_raise_value_error(columns):
    with pytest.raises(ValueError):
        validate_op({"op": "count"}, columns)


@pytest.mark.parametrize("kind, expected", [
    ("group_sum", [{"Revenue": 10, "Revenue_sum": 20}, {"Revenue": 20, "Revenue_sum": 20}]),
    ("group_mean", [{"Revenue": 20, "Revenue_mean": 20}, {"Revenue": 10, "Revenue_mean": 10}]),
])
def test_group_metric_same_as_group_column(kind, expected):
    frame = pd.DataFrame({"Revenue": [20, 10, 10]})
    result = run_operation(frame, {"op": kind, "metric": "Revenue", "group_by": "Revenue"})
    assert result["table"] == expected
    assert result["value_column"] != result["group_by"]


def test_group_count_column_named_count():
    result = run_operation(
        pd.DataFrame({"count": ["A", "A", "B"]}),
        {"op": "group_count", "group_by": "count"},
    )
    assert result["table"] == [
        {"count": "A", "count_count": 2},
        {"count": "B", "count_count": 1},
    ]


def test_group_sum_does_not_turn_all_missing_values_into_zero():
    result = run_operation(
        pd.DataFrame({"Product": ["A", "A"], "Revenue": [None, None]}, dtype=object).astype({"Revenue": "float64"}),
        {"op": "group_sum", "metric": "Revenue", "group_by": "Product"},
    )
    assert result["table"] == [{"Product": "A", "Revenue": None}]


def test_trend_excludes_ambiguous_and_invalid_dates_without_mutating_data():
    frame = pd.DataFrame({
        "Date": ["2025-01-02", "01/02/2025", "03.04.2025", "13/02/2025", "02/13/2025", "Feb 14, 2025", "2025-02-30", None],
        "Revenue": [10, 100, 1000, 20, 30, 40, 500, 600],
    })
    original = frame.copy(deep=True)
    result = run_operation(frame, {"op": "trend", "metric": "Revenue", "date_col": "Date"})
    assert result["rows_used"] == 4
    assert result["invalid_dates"] == 4
    assert result["table"] == [
        {"period": "2025-01-01", "Revenue": 10},
        {"period": "2025-02-01", "Revenue": 90},
    ]
    pd.testing.assert_frame_equal(frame, original)


def test_datetime_filter_rejects_ambiguous_day_month():
    frame = pd.DataFrame({"Date": pd.to_datetime(["2025-01-02", "2025-02-01"])})
    with pytest.raises(ValueError):
        run_operation(frame, {"op": "count", "filters": [
            {"column": "Date", "operator": "==", "value": "01/02/2025"}
        ]})
    result = run_operation(frame, {"op": "count", "filters": [
        {"column": "Date", "operator": "==", "value": "2025-01-02"}
    ]})
    assert result["value"] == 1


def test_trend_metric_named_period_keeps_date_and_aggregate():
    result = run_operation(
        pd.DataFrame({"Date": ["2025-01-02"], "period": [10]}),
        {"op": "trend", "metric": "period", "date_col": "Date"},
    )
    assert result["value_column"] == "period_sum"
    assert result["table"] == [{"period": "2025-01-01", "period_sum": 10}]


def test_datetime_filter_preserves_explicit_timezone_offset():
    frame = pd.DataFrame({"Date": pd.to_datetime(["2025-01-02T10:00:00Z"]).tz_convert("Europe/Berlin")})
    result = run_operation(frame, {"op": "count", "filters": [
        {"column": "Date", "operator": "==", "value": "2025-01-02T12:00:00+02:00"}
    ]})
    assert result["value"] == 1

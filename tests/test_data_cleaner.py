import io

import pandas as pd
import pytest

from data_cleaner import apply_safe_fixes, load_file, quality_report


def test_upload_preserves_identifiers_zeroes_and_ambiguous_numeric_separators(tmp_path):
    path = tmp_path / "upload.csv"
    path.write_text('customer_id,postal_code,reference,amount,formatted\n00123,00100,0005,10,"1,234"\n00456,00200,0006,20,"1,25"\n', encoding="utf-8")
    frame = load_file(path)
    assert frame["customer_id"].tolist() == ["00123", "00456"]
    assert frame["postal_code"].tolist() == ["00100", "00200"]
    assert frame["reference"].tolist() == ["0005", "0006"]
    assert frame["formatted"].tolist() == ["1,234", "1,25"]
    assert pd.api.types.is_numeric_dtype(frame["amount"])
    assert frame["amount"].tolist() == [10, 20]


@pytest.mark.parametrize("separator", [",", ";", "\t", "|"])
def test_utf8_bom_and_common_csv_separators(separator):
    upload = io.BytesIO(("\ufeffname" + separator + "amount\nLaptop" + separator + "12.5\n").encode("utf-8"))
    frame = load_file(upload)
    assert frame.columns.tolist() == ["name", "amount"]
    assert frame.loc[0, "amount"] == 12.5


@pytest.mark.parametrize("contents", ["", "  \n", "a,b\n", "a,b\n1,2,3\n", "a,b\n1\n", "a,a\n1,2\n", 'a,b\n"unterminated,2\n'])
def test_invalid_or_empty_csv_is_rejected(contents):
    with pytest.raises(ValueError):
        load_file(io.StringIO(contents))


def test_non_utf8_csv_reports_encoding_error():
    with pytest.raises(ValueError, match="UTF-8"):
        load_file(io.BytesIO(b"name\n\xff\n"))


def test_literal_na_labels_are_preserved_and_blank_cells_detected():
    frame = load_file(io.StringIO("region,amount\nNA,10\nNULL,20\nn/a,30\n\"   \",40\n,50\n"))
    assert frame["region"].tolist()[:3] == ["NA", "NULL", "n/a"]
    missing = [issue for issue in quality_report(frame)["issues"] if issue["type"] == "missing"]
    assert missing[0]["column"] == "region"
    assert missing[0]["count"] == 2


def test_upload_numeric_inference_preserves_large_integer_precision():
    frame = load_file(io.StringIO("amount,name\n9007199254740993,one\n,two\n"))
    assert frame.loc[0, "amount"] == 9007199254740993
    assert frame.loc[1, "amount"] is pd.NA


def test_safe_fixes_preserve_uncertain_data_and_original():
    frame = pd.DataFrame({
        "Date": ["03/04/2025", "not-a-date", "31/12/2025", "2025-02-30", None],
        "customer_id": ["001", "002", "003", "004", "005"],
        "Product": [" laptop ", "LAPTOP", "Laptop", " phone ", "Phone"],
        "amount": ["1,234", "1,25", "123.45", None, "3"],
    })
    original = frame.copy(deep=True)
    result, log = apply_safe_fixes(frame, approved=True)
    assert result["Date"].tolist()[:4] == ["03/04/2025", "not-a-date", "2025-12-31", "2025-02-30"]
    assert pd.isna(result.loc[4, "Date"])
    assert result["customer_id"].tolist() == original["customer_id"].tolist()
    assert result["Product"].tolist() == ["laptop", "LAPTOP", "Laptop", "phone", "Phone"]
    pd.testing.assert_series_equal(result["amount"], original["amount"])
    pd.testing.assert_frame_equal(frame, original)
    assert log


def test_safe_fixes_require_approval():
    with pytest.raises(PermissionError):
        apply_safe_fixes(pd.DataFrame({"name": [" a "]}))


def test_safe_fixes_preserve_timestamps_and_missing_numeric_values():
    frame = pd.DataFrame({"Date": ["2025-01-02T12:30:00+04:00", None], "amount": ["1", None]})
    result, _ = apply_safe_fixes(frame, approved=True)
    assert result.loc[0, "Date"] == "2025-01-02T12:30:00+04:00"
    assert pd.isna(result.loc[1, "Date"])
    assert result.loc[0, "amount"] == 1
    assert pd.isna(result.loc[1, "amount"])


def test_quality_report_separates_safe_formatting_from_uncertain_values():
    frame = pd.DataFrame({
        "Date": ["03/04/2025", "31/12/2025", "2025-02-30", "not-a-date"],
        "amount": ["1,234", "1,25", "30", "40"],
        "Product": ["laptop", "LAPTOP", "a", "a"],
        "customer_id": ["001", "002", "003", "004"],
    })
    issues = quality_report(frame)["issues"]
    by_type = {issue["type"]: issue for issue in issues}
    assert by_type["date_ambiguous"]["count"] == 1
    assert by_type["date_invalid"]["count"] == 2
    assert by_type["date_format"]["count"] == 1
    assert by_type["date_format"]["safe_fix"] is True
    assert by_type["date_ambiguous"]["safe_fix"] is False
    assert by_type["numeric_format"]["safe_fix"] is False
    assert by_type["spelling"]["safe_fix"] is False
    assert not any(issue["column"] == "customer_id" and issue["type"] == "dtype" for issue in issues)


def test_safe_numeric_text_conversion_handles_nullable_values_without_coercion():
    frame = pd.DataFrame({"amount": ["10", None, "20"], "reference": ["001", "002", "003"]})
    result, _ = apply_safe_fixes(frame, approved=True)
    assert pd.api.types.is_integer_dtype(result["amount"])
    assert result["amount"].isna().sum() == 1
    assert result["reference"].tolist() == frame["reference"].tolist()

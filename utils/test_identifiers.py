import pytest

from utils.identifiers import canonicalize_identifier


@pytest.mark.parametrize(
    "value, expected",
    [
        ("001", "1"),
        ("000123", "123"),
        ("-0005", "-5"),
        ("+0008", "8"),
    ],
)
def test_canonicalize_identifier_normalizes_numeric_strings(value, expected):
    assert canonicalize_identifier(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ("abc", "abc"),
        ("001abc", "001abc"),
        ("0.010", "0.01"),
    ],
)
def test_canonicalize_identifier_preserves_non_integer_strings(value, expected):
    assert canonicalize_identifier(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ("images/PT_0001.png", "PT_0001"),
        ("sensor_sequences/0005.csv", "5"),
        ("PT_0002.jpg", "PT_0002"),
        ("reports/followup.json", "followup"),
    ],
)
def test_canonicalize_identifier_strips_asset_paths_and_extensions(value, expected):
    assert canonicalize_identifier(value) == expected

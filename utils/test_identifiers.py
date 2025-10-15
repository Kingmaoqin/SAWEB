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

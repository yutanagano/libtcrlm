import pytest

from libtcrlm import schema


@pytest.mark.parametrize(
    argnames=("anchor", "comparison", "expected"),
    argvalues=(
        (
            schema.make_pmhc_from_components("AAA", "HLA-A*01:01", "B2M"),
            schema.make_pmhc_from_components("AAA", "HLA-A*01:01", "B2M"),
            True,
        ),
        (
            schema.make_pmhc_from_components("AAA", "HLA-A*01:01", "B2M"),
            schema.make_pmhc_from_components("CCC", "HLA-DRA*01:01", "HLA-DRB1*01:01"),
            False,
        ),
        (
            schema.make_pmhc_from_components("AAA", "HLA-A*01:01", "B2M"),
            schema.make_pmhc_from_components("AAA", None, None),
            True,
        ),
        (
            schema.make_pmhc_from_components("CCC", "HLA-DRA*01:01", "HLA-DRB1*01:01"),
            schema.make_pmhc_from_components("CCC", "HLA-DRA", "HLA-DRB1"),
            True,
        ),
        (
            schema.make_pmhc_from_components(None, "HLA-DRA*01:01", "HLA-DRB1*01:01"),
            schema.make_pmhc_from_components(None, "HLA-DRA*01:01", "HLA-DRB1*01:01"),
            False,
        ),
        (
            schema.make_pmhc_from_components("AAA", None, None),
            schema.make_pmhc_from_components("AAA", None, None),
            True,
        ),
    ),
)
def test_equality(anchor, comparison, expected):
    result = anchor == comparison
    assert result == expected


def test_repr():
    pmhc = schema.make_pmhc_from_components("AAA", "HLA-A*01:01", "B2M")
    assert repr(pmhc) == "AAA/HLA-A*01:01/B2M"

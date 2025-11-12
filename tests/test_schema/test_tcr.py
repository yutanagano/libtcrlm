import pytest
import libtcrlm
from libtcrlm import schema
from libtcrlm.schema import exception


def test_cdr1a_sequence(mock_tcr):
    result = mock_tcr.cdr1a_sequence
    expected = "TSGFYG"

    assert result == expected


def test_cdr1b_sequence(mock_tcr):
    result = mock_tcr.cdr1b_sequence
    expected = "SNHLY"

    assert result == expected


def test_cdr2a_sequence(mock_tcr):
    result = mock_tcr.cdr2a_sequence
    expected = "NALDGL"

    assert result == expected


def test_cdr2b_sequence(mock_tcr):
    result = mock_tcr.cdr2b_sequence
    expected = "FYNNEI"

    assert result == expected


def test_junction_a_sequence(mock_tcr):
    result = mock_tcr.junction_a_sequence
    expected = "CATQYF"

    assert result == expected


def test_junction_b_sequence(mock_tcr):
    result = mock_tcr.junction_b_sequence
    expected = "CASQYF"

    assert result == expected


def test_repr(mock_tcr):
    result = repr(mock_tcr)
    expected = "Tra(TRAV1-1*01/CATQYF)/Trb(TRBV2*01/CASQYF)"

    assert result == expected


@pytest.mark.parametrize(
    argnames=("anchor", "comparison", "expected"),
    argvalues=(
        (
            schema.make_tcr_from_components(
                "TRAV1-1*01", "CATQYF", "TRBV2*01", "CASQYF"
            ),
            schema.make_tcr_from_components(
                "TRAV1-1*01", "CATQYF", "TRBV2*01", "CASQYF"
            ),
            True,
        ),
        (
            schema.make_tcr_from_components(
                "TRAV1-1*01", "CATQYF", "TRBV2*01", "CASQYF"
            ),
            schema.make_tcr_from_components(
                "TRAV5*01", "CASSRPLWYF", "TRBV3-1*01", "CASKLAQF"
            ),
            False,
        ),
        (
            schema.make_tcr_from_components(
                "TRAV1-1*01", "CATQYF", "TRBV2*01", "CASQYF"
            ),
            schema.make_tcr_from_components("TRAV1-1*01", "CATQYF", None, None),
            False,
        ),
    ),
)
def test_equality(anchor, comparison, expected):
    result = anchor == comparison
    assert result == expected


def test_allele_imputation():
    tcr = schema.make_tcr_from_components("TRAV1-1", "CATQYF", "TRBV12-2", "CASQYF")
    assert tcr._trav.allele_num == 1
    assert tcr._trbv.allele_num == 2


class TestSetup:
    def test_homosapiens_setup(self):
        tcr = schema.make_tcr_from_components(
            "TRAV1-1*01", "CATQYF", "TRBV2*01", "CASQYF"
        )
        tcr.cdr1a_sequence == "TSGFYG"
        tcr.cdr2a_sequence == "NALDGL"
        tcr.junction_a_sequence == "CATQYF"
        tcr.cdr1b_sequence == "SNHLY"
        tcr.cdr2b_sequence == "FYNNEI"
        tcr.junction_b_sequence == "CASQYF"

        with pytest.raises(exception.BadV):
            tcr = schema.make_tcr_from_components(
                "TRAV1*01", "CATQYF", "TRBV1*01", "CASQYF"
            )

    def test_musmusculus_setup(self, trigger_musmusculus_setup):
        tcr = schema.make_tcr_from_components(
            "TRAV1*01", "CATQYF", "TRBV1*01", "CASQYF"
        )
        tcr.cdr1a_sequence == "TSGFYG"
        tcr.cdr2a_sequence == "NALDGL"
        tcr.junction_a_sequence == "CATQYF"
        tcr.cdr1b_sequence == "SNHLY"
        tcr.cdr2b_sequence == "FYNNEI"
        tcr.junction_b_sequence == "CASQYF"

        with pytest.raises(exception.BadV):
            tcr = schema.make_tcr_from_components(
                "TRAV1-1*01", "CATQYF", "TRBV2*01", "CASQYF"
            )


@pytest.fixture
def trigger_musmusculus_setup():
    libtcrlm.setup("musmusculus")
    yield
    libtcrlm.setup("homosapiens")

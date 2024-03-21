import pytest
from libtcrlm.dataset import TcrDataset
from libtcrlm import schema


def test_len(tcr_dataset):
    assert len(tcr_dataset) == 3


def test_getitem(tcr_dataset):
    first_tcr_pmhc_pair = tcr_dataset[0]
    expected_tcr = schema.make_tcr_from_components(
        "TRAV1-1*01", "CAVKASGSRLT", "TRBV2*01", "CASSDRAQPQHF"
    )
    expected_pmhc = schema.make_pmhc_from_components("CLAMP", "HLA-A*01", "B2M")

    assert first_tcr_pmhc_pair.tcr == expected_tcr
    assert first_tcr_pmhc_pair.pmhc == expected_pmhc


@pytest.fixture
def tcr_dataset(mock_data_df):
    return TcrDataset(mock_data_df)

from libtcrlm.schema import df_manipulation
import pandas as pd
import pytest


def test_bad_v():
    df = pd.read_csv("tests/resources/bad_trav.csv")
    with pytest.raises(ValueError, match="Bad TRAV symbol at index 2"):
        df_manipulation.generate_tcr_series(df)


def test_bad_junction():
    df = pd.read_csv("tests/resources/bad_cdr3b.csv")
    with pytest.raises(ValueError, match="Bad CDR3B sequence at index 2"):
        df_manipulation.generate_tcr_series(df)

from libtcrlm import schema
from libtcrlm.schema import exception
from libtcrlm.schema import Tcr, TcrPmhcPair
import pandas as pd
from pandas import DataFrame, Series


def generate_tcr_pmhc_series(data: DataFrame) -> Series:
    tcr_pmhc_series = data.apply(_generate_tcr_pmhc_pair_from_row, axis="columns")
    return tcr_pmhc_series


def generate_tcr_series(data: DataFrame) -> Series:
    tcr_series = data.apply(_generate_tcr_from_row, axis="columns")
    return tcr_series


def _generate_tcr_pmhc_pair_from_row(row: Series) -> TcrPmhcPair:
    tcr = _generate_tcr_from_row(row)

    epitope = _get_value_if_not_na_else_none(row.Epitope)
    mhc_a = _get_value_if_not_na_else_none(row.MHCA)
    mhc_b = _get_value_if_not_na_else_none(row.MHCB)

    try:
        pmhc = schema.make_pmhc_from_components(
            epitope_sequence=epitope, mhc_a_symbol=mhc_a, mhc_b_symbol=mhc_b
        )
    except Exception:
        raise ValueError(
            f"Bad pMHC data at index {row.name}. "
            "Most likely, there is a non-standard MH symbol, or an invalid peptide sequence. "
            "You can use tidytcells with specific flags to filter out non-valid data (see https://tidytcells.readthedocs.io).\n\n"
            f"Epitope: {row.Epitope}\n"
            f"MHCA:    {row.MHCA}\n"
            f"MHCB:    {row.MHCB}"
        )

    return TcrPmhcPair(tcr, pmhc)


def _generate_tcr_from_row(row: Series) -> Tcr:
    trav = _get_value_if_not_na_else_none(row.TRAV)
    trbv = _get_value_if_not_na_else_none(row.TRBV)
    junction_a = _get_value_if_not_na_else_none(row.CDR3A)
    junction_b = _get_value_if_not_na_else_none(row.CDR3B)

    try:
        return schema.make_tcr_from_components(
            trav_symbol=trav,
            junction_a_sequence=junction_a,
            trbv_symbol=trbv,
            junction_b_sequence=junction_b,
        )
    except exception.BadV as e:
        colname = f"TR{e.chain}V"
        bad_symbol = row[colname]
        raise ValueError(
            f"Bad {colname} symbol at index {row.name}: {bad_symbol}. "
            "Have you ensured that all TR V symbols are standardised and functional? "
            "You can use tidytcells with specific flags to filter out non-valid data (see https://tidytcells.readthedocs.io)."
        )
    except exception.BadJunction as e:
        colname = f"CDR3{e.chain}"
        bad_junction = row[colname]
        raise ValueError(
            f"Bad {colname} sequence at index {row.name}: {bad_junction}. "
            "Have you ensured that all CDR3 sequences are standardised and valid? "
            "You can use tidytcells with specific flags to filter out non-valid data (see https://tidytcells.readthedocs.io)."
        )


def _get_value_if_not_na_else_none(value) -> any:
    if pd.isna(value):
        return None
    return value

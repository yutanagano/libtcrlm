from libtcrlm import schema
from libtcrlm.schema import TcrPmhcPair
from pandas import DataFrame
from torch.utils.data import Dataset


class TcrDataset(Dataset):
    def __init__(self, data: DataFrame):
        super().__init__()
        self._tcr_pmhc_series = schema.generate_tcr_pmhc_series(data)

    def __len__(self) -> int:
        return len(self._tcr_pmhc_series)

    def __getitem__(self, index: int) -> TcrPmhcPair:
        return self._tcr_pmhc_series.iloc[index]


class EpitopeBalancedTcrDataset(Dataset):
    def __init__(self, data: DataFrame, num_samples_per_pmhc_per_epoch: int):
        super().__init__()
        tcr_pmhc_series = schema.generate_tcr_pmhc_series(data)
        tcr_pmhc_df = tcr_pmhc_series.to_frame().apply(
            lambda row: (row.item().tcr, row.item().pmhc),
            axis=1,
            result_type="expand"
        )
        tcr_pmhc_df.columns = ["tcr", "pmhc"]

        self._tcr_pmhc_df = tcr_pmhc_df
        self._pmhcs = tcr_pmhc_df.pmhc.unique()
        self._num_samples_per_specificity_per_epoch = num_samples_per_pmhc_per_epoch

    def __len__(self) -> int:
        return len(self._pmhcs) * self._num_samples_per_specificity_per_epoch

    def __getitem__(self, index: int) -> TcrPmhcPair:
        pmhc = self._pmhcs[index % len(self._pmhcs)]
        tcrs_against_pmhc = self._tcr_pmhc_df.tcr[self._tcr_pmhc_df.pmhc == pmhc]
        tcr = tcrs_against_pmhc.sample().item()

        return TcrPmhcPair(tcr, pmhc)
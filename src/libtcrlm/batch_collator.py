from abc import ABC, abstractmethod
from libtcrlm.tokeniser import Tokeniser
from libtcrlm.tokeniser.token_indices import DefaultTokenIndex
from libtcrlm.schema import Tcr, TcrPmhcPair
from torch import LongTensor
from torch.nn import utils
from typing import Iterable, Tuple


class BatchCollator(ABC):
    def __init__(self, tokeniser: Tokeniser) -> None:
        self._tokeniser = tokeniser

    def tokenise(self, tcr: Tcr) -> LongTensor:
        return self._tokeniser.tokenise(tcr)

    @abstractmethod
    def collate_fn(self, tcr_pmhc_pairs: Iterable[TcrPmhcPair]) -> Tuple[LongTensor]:
        pass


class DefaultBatchCollator(BatchCollator):
    def collate_fn(self, tcr_pmhc_pairs: Iterable[TcrPmhcPair]) -> Tuple[LongTensor]:
        batch = [
            self.tokenise(tcr_pmhc_pair.tcr)
            for tcr_pmhc_pair in tcr_pmhc_pairs
        ]
        padded_batch = utils.rnn.pad_sequence(
            sequences=batch, batch_first=True, padding_value=DefaultTokenIndex.NULL
        )
        return (padded_batch,)
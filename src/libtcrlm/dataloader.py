from libtcrlm.batch_collator import BatchCollator
from libtcrlm.schema import TcrPmhcPair
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Iterable, Tuple


class TcrDataLoader(DataLoader):
    def __init__(
        self, *args, batch_collator: BatchCollator, device: torch.device, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._batch_collator = batch_collator
        self._device = device
        self.collate_fn = self._custom_collate_fn

    def _custom_collate_fn(
        self, tcr_pmhc_pairs: Iterable[TcrPmhcPair]
    ) -> Tuple[Tensor]:
        batch = self._batch_collator.collate_fn(tcr_pmhc_pairs)
        batch = [tensor.to(self._device) for tensor in batch]
        return tuple(batch)

    def set_epoch(self, epoch: int) -> None:
        self.sampler.set_epoch(epoch)

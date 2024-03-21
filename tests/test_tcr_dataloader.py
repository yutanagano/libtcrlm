import pytest
import torch
from torch import Tensor

from libtcrlm.tokeniser.cdr_tokeniser import CdrTokeniser
from libtcrlm.dataset import TcrDataset
from libtcrlm.dataloader import TcrDataLoader
from libtcrlm.batch_collator import DefaultBatchCollator


BATCH_DIMENSIONALITY = 3
BATCH_SIZE = 3
MAX_TOKENISED_TCR_LENGTH = 48
TOKEN_NUM_DIMS = 4


def test_iter(tcr_dataset, default_batch_collator):
    dataloader = TcrDataLoader(
        tcr_dataset,
        batch_collator=default_batch_collator,
        device=torch.device("cpu"),
        batch_size=BATCH_SIZE,
    )

    for (batch,) in dataloader:
        assert type(batch) == Tensor
        assert batch.dim() == BATCH_DIMENSIONALITY
        assert batch.size(0) == BATCH_SIZE
        assert batch.size(1) == MAX_TOKENISED_TCR_LENGTH
        assert batch.size(2) == TOKEN_NUM_DIMS


def test_set_epoch(tcr_dataset, default_batch_collator):
    mock_sampler = MockSampler()
    dataloader = TcrDataLoader(
        tcr_dataset,
        batch_collator=default_batch_collator,
        device=torch.device("cpu"),
        sampler=mock_sampler,
    )
    EPOCH = 420

    dataloader.set_epoch(EPOCH)

    assert mock_sampler.epoch_set_as == EPOCH


class MockSampler:
    def __init__(self) -> None:
        self.epoch_set_as = None

    def set_epoch(self, epoch: int):
        self.epoch_set_as = epoch


@pytest.fixture
def tcr_dataset(mock_data_df):
    return TcrDataset(mock_data_df)


@pytest.fixture
def default_batch_collator():
    tokeniser = CdrTokeniser()
    return DefaultBatchCollator(tokeniser=tokeniser)

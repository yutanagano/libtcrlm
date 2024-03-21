import pytest
import torch
from torch import Tensor

from libtcrlm.bert import Bert


BATCH_SIZE = 5
TOKENISED_TCR_LENGTH = 5
TOKEN_DIMENSIONALITY = 4

AMINO_ACID_VOCABULARY_SIZE = 20


@pytest.fixture
def mock_tokenised_tcrs():
    return torch.ones(
        (BATCH_SIZE, TOKENISED_TCR_LENGTH, TOKEN_DIMENSIONALITY), dtype=torch.long
    )


def test_get_vector_representations_of(
    toy_bert: Bert, toy_bert_d_model, mock_tokenised_tcrs
):
    result = toy_bert.get_vector_representations_of(mock_tokenised_tcrs)

    assert type(result) == Tensor
    assert result.dim() == 2
    assert result.size(0) == BATCH_SIZE
    assert result.size(1) == toy_bert_d_model


def test_get_mlm_token_predictions_for(toy_bert: Bert, mock_tokenised_tcrs):
    result = toy_bert.get_mlm_token_predictions_for(mock_tokenised_tcrs)

    assert type(result) == Tensor
    assert result.dim() == 3
    assert result.size(0) == BATCH_SIZE
    assert result.size(1) == TOKENISED_TCR_LENGTH
    assert result.size(2) == AMINO_ACID_VOCABULARY_SIZE

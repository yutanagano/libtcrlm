import pytest
import torch
from torch import Tensor

from libtcrlm.self_attention_stack import SelfAttentionStackWithBuiltins
from libtcrlm.vector_representation_delegate import ClsVectorRepresentationDelegate


D_MODEL = 4
BATCH_SIZE = 3
TOKENISED_TCR_LENGTH = 5


@pytest.fixture
def mock_token_embeddings():
    return torch.ones((BATCH_SIZE, TOKENISED_TCR_LENGTH, D_MODEL))


@pytest.fixture
def mock_padding_mask():
    return torch.zeros((BATCH_SIZE, TOKENISED_TCR_LENGTH), dtype=torch.bool)


def test_get_vector_representations_of(mock_token_embeddings, mock_padding_mask):
    self_attention_stack = SelfAttentionStackWithBuiltins(
        num_layers=2, d_model=D_MODEL, nhead=2
    )
    vector_representation_delegate = ClsVectorRepresentationDelegate(
        self_attention_stack
    )

    result = vector_representation_delegate.get_vector_representations_of(
        mock_token_embeddings, mock_padding_mask
    )

    assert type(result) == Tensor
    assert result.dim() == 2
    assert result.size(0) == BATCH_SIZE
    assert result.size(1) == D_MODEL

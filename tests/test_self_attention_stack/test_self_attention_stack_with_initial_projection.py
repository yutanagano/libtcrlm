import pytest
import torch
from torch import Tensor

from libtcrlm.self_attention_stack import SelfAttentionStackWithInitialProjection


EMBEDDING_DIM = 10
D_MODEL = 4
BATCH_SIZE = 3
TOKENISED_TCR_LENGTH = 5


@pytest.fixture
def self_attention_stack_with_initial_projection():
    self_attention_stack_with_builtins = SelfAttentionStackWithInitialProjection(
        num_layers=3, embedding_dim=EMBEDDING_DIM, d_model=D_MODEL, nhead=2
    )
    return self_attention_stack_with_builtins


@pytest.fixture
def mock_token_embeddings():
    return torch.ones((BATCH_SIZE, TOKENISED_TCR_LENGTH, EMBEDDING_DIM))


@pytest.fixture
def mock_padding_mask():
    return torch.zeros((BATCH_SIZE, TOKENISED_TCR_LENGTH), dtype=torch.bool)


def test_forward(
    self_attention_stack_with_initial_projection: SelfAttentionStackWithInitialProjection,
    mock_token_embeddings,
    mock_padding_mask,
):
    result = self_attention_stack_with_initial_projection.forward(
        mock_token_embeddings, mock_padding_mask
    )

    assert type(result) == Tensor
    assert result.dim() == 3
    assert result.size(0) == BATCH_SIZE
    assert result.size(1) == TOKENISED_TCR_LENGTH
    assert result.size(2) == D_MODEL


def test_get_token_embeddings_at_penultimate_layer(
    self_attention_stack_with_initial_projection: SelfAttentionStackWithInitialProjection,
    mock_token_embeddings,
    mock_padding_mask,
):
    result = self_attention_stack_with_initial_projection.get_token_embeddings_at_penultimate_layer(
        mock_token_embeddings, mock_padding_mask
    )

    assert type(result) == Tensor
    assert result.dim() == 3
    assert result.size(0) == BATCH_SIZE
    assert result.size(1) == TOKENISED_TCR_LENGTH
    assert result.size(2) == D_MODEL

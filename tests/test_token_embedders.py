import torch
from torch import Tensor
from libtcrlm.schema import Tcr
from libtcrlm.tokeniser import *
from libtcrlm.token_embedder import *


def test_single_chain_cdr3_simple_embedder(mock_tcr):
    BATCH_DIMENSIONALITY = 3
    BATCH_SIZE = 2

    batch = get_mock_batch_of_tokenised_tcrs(mock_tcr, BetaCdr3Tokeniser())
    embedder = SingleChainCdr3SimpleEmbedder()
    batch_of_embedded_mock_tcrs = embedder.forward(batch)
    embedded_mock_tcr = batch_of_embedded_mock_tcrs[0]
    expected_embedded_mock_tcr = torch.tensor(
        [
            [
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.2,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0.4,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0.6,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0.8,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
            ],
        ]
    )

    assert type(batch_of_embedded_mock_tcrs) == Tensor

    assert batch_of_embedded_mock_tcrs.dim() == BATCH_DIMENSIONALITY
    assert batch_of_embedded_mock_tcrs.size(0) == BATCH_SIZE

    assert torch.equal(embedded_mock_tcr, expected_embedded_mock_tcr)


def test_cdr_simple_embedder(mock_tcr):
    BATCH_DIMENSIONALITY = 3
    BATCH_SIZE = 2

    batch = get_mock_batch_of_tokenised_tcrs(mock_tcr, CdrTokeniser())
    embedder = CdrSimpleEmbedder()
    batch_of_embedded_mock_tcrs = embedder.forward(batch)
    embedded_mock_tcr = batch_of_embedded_mock_tcrs[0]
    expected_embedded_mock_tcr = torch.tensor(
        [
            [
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0.2,
                1,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.4,
                1,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.6,
                1,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0.8,
                1,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.2,
                0,
                1,
                0,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.4,
                0,
                1,
                0,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.6,
                0,
                1,
                0,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.8,
                0,
                1,
                0,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.2,
                0,
                0,
                1,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0.4,
                0,
                0,
                1,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0.6,
                0,
                0,
                1,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0.8,
                0,
                0,
                1,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.25,
                0,
                0,
                0,
                1,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.5,
                0,
                0,
                0,
                1,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.75,
                0,
                0,
                0,
                1,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0.2,
                0,
                0,
                0,
                0,
                1,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.4,
                0,
                0,
                0,
                0,
                1,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.6,
                0,
                0,
                0,
                0,
                1,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.8,
                0,
                0,
                0,
                0,
                1,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
            ],
            [
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
            ],
            [
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.2,
                0,
                0,
                0,
                0,
                0,
                1,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0.4,
                0,
                0,
                0,
                0,
                0,
                1,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0.6,
                0,
                0,
                0,
                0,
                0,
                1,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0.8,
                0,
                0,
                0,
                0,
                0,
                1,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                1,
            ],
        ]
    )

    assert type(batch_of_embedded_mock_tcrs) == Tensor

    assert batch_of_embedded_mock_tcrs.dim() == BATCH_DIMENSIONALITY
    assert batch_of_embedded_mock_tcrs.size(0) == BATCH_SIZE

    assert torch.equal(embedded_mock_tcr, expected_embedded_mock_tcr)


def test_cdr_blosum_embedder(mock_tcr):
    BATCH_DIMENSIONALITY = 3
    BATCH_SIZE = 2
    TOKENISED_TCR_LENGTH = 36
    EMBEDDING_DIM = 20 + 1 + 6

    batch = get_mock_batch_of_tokenised_tcrs(mock_tcr, CdrTokeniser())
    embedder = CdrBlosumEmbedder()
    mock_batch_of_embedded_tcrs = embedder.forward(batch)

    assert type(mock_batch_of_embedded_tcrs) == Tensor

    assert mock_batch_of_embedded_tcrs.dim() == BATCH_DIMENSIONALITY
    assert mock_batch_of_embedded_tcrs.size(0) == BATCH_SIZE
    assert mock_batch_of_embedded_tcrs.size(1) == TOKENISED_TCR_LENGTH
    assert mock_batch_of_embedded_tcrs.size(2) == EMBEDDING_DIM


def test_single_chain_cdr_embedder(mock_tcr):
    BATCH_SIZE = 2
    TOKENISED_TCR_LENGTH = 18
    EMBEDDING_DIM = 10

    batch = get_mock_batch_of_tokenised_tcrs(mock_tcr, BetaCdrTokeniser())
    embedder = SingleChainCdrEmbedder(embedding_dim=EMBEDDING_DIM)
    mock_batch_of_embedded_tcrs = embedder.forward(batch)

    assert type(mock_batch_of_embedded_tcrs) == Tensor
    assert mock_batch_of_embedded_tcrs.dim() == 3
    assert mock_batch_of_embedded_tcrs.size(0) == BATCH_SIZE
    assert mock_batch_of_embedded_tcrs.size(1) == TOKENISED_TCR_LENGTH
    assert mock_batch_of_embedded_tcrs.size(2) == EMBEDDING_DIM


def test_single_chain_cdr_simple_embedder(mock_tcr):
    BATCH_DIMENSIONALITY = 3
    BATCH_SIZE = 2

    batch = get_mock_batch_of_tokenised_tcrs(mock_tcr, BetaCdrTokeniser())
    embedder = SingleChainCdrSimpleEmbedder()
    batch_of_embedded_mock_tcrs = embedder.forward(batch)
    embedded_mock_tcr = batch_of_embedded_mock_tcrs[0]
    expected_embedded_mock_tcr = torch.tensor(
        [
            [
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.25,
                1,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.5,
                1,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.75,
                1,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                0,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0.2,
                0,
                1,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.4,
                0,
                1,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.6,
                0,
                1,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.8,
                0,
                1,
                0,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
            ],
            [
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
            ],
            [
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.2,
                0,
                0,
                1,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0.4,
                0,
                0,
                1,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0.6,
                0,
                0,
                1,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0.8,
                0,
                0,
                1,
            ],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
            ],
        ]
    )

    assert type(batch_of_embedded_mock_tcrs) == Tensor

    assert batch_of_embedded_mock_tcrs.dim() == BATCH_DIMENSIONALITY
    assert batch_of_embedded_mock_tcrs.size(0) == BATCH_SIZE

    assert torch.equal(embedded_mock_tcr, expected_embedded_mock_tcr)


def get_mock_batch_of_tokenised_tcrs(tcr: Tcr, tokeniser: Tokeniser):
    mock_tokenised_tcr = tokeniser.tokenise(tcr)
    return torch.stack([mock_tokenised_tcr, mock_tokenised_tcr])

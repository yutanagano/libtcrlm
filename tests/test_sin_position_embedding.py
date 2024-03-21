import torch
from torch import testing
from libtcrlm.token_embedder.sin_position_embedding import SinPositionEmbedding


def test_embedding():
    embedding = SinPositionEmbedding(num_embeddings=10, embedding_dim=4)
    mock_batch_of_token_indices = torch.tensor([[1, 2], [1, 0]], dtype=torch.long)

    result = embedding.forward(mock_batch_of_token_indices)
    expected = torch.tensor(
        [
            [[0.0000, 1.0000, 0.0000, 1.0000], [0.8415, 0.5403, 0.1816, 0.9834]],
            [[0.0000, 1.0000, 0.0000, 1.0000], [0.0000, 0.0000, 0.0000, 0.0000]],
        ]
    )

    testing.assert_close(result, expected, rtol=0, atol=0.00005)

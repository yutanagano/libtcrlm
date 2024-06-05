import math
from libtcrlm.tokeniser.token_indices import (
    DefaultTokenIndex,
    AminoAcidTokenIndex,
    CdrCompartmentIndex,
    SingleChainCdrCompartmentIndex,
)
from libtcrlm.token_embedder.token_embedder import TokenEmbedder
from libtcrlm.token_embedder.simple_relative_position_embedding import (
    SimpleRelativePositionEmbedding,
)
from libtcrlm.token_embedder.sin_position_embedding import SinPositionEmbedding
from libtcrlm.token_embedder.one_hot_token_index_embedding import (
    OneHotTokenIndexEmbedding,
)
from libtcrlm.token_embedder.blosum_embedding import BlosumEmbedding
import torch
from torch import FloatTensor, LongTensor
from torch.nn import Embedding


MAX_PLAUSIBLE_CDR_LENGTH = 100


class CdrBlosumEmbedder(TokenEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self._blosum_embedding = BlosumEmbedding()
        self._position_embedding = SimpleRelativePositionEmbedding()
        self._compartment_embedding = OneHotTokenIndexEmbedding(CdrCompartmentIndex)

    def forward(self, tokenised_tcrs: LongTensor) -> FloatTensor:
        token_component = self._blosum_embedding.forward(tokenised_tcrs[:, :, 0])
        position_component = self._position_embedding.forward(tokenised_tcrs[:, :, 1:3])
        compartment_component = self._compartment_embedding.forward(
            tokenised_tcrs[:, :, 3]
        )
        all_components_stacked = torch.concatenate(
            [token_component, position_component, compartment_component], dim=-1
        )
        return all_components_stacked


class CdrSimpleEmbedder(TokenEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self._token_embedding = OneHotTokenIndexEmbedding(AminoAcidTokenIndex)
        self._position_embedding = SimpleRelativePositionEmbedding()
        self._compartment_embedding = OneHotTokenIndexEmbedding(CdrCompartmentIndex)

    def forward(self, tokenised_tcrs: LongTensor) -> FloatTensor:
        token_component = self._token_embedding.forward(tokenised_tcrs[:, :, 0])
        position_component = self._position_embedding.forward(tokenised_tcrs[:, :, 1:3])
        compartment_component = self._compartment_embedding.forward(
            tokenised_tcrs[:, :, 3]
        )
        all_components_stacked = torch.concatenate(
            [token_component, position_component, compartment_component], dim=-1
        )
        return all_components_stacked


class CdrEmbedder(TokenEmbedder):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()

        self._embedding_dim = embedding_dim

        self.token_embedding = Embedding(
            num_embeddings=len(AminoAcidTokenIndex),
            embedding_dim=embedding_dim,
            padding_idx=DefaultTokenIndex.NULL,
        )
        self.position_embedding = SinPositionEmbedding(
            num_embeddings=MAX_PLAUSIBLE_CDR_LENGTH, embedding_dim=embedding_dim
        )
        self.compartment_embedding = Embedding(
            num_embeddings=len(CdrCompartmentIndex),
            embedding_dim=embedding_dim,
            padding_idx=DefaultTokenIndex.NULL,
        )

    def forward(self, tokenised_tcrs: LongTensor) -> FloatTensor:
        token_component = self.token_embedding.forward(tokenised_tcrs[:, :, 0])
        position_component = self.position_embedding.forward(tokenised_tcrs[:, :, 1])
        compartment_component = self.compartment_embedding.forward(
            tokenised_tcrs[:, :, 3]
        )

        all_components_summed = (
            token_component + position_component + compartment_component
        )

        return all_components_summed * math.sqrt(self._embedding_dim)


class SingleChainCdrEmbedder(TokenEmbedder):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()

        self._embedding_dim = embedding_dim

        self.token_embedding = Embedding(
            num_embeddings=len(AminoAcidTokenIndex),
            embedding_dim=embedding_dim,
            padding_idx=DefaultTokenIndex.NULL,
        )
        self.position_embedding = SinPositionEmbedding(
            num_embeddings=MAX_PLAUSIBLE_CDR_LENGTH, embedding_dim=embedding_dim
        )
        self.compartment_embedding = Embedding(
            num_embeddings=len(SingleChainCdrCompartmentIndex),
            embedding_dim=embedding_dim,
            padding_idx=DefaultTokenIndex.NULL,
        )

    def forward(self, tokenised_tcrs: LongTensor) -> FloatTensor:
        token_component = self.token_embedding.forward(tokenised_tcrs[:, :, 0])
        position_component = self.position_embedding.forward(tokenised_tcrs[:, :, 1])
        compartment_component = self.compartment_embedding.forward(
            tokenised_tcrs[:, :, 3]
        )

        all_components_summed = (
            token_component + position_component + compartment_component
        )

        return all_components_summed * math.sqrt(self._embedding_dim)


class SingleChainCdrEmbedderWithRelativePositions(TokenEmbedder):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()

        self._embedding_dim = embedding_dim

        self._token_embedding = Embedding(
            num_embeddings=len(AminoAcidTokenIndex),
            embedding_dim=embedding_dim,
            padding_idx=DefaultTokenIndex.NULL,
        )
        self._position_embedding = SimpleRelativePositionEmbedding()
        self._compartment_embedding = OneHotTokenIndexEmbedding(
            SingleChainCdrCompartmentIndex
        )

    def forward(self, tokenised_tcrs: LongTensor) -> FloatTensor:
        token_component = self._token_embedding.forward(tokenised_tcrs[:, :, 0])
        position_component = self._position_embedding.forward(tokenised_tcrs[:, :, 1:3])
        compartment_component = self._compartment_embedding.forward(
            tokenised_tcrs[:, :, 3]
        )
        all_components_stacked = torch.concatenate(
            [token_component, position_component, compartment_component], dim=-1
        )
        return all_components_stacked * math.sqrt(self._embedding_dim)


class SingleChainCdrSimpleEmbedder(TokenEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self._token_embedding = OneHotTokenIndexEmbedding(AminoAcidTokenIndex)
        self._position_embedding = SimpleRelativePositionEmbedding()
        self._compartment_embedding = OneHotTokenIndexEmbedding(
            SingleChainCdrCompartmentIndex
        )

    def forward(self, tokenised_tcrs: LongTensor) -> FloatTensor:
        token_component = self._token_embedding.forward(tokenised_tcrs[:, :, 0])
        position_component = self._position_embedding.forward(tokenised_tcrs[:, :, 1:3])
        compartment_component = self._compartment_embedding.forward(
            tokenised_tcrs[:, :, 3]
        )
        all_components_stacked = torch.concatenate(
            [token_component, position_component, compartment_component], dim=-1
        )
        return all_components_stacked

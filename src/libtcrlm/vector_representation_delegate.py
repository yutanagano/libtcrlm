from abc import ABC, abstractmethod
from libtcrlm.self_attention_stack import SelfAttentionStack
from torch import Tensor
from torch.nn import functional as F


LOCATION_OF_CLS_TOKEN = 0


class VectorRepresentationDelegate(ABC):
    @abstractmethod
    def __init__(self, self_attention_stack: SelfAttentionStack) -> None:
        pass

    @abstractmethod
    def get_vector_representations_of(
        self, token_embeddings: Tensor, padding_mask: Tensor
    ) -> Tensor:
        pass


class AveragePoolVectorRepresentationDelegate(VectorRepresentationDelegate):
    def __init__(self, self_attention_stack: SelfAttentionStack) -> None:
        self._self_attention_stack = self_attention_stack

    def get_vector_representations_of(
        self, token_embeddings: Tensor, padding_mask: Tensor
    ) -> Tensor:
        embeddings_from_penultimate_layer = (
            self._self_attention_stack.get_token_embeddings_at_penultimate_layer(
                token_embeddings, padding_mask
            )
        )
        padding_and_cls_mask = self._edit_mask_to_also_mark_cls_tokens(padding_mask)

        aa_embeddings_averaged = self._masked_average_pool(
            embeddings_from_penultimate_layer, padding_and_cls_mask
        )

        l2_normed_embeddings = F.normalize(aa_embeddings_averaged, p=2, dim=1)

        return l2_normed_embeddings

    def _edit_mask_to_also_mark_cls_tokens(self, padding_mask: Tensor) -> Tensor:
        padding_mask_with_cls_also_masked = padding_mask.clone()
        padding_mask_with_cls_also_masked[:, 0] = 1

        return padding_mask_with_cls_also_masked

    def _masked_average_pool(
        self, token_embeddings: Tensor, padding_and_cls_mask: Tensor
    ) -> Tensor:
        zero_where_cls_and_padding = padding_and_cls_mask.logical_not()
        mask_broadcastable_with_embeddings = zero_where_cls_and_padding.unsqueeze(-1)

        only_non_cls_embeddings = token_embeddings * mask_broadcastable_with_embeddings
        token_embeddings_summed = only_non_cls_embeddings.sum(1)
        token_embeddings_averaged = (
            token_embeddings_summed / mask_broadcastable_with_embeddings.sum(1)
        )

        return token_embeddings_averaged


class ClsVectorRepresentationDelegate(VectorRepresentationDelegate):
    def __init__(self, self_attention_stack: SelfAttentionStack) -> None:
        self._self_attention_stack = self_attention_stack

    def get_vector_representations_of(
        self, token_embeddings: Tensor, padding_mask: Tensor
    ) -> Tensor:
        final_token_embeddings = self._self_attention_stack.forward(
            token_embeddings, padding_mask
        )
        final_cls_embeddings = final_token_embeddings[:, LOCATION_OF_CLS_TOKEN, :]
        l2_normed_cls_embeddings = F.normalize(final_cls_embeddings, p=2, dim=1)

        return l2_normed_cls_embeddings

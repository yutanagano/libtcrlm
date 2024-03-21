from types import ModuleType

import libtcrlm.tokeniser as tokeniser_module
import libtcrlm.self_attention_stack as self_attention_stack_module
import libtcrlm.mlm_token_prediction_projector as mlm_token_prediction_projector_module
import libtcrlm.vector_representation_delegate as vector_representation_delegate_module
import libtcrlm.token_embedder as token_embedder_module

from libtcrlm.tokeniser import Tokeniser
from libtcrlm.bert import Bert
from libtcrlm.token_embedder.token_embedder import TokenEmbedder
from libtcrlm.self_attention_stack import SelfAttentionStack
from libtcrlm.mlm_token_prediction_projector import MlmTokenPredictionProjector
from libtcrlm.vector_representation_delegate import VectorRepresentationDelegate


class ConfigReader:
    def __init__(self, config: dict) -> None:
        self._config = config

    def get_model_name(self) -> str:
        return self._config["model"]["name"]

    def get_config(self) -> dict:
        return self._config

    def get_tokeniser(self) -> Tokeniser:
        config = self._config["data"]["tokeniser"]
        return self._get_object_from_module_using_config(tokeniser_module, config)

    def get_bert(self) -> Bert:
        token_embedder = self._get_token_embedder()
        self_attention_stack = self._get_self_attention_stack()
        mlm_token_prediction_projector = self._get_mlm_token_prediction_projector()
        vector_representation_delegate = (
            self._get_vector_representation_delegate_for_self_attention_stack(
                self_attention_stack
            )
        )

        bert = Bert(
            token_embedder=token_embedder,
            self_attention_stack=self_attention_stack,
            mlm_token_prediction_projector=mlm_token_prediction_projector,
            vector_representation_delegate=vector_representation_delegate,
        )

        return bert

    def _get_token_embedder(self) -> TokenEmbedder:
        config = self._config["model"]["token_embedder"]
        return self._get_object_from_module_using_config(token_embedder_module, config)

    def _get_self_attention_stack(self) -> SelfAttentionStack:
        config = self._config["model"]["self_attention_stack"]
        return self._get_object_from_module_using_config(
            self_attention_stack_module, config
        )

    def _get_mlm_token_prediction_projector(
        self,
    ) -> MlmTokenPredictionProjector:
        config = self._config["model"]["mlm_token_prediction_projector"]
        return self._get_object_from_module_using_config(
            mlm_token_prediction_projector_module, config
        )

    def _get_vector_representation_delegate_for_self_attention_stack(
        self, self_attention_stack: SelfAttentionStack
    ) -> VectorRepresentationDelegate:
        config = self._config["model"]["vector_representation_delegate"]
        class_name = config["class"]
        initargs = config["initargs"]
        VectorRepresentationDelegateClass = getattr(
            vector_representation_delegate_module, class_name
        )
        return VectorRepresentationDelegateClass(
            self_attention_stack=self_attention_stack, **initargs
        )

    def _get_object_from_module_using_config(
        self, module: ModuleType, config: dict
    ) -> any:
        class_name = config["class"]
        initargs = config["initargs"]
        Class = getattr(module, class_name)
        return Class(**initargs)

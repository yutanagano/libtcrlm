from libtcrlm import schema
from libtcrlm.bert import Bert
from libtcrlm.token_embedder import SingleChainCdrEmbedder
from libtcrlm.self_attention_stack import SelfAttentionStackWithBuiltins
from libtcrlm.mlm_token_prediction_projector import AminoAcidTokenProjector
from libtcrlm.vector_representation_delegate import (
    AveragePoolVectorRepresentationDelegate,
)
import pandas as pd
from pathlib import Path
import pytest


@pytest.fixture
def toy_bert(toy_bert_d_model):
    token_embedder = SingleChainCdrEmbedder(embedding_dim=toy_bert_d_model)
    self_attention_stack = SelfAttentionStackWithBuiltins(
        num_layers=2, d_model=toy_bert_d_model, nhead=2
    )
    mlm_token_prediction_projector = AminoAcidTokenProjector(d_model=toy_bert_d_model)
    vector_representation_delegate = AveragePoolVectorRepresentationDelegate(
        self_attention_stack=self_attention_stack
    )

    bert = Bert(
        token_embedder=token_embedder,
        self_attention_stack=self_attention_stack,
        mlm_token_prediction_projector=mlm_token_prediction_projector,
        vector_representation_delegate=vector_representation_delegate,
    )

    return bert


@pytest.fixture
def toy_bert_d_model():
    return 4


@pytest.fixture
def mock_tcr():
    return schema.make_tcr_from_components("TRAV1-1*01", "CATQYF", "TRBV2*01", "CASQYF")


@pytest.fixture
def mock_data_path():
    return Path("tests") / "resources" / "mock_data.csv"


@pytest.fixture
def mock_data_df(mock_data_path):
    return pd.read_csv(mock_data_path)

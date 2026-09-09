"""
test_models.py
Unit tests for Task 1, 2, 3, 4 models and baseline architectures.
"""

import torch
import pytest

from src.bert_encoder import MusicBERTClassifier, tokenize_texts
from src.gnn_model import MusicGNNEncoder, RelationAwareGraphSAGEConv
from src.fusion_model import GNNBERTFusionModel
from src.contrastive import DualEncoderContrastive
from src.baselines import MelSpectrogramCNN, HandcraftedMLPBaseline, RandomMajorityPredictor


def test_bert_classifier():
    model = MusicBERTClassifier(num_classes=25, embedding_dim=128)
    texts = ["An energetic rock track with drums and guitar.", "Calm jazz piano melody."]
    tok = tokenize_texts(texts, max_length=16)

    out = model(tok["input_ids"], tok["attention_mask"])
    assert out["logits"].shape == (2, 25)
    assert out["probs"].shape == (2, 25)
    assert out["cls_embedding"].shape[0] == 2


def test_gnn_encoder():
    model = MusicGNNEncoder(in_dim=32, hidden_dim=64, out_dim=64, num_classes=25)
    x = torch.randn(8, 32)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)
    edge_weight = torch.ones(7, dtype=torch.float32)

    # Single graph forward
    out = model(x, edge_index, edge_weight)
    assert out["logits"].shape == (1, 25)
    assert out["graph_embedding"].shape == (1, 64)

    # Batched forward
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    out_batch = model(x, edge_index, edge_weight, batch=batch)
    assert out_batch["logits"].shape == (2, 25)
    assert out_batch["graph_embedding"].shape == (2, 64)


def test_fusion_model():
    gnn = MusicGNNEncoder(in_dim=32, hidden_dim=64, out_dim=64, num_classes=25)
    bert = MusicBERTClassifier(num_classes=25, embedding_dim=128)
    fusion = GNNBERTFusionModel(gnn, bert, fusion_type="cross_attention", attn_dim=64, num_classes=25)

    x = torch.randn(6, 32)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    tok = tokenize_texts(["Rock music", "Jazz music"], max_length=16)

    out = fusion(x, edge_index, batch=batch, input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
    assert out["logits"].shape == (2, 25)
    assert out["valence"].shape == (2,)
    assert out["arousal"].shape == (2,)
    assert out["z"].shape[0] == 2

    # Multi-task loss
    tags = torch.zeros((2, 25))
    tags[0, 0] = 1.0
    val = torch.tensor([5.0, 7.0])
    aro = torch.tensor([6.0, 3.0])

    loss_dict = fusion.compute_loss(out, tags, val, aro)
    assert "loss" in loss_dict
    assert loss_dict["loss"].item() > 0


def test_contrastive_dual_encoder():
    gnn = MusicGNNEncoder(in_dim=32, hidden_dim=64, out_dim=64, num_classes=25)
    bert = MusicBERTClassifier(num_classes=25, embedding_dim=128)
    dual = DualEncoderContrastive(gnn, bert, projection_dim=64)

    x = torch.randn(4, 32)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    tok = tokenize_texts(["Description one", "Description two"], max_length=16)

    g_emb, t_emb = dual(x, edge_index, batch=batch, input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
    assert g_emb.shape == (2, 64)
    assert t_emb.shape == (2, 64)

    loss, sim = dual.compute_infonce_loss(g_emb, t_emb)
    assert loss.item() > 0
    assert sim.shape == (2, 2)


def test_baselines():
    # B1: Random / Majority
    b1 = RandomMajorityPredictor(num_classes=25)
    probs = b1.predict_proba(10)
    assert probs.shape == (10, 25)

    # B2: CNN Mel-Spec
    b2 = MelSpectrogramCNN(num_classes=25)
    mel = torch.randn(4, 1, 128, 50)
    out_b2 = b2(mel)
    assert out_b2["logits"].shape == (4, 25)

    # B4: Handcrafted MLP
    b4 = HandcraftedMLPBaseline(in_features=48, num_classes=25)
    feat = torch.randn(4, 48)
    out_b4 = b4(feat)
    assert out_b4["logits"].shape == (4, 25)


def test_relation_aware_graphsage_conv():
    conv = RelationAwareGraphSAGEConv(in_channels=32, out_channels=64)
    x = torch.randn(6, 32)
    # Temporal edges: (0,1), (1,2), (3,4), (4,5)
    # Recurrence edges: (0,3), (1,4), (2,5)
    edge_index = torch.tensor([
        [0, 1, 1, 2, 3, 4, 4, 5, 0, 1, 2],
        [1, 0, 2, 1, 4, 3, 5, 4, 3, 4, 5]
    ], dtype=torch.long)
    edge_type = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.long)
    edge_weight = torch.ones(11, dtype=torch.float32)

    # Forward with explicit edge_type
    out1 = conv(x, edge_index, edge_weight, edge_type=edge_type)
    assert out1.shape == (6, 64)

    # Forward with inferred edge_type (edge_type=None)
    out2 = conv(x, edge_index, edge_weight)
    assert out2.shape == (6, 64)


def test_relation_aware_gnn_encoder():
    model = MusicGNNEncoder(
        in_dim=32,
        hidden_dim=64,
        out_dim=64,
        num_classes=25,
        num_layers=2,
        gnn_type="relation_aware_graphsage"
    )
    x = torch.randn(8, 32)
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 0], [1, 2, 3, 4, 5, 6, 7, 5]], dtype=torch.long)
    edge_weight = torch.ones(8, dtype=torch.float32)
    edge_type = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.long)

    # Single graph forward
    out = model(x, edge_index, edge_weight, edge_type=edge_type)
    assert out["logits"].shape == (1, 25)
    assert out["probs"].shape == (1, 25)
    assert out["graph_embedding"].shape == (1, 64)
    assert out["node_embeddings"].shape == (8, 64)

    # Batched forward without explicit edge_type (auto-inferred)
    batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    out_batch = model(x, edge_index, edge_weight, batch=batch)
    assert out_batch["logits"].shape == (2, 25)
    assert out_batch["graph_embedding"].shape == (2, 64)


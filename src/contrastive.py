"""
contrastive.py
Task 4 (Advanced): Cross-Modal MusicCaps Alignment.
Learns a shared embedding space between audio structure graphs and natural-language
descriptions using InfoNCE contrastive learning per Section 4.4 of the project specification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any

from .bert_encoder import MusicBERTClassifier
from .gnn_model import MusicGNNEncoder


class DualEncoderContrastive(nn.Module):
    """
    Dual-Encoder architecture projecting audio graphs g_i and captions t_i
    into a shared, normalized D-dimensional latent space.
    """
    def __init__(
        self,
        gnn_encoder: MusicGNNEncoder,
        bert_encoder: MusicBERTClassifier,
        projection_dim: int = 128,
        temperature: float = 0.07,
        learnable_temperature: bool = True
    ):
        super().__init__()
        self.gnn = gnn_encoder
        self.bert = bert_encoder
        self.projection_dim = projection_dim

        # Graph projection head
        self.graph_proj = nn.Sequential(
            nn.Linear(gnn_encoder.out_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

        # Text projection head
        self.text_proj = nn.Sequential(
            nn.Linear(bert_encoder.embedding_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

        # Temperature parameter tau
        if learnable_temperature:
            self.logit_scale = nn.Parameter(torch.ones([]) * (1.0 / temperature))
        else:
            self.register_buffer("logit_scale", torch.tensor(1.0 / temperature))

    def encode_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extracts and normalizes graph representation: g_i / ||g_i||
        """
        g, _ = self.gnn.forward_graph(x, edge_index, edge_weight, batch)
        proj = self.graph_proj(g)
        return F.normalize(proj, p=2, dim=-1)

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extracts and normalizes text CLS representation: t_i / ||t_i||
        """
        bert_out = self.bert(input_ids, attention_mask=attention_mask)
        t = bert_out["cls_embedding"]
        proj = self.text_proj(t)
        return F.normalize(proj, p=2, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        g_emb = self.encode_graph(x, edge_index, edge_weight, batch)
        t_emb = self.encode_text(input_ids, attention_mask)
        return g_emb, t_emb

    def compute_infonce_loss(
        self,
        g_emb: torch.Tensor,
        t_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes symmetric InfoNCE loss:
        S_ij = g_i^T t_j / tau
        L_NCE = -0.5 * (log(exp(S_ii) / sum_j exp(S_ij)) + log(exp(S_ii) / sum_j exp(S_ji)))
        """
        scale = torch.clamp(self.logit_scale, max=100.0)
        # Similarity matrix: (B, B)
        sim_matrix = torch.matmul(g_emb, t_emb.t()) * scale

        labels = torch.arange(g_emb.size(0), device=g_emb.device)
        loss_g2t = F.cross_entropy(sim_matrix, labels)
        loss_t2g = F.cross_entropy(sim_matrix.t(), labels)
        total_loss = 0.5 * (loss_g2t + loss_t2g)

        return total_loss, sim_matrix


def compute_retrieval_metrics(
    query_embeds: torch.Tensor,
    target_embeds: torch.Tensor,
    k_vals: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Computes Recall@K (R@1, R@5, R@10) and Mean Reciprocal Rank (MRR).
    query_embeds: (N, D) normalized
    target_embeds: (N, D) normalized
    """
    # Dot product similarity: (N, N)
    sim = torch.matmul(query_embeds, target_embeds.t())
    N = sim.size(0)

    # Ground truth is diagonal (i -> i)
    # Get ranked indices per query
    ranks = torch.argsort(sim, dim=-1, descending=True)
    targets = torch.arange(N, device=sim.device).unsqueeze(1)

    # Position of true target in ranked list (0-indexed)
    matched_pos = (ranks == targets).nonzero(as_tuple=True)[1].float()

    metrics = {}
    for k in k_vals:
        r_at_k = (matched_pos < k).float().mean().item()
        metrics[f"R@{k}"] = float(r_at_k)

    mrr = (1.0 / (matched_pos + 1.0)).mean().item()
    metrics["MRR"] = float(mrr)
    return metrics

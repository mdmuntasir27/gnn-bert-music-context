"""
fusion_model.py
Task 3 (Hard): GNN-BERT Fusion for Multi-Context Understanding.
Combines structural graph representations and contextual text representations
via Cross-Attention Fusion or Early Concat, with multi-task prediction heads
(multi-label tag classification + valence/arousal emotion regression) per Section 4.3.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any

from .bert_encoder import MusicBERTClassifier
from .gnn_model import MusicGNNEncoder


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention mechanism:
    Q = g * W_Q (from graph embedding g: (B, 1, d_attn))
    K = H_text * W_K (from token representations H_text: (B, L, d_attn))
    V = H_text * W_V
    A = softmax(Q * K^T / sqrt(d))
    context = A * V
    z = CONCAT(g, context)
    """
    def __init__(self, gnn_dim: int, text_dim: int, attn_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.gnn_dim = gnn_dim
        self.text_dim = text_dim
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads

        assert attn_dim % num_heads == 0, "attn_dim must be divisible by num_heads"

        self.w_q = nn.Linear(gnn_dim, attn_dim)
        self.w_k = nn.Linear(text_dim, attn_dim)
        self.w_v = nn.Linear(text_dim, attn_dim)
        self.w_out = nn.Linear(attn_dim, text_dim)
        self.gate = nn.Sequential(
            nn.Linear(gnn_dim + text_dim, text_dim),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        g: torch.Tensor,
        h_text: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            g: Graph embedding (B, gnn_dim)
            h_text: Text token representations (B, L, text_dim)
            t: CLS embedding (B, text_dim)
            text_mask: (B, L) with 1 for real tokens, 0 for padding
        Returns:
            z (torch.Tensor): Fused representation
            attn_weights (torch.Tensor): Cross-attention map (B, num_heads, 1, L)
        """
        B = g.size(0)
        L = h_text.size(1)

        # Q from graph: (B, 1, attn_dim) -> (B, num_heads, 1, head_dim)
        q = self.w_q(g).unsqueeze(1)
        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # K, V from text: (B, L, attn_dim) -> (B, num_heads, L, head_dim)
        k = self.w_k(h_text).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(h_text).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim) # (B, num_heads, 1, L)

        if text_mask is not None:
            # Mask out padding tokens
            mask = text_mask.unsqueeze(1).unsqueeze(2) # (B, 1, 1, L)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of text tokens
        context = torch.matmul(attn_weights, v) # (B, num_heads, 1, head_dim)
        context = context.transpose(1, 2).contiguous().view(B, 1, self.attn_dim).squeeze(1) # (B, attn_dim)
        context = self.w_out(context)

        # Gated Residual Fusion:
        if t is not None:
            g_gate = self.gate(torch.cat([g, t], dim=-1))
            fused_text = t + g_gate * context
            z = torch.cat([g, fused_text], dim=-1)
        else:
            z = torch.cat([g, context], dim=-1)
        return z, attn_weights


class GNNBERTFusionModel(nn.Module):
    """
    Complete End-to-End GNN-BERT Fusion System with Multi-Task Heads:
    - Multi-label tag classification (genre, mood, instrument)
    - Emotion regression (Valence and Arousal in range [1, 9] per DEAM)
    Supports ablation modes:
        - 'cross_attention' (full model)
        - 'early_concat' (simple concatenation of g and t)
        - 'bert_only' (text-only baseline B3)
        - 'gnn_only' (audio graph-only baseline Task 2)
    """
    def __init__(
        self,
        gnn_encoder: MusicGNNEncoder,
        bert_encoder: MusicBERTClassifier,
        fusion_type: str = "cross_attention",
        attn_dim: int = 128,
        num_heads: int = 4,
        num_classes: int = 25,
        dropout: float = 0.3
    ):
        super().__init__()
        self.gnn = gnn_encoder
        self.bert = bert_encoder
        self.fusion_type = fusion_type.lower()
        self.num_classes = num_classes

        gnn_dim = gnn_encoder.out_dim
        text_dim = bert_encoder.embedding_dim

        if self.fusion_type == "cross_attention":
            self.cross_attn = CrossAttentionFusion(gnn_dim, text_dim, attn_dim, num_heads, dropout=0.1)
            fusion_dim = gnn_dim + text_dim
        elif self.fusion_type == "early_concat":
            fusion_dim = gnn_dim + text_dim
        elif self.fusion_type == "bert_only":
            fusion_dim = text_dim
        elif self.fusion_type == "gnn_only":
            fusion_dim = gnn_dim
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        self.norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)

        # Multi-task Prediction Heads
        # 1. Multi-label context tags
        self.tag_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        # 2. Valence regression head (bounded to DEAM [1.0, 9.0])
        self.valence_head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # 3. Arousal regression head (bounded to DEAM [1.0, 9.0])
        self.arousal_head = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Forward pass producing multi-task predictions and latent fusion vector z.
        """
        attn_weights = None

        # 1. Audio Graph Representation
        if self.fusion_type != "bert_only":
            g_out = self.gnn(x, edge_index, edge_weight=edge_weight, batch=batch)
            g = g_out["graph_embedding"]
        else:
            g = None

        # 2. Linguistic Text Representation
        if self.fusion_type != "gnn_only" and input_ids is not None:
            bert_out = self.bert(input_ids, attention_mask=attention_mask)
            h_text = bert_out["sequence_output"]  # (B, L, d)
            t = bert_out["cls_embedding"]         # (B, d)
        else:
            h_text = None
            t = None

        # 3. Perform Fusion
        if self.fusion_type == "cross_attention":
            z, attn_weights = self.cross_attn(g, h_text, t=t, text_mask=attention_mask)
        elif self.fusion_type == "early_concat":
            z = torch.cat([g, t], dim=-1)
        elif self.fusion_type == "bert_only":
            z = t
        elif self.fusion_type == "gnn_only":
            z = g

        z_norm = self.norm(z)
        z_dropped = self.dropout(z_norm)

        # 4. Multi-task predictions
        tag_logits = self.tag_head(z_dropped)
        tag_probs = torch.sigmoid(tag_logits)
        pred_valence = 1.0 + 8.0 * self.valence_head(z_dropped).squeeze(-1)
        pred_arousal = 1.0 + 8.0 * self.arousal_head(z_dropped).squeeze(-1)

        return {
            "logits": tag_logits,
            "probs": tag_probs,
            "valence": pred_valence,
            "arousal": pred_arousal,
            "z": z_norm,
            "attn_weights": attn_weights,
            "g": g,
            "t": t
        }

    def compute_loss(
        self,
        outputs: Dict[str, Any],
        true_tags: torch.Tensor,
        true_valence: Optional[torch.Tensor] = None,
        true_arousal: Optional[torch.Tensor] = None,
        alpha: float = 0.5,
        beta: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Computes Section 4.3 Multi-task loss:
        L = L_tags + alpha * ||v - v_hat||^2 + beta * ||a - a_hat||^2
        """
        pos_weight = torch.ones(true_tags.size(-1), device=true_tags.device) * 3.5
        bce_loss = F.binary_cross_entropy_with_logits(outputs["logits"], true_tags, pos_weight=pos_weight)
        total_loss = bce_loss
        loss_v = torch.tensor(0.0, device=true_tags.device)
        loss_a = torch.tensor(0.0, device=true_tags.device)

        if true_valence is not None and outputs["valence"] is not None:
            # Normalize DEAM scale [1, 9] to [0, 1] for balanced MSE magnitude with BCE
            norm_v = (true_valence - 1.0) / 8.0
            norm_pred_v = (outputs["valence"] - 1.0) / 8.0
            loss_v = F.mse_loss(norm_pred_v, norm_v)
            total_loss = total_loss + alpha * loss_v

        if true_arousal is not None and outputs["arousal"] is not None:
            norm_a = (true_arousal - 1.0) / 8.0
            norm_pred_a = (outputs["arousal"] - 1.0) / 8.0
            loss_a = F.mse_loss(norm_pred_a, norm_a)
            total_loss = total_loss + beta * loss_a

        return {
            "loss": total_loss,
            "loss_tags": bce_loss,
            "loss_valence": loss_v,
            "loss_arousal": loss_a
        }

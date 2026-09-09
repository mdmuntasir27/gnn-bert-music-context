"""
gnn_model.py
Task 2 (Medium): GNN on Music Structure Graphs.
Implements GraphSAGE and GAT architectures with mean/max readout on music segment/chord graphs
per Section 4.2 of the project specification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any

try:
    from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool, global_max_pool
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False


class DenseGraphSAGEConv(nn.Module):
    """
    Pure PyTorch implementation of GraphSAGE:
    h_i^(l+1) = sigma(W * CONCAT(h_i^(l), MEAN_{j in N(i)} h_j^(l)))
    Ensures zero-dependency fallback that matches the exact formula in the PDF.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin_self = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x: (N, in_channels)
        edge_index: (2, E)
        """
        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # Aggregate neighbor features via scatter mean
        if edge_weight is None:
            weights = torch.ones_like(src, dtype=torch.float32, device=x.device)
        else:
            weights = edge_weight

        # Weighted neighbor aggregation
        neigh_sum = torch.zeros((num_nodes, x.size(1)), device=x.device)
        deg = torch.zeros((num_nodes, 1), device=x.device)

        neigh_sum.index_add_(0, dst, x[src] * weights.unsqueeze(-1))
        deg.index_add_(0, dst, weights.unsqueeze(-1))
        deg = torch.clamp(deg, min=1.0)
        neigh_mean = neigh_sum / deg

        # Concat update
        out = self.lin_self(x) + self.lin_neigh(neigh_mean)
        return out


class RelationAwareGraphSAGEConv(nn.Module):
    """
    Relation-Aware GraphSAGE Convolution for music segment graphs:
    m_i^{temp} = MEAN_{j in N_temp(i)} h_j
    m_i^{rec}  = MEAN_{j in N_rec(i)} h_j
    gamma_i    = sigma(W_g [h_i || m_i^{temp} || m_i^{rec}])
    m_i        = gamma_i * m_i^{temp} + (1 - gamma_i) * m_i^{rec}
    h_i^{l+1}  = sigma(W_self * h_i + W_neigh * m_i + b)
    """
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin_self = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.gate_linear = nn.Linear(in_channels * 3, in_channels, bias=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_self.weight)
        nn.init.xavier_uniform_(self.lin_neigh.weight)
        nn.init.xavier_uniform_(self.gate_linear.weight)
        nn.init.zeros_(self.gate_linear.bias)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        edge_type: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        num_nodes = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        if edge_weight is None:
            weights = torch.ones_like(src, dtype=torch.float32, device=x.device)
        else:
            weights = edge_weight

        # Infer edge types if not explicitly provided:
        # Type 0 (temporal adjacency): |src - dst| == 1
        # Type 1 (recurrence/similarity): |src - dst| != 1
        if edge_type is None:
            is_temporal = (torch.abs(src - dst) == 1)
            edge_type = torch.where(is_temporal, torch.zeros_like(src), torch.ones_like(src))

        mask_temp = (edge_type == 0)
        mask_rec = (edge_type == 1)

        # 1. Temporal neighbor aggregation: m_i^{temp} = MEAN_{j in N_temp(i)} h_j
        neigh_sum_temp = torch.zeros((num_nodes, x.size(1)), device=x.device)
        deg_temp = torch.zeros((num_nodes, 1), device=x.device)
        if mask_temp.any():
            src_t, dst_t, w_t = src[mask_temp], dst[mask_temp], weights[mask_temp]
            neigh_sum_temp.index_add_(0, dst_t, x[src_t] * w_t.unsqueeze(-1))
            deg_temp.index_add_(0, dst_t, w_t.unsqueeze(-1))
        deg_temp_clamp = torch.clamp(deg_temp, min=1.0)
        m_temp = neigh_sum_temp / deg_temp_clamp

        # 2. Recurrence neighbor aggregation: m_i^{rec} = MEAN_{j in N_rec(i)} h_j
        neigh_sum_rec = torch.zeros((num_nodes, x.size(1)), device=x.device)
        deg_rec = torch.zeros((num_nodes, 1), device=x.device)
        if mask_rec.any():
            src_r, dst_r, w_r = src[mask_rec], dst[mask_rec], weights[mask_rec]
            neigh_sum_rec.index_add_(0, dst_r, x[src_r] * w_r.unsqueeze(-1))
            deg_rec.index_add_(0, dst_r, w_r.unsqueeze(-1))
        deg_rec_clamp = torch.clamp(deg_rec, min=1.0)
        m_rec = neigh_sum_rec / deg_rec_clamp

        # 3. Learnable sigmoid gating: gamma_i = sigma(W_g [h_i || m_i^{temp} || m_i^{rec}])
        gate_input = torch.cat([x, m_temp, m_rec], dim=-1)
        gamma = torch.sigmoid(self.gate_linear(gate_input))

        # Respect degree presence: if node has only temporal or only recurrence neighbors
        has_temp = (deg_temp > 0).float()
        has_rec = (deg_rec > 0).float()
        effective_gamma = torch.where(
            (has_temp > 0) & (has_rec == 0),
            torch.ones_like(gamma),
            torch.where(
                (has_temp == 0) & (has_rec > 0),
                torch.zeros_like(gamma),
                gamma
            )
        )

        # 4. Gated combination: m_i = gamma_i * m_i^{temp} + (1 - gamma_i) * m_i^{rec}
        m = effective_gamma * m_temp + (1.0 - effective_gamma) * m_rec

        # 5. GraphSAGE update: W_self * h_i + W_neigh * m_i
        out = self.lin_self(x) + self.lin_neigh(m)
        if self.bias is not None:
            out = out + self.bias
        return out


class MusicGNNEncoder(nn.Module):
    """
    GNN Encoder on music segment/chord graphs.
    Supports GraphSAGE, GAT, and RelationAwareGraphSAGE layers with multi-layer
    message passing and graph readout.
    """
    def __init__(
        self,
        in_dim: int = 32,
        hidden_dim: int = 64,
        out_dim: int = 64,
        num_classes: int = 25,
        num_layers: int = 2,
        gnn_type: str = "graphsage",
        num_heads: int = 4,
        dropout: float = 0.2,
        readout: str = "mean"
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.gnn_type = gnn_type.lower()
        self.readout_type = readout.lower()
        self.dropout = nn.Dropout(dropout)

        # Build message passing layers
        self.layers = nn.ModuleList()
        current_dim = in_dim

        for i in range(num_layers):
            layer_out = out_dim if i == num_layers - 1 else hidden_dim
            if self.gnn_type in ["relation_aware_graphsage", "relation_sage", "relation_aware"]:
                self.layers.append(RelationAwareGraphSAGEConv(current_dim, layer_out))
            elif PYG_AVAILABLE:
                if self.gnn_type == "gat":
                    self.layers.append(GATConv(current_dim, layer_out // num_heads, heads=num_heads, concat=True))
                else:
                    self.layers.append(SAGEConv(current_dim, layer_out, aggr="mean"))
            else:
                self.layers.append(DenseGraphSAGEConv(current_dim, layer_out))
            current_dim = layer_out

        # Track-level classification head
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        edge_type: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Runs message passing and returns node features and track-level graph embedding g.
        Args:
            x: (N, in_dim) node features
            edge_index: (2, E)
            edge_weight: (E,) optional edge weights
            batch: (N,) tensor mapping nodes to batch elements
            edge_type: (E,) optional edge types (0: temporal, 1: recurrence)
        Returns:
            g (torch.Tensor): Graph-level representation (B, out_dim)
            h (torch.Tensor): Final layer node representations (N, out_dim)
        """
        h = x
        for i, layer in enumerate(self.layers):
            if isinstance(layer, RelationAwareGraphSAGEConv):
                h = layer(h, edge_index, edge_weight, edge_type=edge_type)
            elif PYG_AVAILABLE and not isinstance(layer, DenseGraphSAGEConv):
                h = layer(h, edge_index)
            else:
                h = layer(h, edge_index, edge_weight)

            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)

        # Graph Readout pooling
        if batch is None:
            # Single graph: average all nodes
            if self.readout_type == "sum":
                g = torch.sum(h, dim=0, keepdim=True)
            elif self.readout_type == "max":
                g, _ = torch.max(h, dim=0, keepdim=True)
            else:
                g = torch.mean(h, dim=0, keepdim=True)
        else:
            # Batched graph
            if PYG_AVAILABLE and not any(isinstance(l, (DenseGraphSAGEConv, RelationAwareGraphSAGEConv)) for l in self.layers):
                if self.readout_type == "sum":
                    import torch_geometric.nn
                    g = torch_geometric.nn.global_add_pool(h, batch)
                elif self.readout_type == "max":
                    g = global_max_pool(h, batch)
                else:
                    g = global_mean_pool(h, batch)
            else:
                # Manual batched pooling
                batch_size = int(batch.max().item()) + 1
                g_list = []
                for b in range(batch_size):
                    mask = (batch == b)
                    if mask.sum() > 0:
                        if self.readout_type == "max":
                            g_list.append(torch.max(h[mask], dim=0)[0])
                        elif self.readout_type == "sum":
                            g_list.append(torch.sum(h[mask], dim=0))
                        else:
                            g_list.append(torch.mean(h[mask], dim=0))
                    else:
                        g_list.append(torch.zeros(h.size(1), device=h.device))
                g = torch.stack(g_list, dim=0)

        return g, h

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        edge_type: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Full classification forward pass for Task 2.
        """
        g, h = self.forward_graph(x, edge_index, edge_weight, batch, edge_type=edge_type)
        logits = self.classifier(self.dropout(g))
        probs = torch.sigmoid(logits)
        return {
            "logits": logits,
            "probs": probs,
            "graph_embedding": g,
            "node_embeddings": h
        }

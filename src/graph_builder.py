"""
graph_builder.py
Constructs music structure graphs:
1. Segment graph: nodes = time segments; edges = temporal adjacency + cosine similarity of MFCC/chroma > tau
2. Chord-transition graph: nodes = unique chords; edges = observed transitions weighted by count
Outputs PyTorch Geometric Data objects and JSON serializable dictionaries.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

try:
    import torch
    from torch_geometric.data import Data as PyGData
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def cosine_similarity_matrix(features: np.ndarray) -> np.ndarray:
    """
    Computes pairwise cosine similarity between node feature rows.
    """
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1e-8, norms)
    normalized = features / norms
    return np.dot(normalized, normalized.T)


def build_segment_similarity_graph(
    segment_features: np.ndarray,
    tau: float = 0.65,
    top_k: int = 3,
    connect_temporal_adjacency: bool = True
) -> Dict[str, Any]:
    """
    Builds a segment-based music structure graph.
    Nodes = time segments.
    Edges = temporal adjacency (i <-> i+1) + cosine similarity > tau (k-nearest neighbors).
    
    Args:
        segment_features (np.ndarray): Shape (num_segments, feature_dim)
        tau (float): Cosine similarity threshold
        top_k (int): Maximum similarity neighbors per node
        connect_temporal_adjacency (bool): Whether consecutive time segments share an edge
        
    Returns:
        dict: Graph representation with nodes, edge_index, edge_weights, num_nodes
    """
    num_nodes = len(segment_features)
    if num_nodes == 0:
        return {"x": [], "edge_index": [[], []], "edge_weight": [], "num_nodes": 0}

    edges = set()
    edge_weights = {}
    edge_types = {}

    # 1. Temporal Adjacency edges (i <-> i + 1) -> Edge Type 0
    if connect_temporal_adjacency and num_nodes > 1:
        for i in range(num_nodes - 1):
            edges.add((i, i + 1))
            edges.add((i + 1, i))
            edge_weights[(i, i + 1)] = 1.0
            edge_weights[(i + 1, i)] = 1.0
            edge_types[(i, i + 1)] = 0
            edge_types[(i + 1, i)] = 0

    # 2. Cosine Similarity edges > tau -> Edge Type 1 (Recurrence / Similarity)
    if num_nodes > 1:
        sim_matrix = cosine_similarity_matrix(segment_features)
        for i in range(num_nodes):
            # Sort similarities excluding self-loop
            sim_scores = [(j, sim_matrix[i, j]) for j in range(num_nodes) if i != j]
            sim_scores.sort(key=lambda x: x[1], reverse=True)

            added = 0
            for j, score in sim_scores:
                if score >= tau and added < top_k:
                    # If edge not already present as temporal adjacency, mark as recurrence (type 1)
                    if (i, j) not in edge_types:
                        edge_types[(i, j)] = 1
                    edges.add((i, j))
                    edge_weights[(i, j)] = max(float(score), edge_weights.get((i, j), 0.0))
                    added += 1

    # Format edges
    if len(edges) > 0:
        src = [e[0] for e in edges]
        dst = [e[1] for e in edges]
        weights = [edge_weights[e] for e in edges]
        types = [edge_types.get(e, 0) for e in edges]
    else:
        # Fallback self-loops if isolated
        src = list(range(num_nodes))
        dst = list(range(num_nodes))
        weights = [1.0] * num_nodes
        types = [0] * num_nodes

    return {
        "x": segment_features.tolist(),
        "edge_index": [src, dst],
        "edge_weight": weights,
        "edge_type": types,
        "num_nodes": num_nodes,
        "feature_dim": segment_features.shape[1]
    }


def build_chord_transition_graph(
    chord_sequence: List[str],
    chord_dim: int = 32
) -> Dict[str, Any]:
    """
    Builds a chord-transition graph:
    Nodes = unique chords in track.
    Edges = observed transitions weighted by transition count.
    """
    if not chord_sequence:
        return {"x": [], "edge_index": [[], []], "edge_weight": [], "num_nodes": 0}

    unique_chords = sorted(list(set(chord_sequence)))
    chord_to_idx = {chord: i for i, chord in enumerate(unique_chords)}
    num_nodes = len(unique_chords)

    # Simple 12-pitch chroma one-hot or projection for chord representations
    NOTE_MAP = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
    node_features = np.zeros((num_nodes, chord_dim), dtype=np.float32)
    for chord, idx in chord_to_idx.items():
        # Assign base note chroma
        base_note = chord.replace('m', '').replace('7', '')
        note_id = NOTE_MAP.get(base_note, 0)
        node_features[idx, note_id] = 1.0
        # Add fifth
        node_features[idx, (note_id + 7) % 12] = 0.8
        # Minor vs major third
        if 'm' in chord:
            node_features[idx, (note_id + 3) % 12] = 0.8
        else:
            node_features[idx, (note_id + 4) % 12] = 0.8

    # Count transitions
    transition_counts = {}
    for i in range(len(chord_sequence) - 1):
        u = chord_to_idx[chord_sequence[i]]
        v = chord_to_idx[chord_sequence[i + 1]]
        transition_counts[(u, v)] = transition_counts.get((u, v), 0) + 1

    src, dst, weights = [], [], []
    for (u, v), count in transition_counts.items():
        src.append(u)
        dst.append(v)
        weights.append(float(count))

    if not src:
        src = list(range(num_nodes))
        dst = list(range(num_nodes))
        weights = [1.0] * num_nodes

    return {
        "x": node_features.tolist(),
        "edge_index": [src, dst],
        "edge_weight": weights,
        "num_nodes": num_nodes,
        "chord_names": unique_chords,
        "feature_dim": chord_dim
    }


def to_pyg_data(graph_dict: Dict[str, Any], labels: Optional[Any] = None) -> Any:
    """
    Converts graph dictionary to a PyTorch Geometric Data instance.
    """
    if not TORCH_AVAILABLE:
        return graph_dict

    x = torch.tensor(graph_dict["x"], dtype=torch.float32)
    edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
    edge_weight = torch.tensor(graph_dict["edge_weight"], dtype=torch.float32)

    data = PyGData(x=x, edge_index=edge_index, edge_weight=edge_weight)
    data.num_nodes = graph_dict["num_nodes"]
    if "edge_type" in graph_dict:
        data.edge_type = torch.tensor(graph_dict["edge_type"], dtype=torch.long)

    if labels is not None:
        if isinstance(labels, (list, np.ndarray)):
            data.y = torch.tensor(labels, dtype=torch.float32)
        elif isinstance(labels, torch.Tensor):
            data.y = labels.float()

    return data


def save_graph_sample(graph_dict: Dict[str, Any], file_path: str):
    """
    Saves graph sample as .json or .pt file.
    """
    if file_path.endswith(".pt") and TORCH_AVAILABLE:
        pyg_data = to_pyg_data(graph_dict)
        torch.save(pyg_data, file_path)
    else:
        json_path = file_path if file_path.endswith(".json") else file_path + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(graph_dict, f, indent=2)

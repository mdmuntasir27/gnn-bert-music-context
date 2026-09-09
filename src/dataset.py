"""
dataset.py
Handles dataset construction, multi-modal data loading (audio, graph, text, emotion targets),
strict 3-way dataset splitting (training, validation, test) without track/artist leakage,
and generation of preprocessed .pt/.json graph samples.
"""

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any

from .audio_features import generate_synthetic_audio, segment_audio, extract_segment_features, extract_features
from .graph_builder import build_segment_similarity_graph, build_chord_transition_graph, to_pyg_data, save_graph_sample
from .bert_encoder import tokenize_texts

# Top musical context tags (combining genres, moods, instrumentation per FMA/MagnaTagATune/MusicCaps)
CONTEXT_TAGS = [
    # Genres
    "rock", "jazz", "classical", "pop", "electronic", "folk", "hip-hop", "ambient",
    # Moods
    "happy", "sad", "energetic", "calm", "melancholic", "uplifting", "dark", "romantic",
    # Instruments & Context
    "acoustic guitar", "electric guitar", "piano", "synthesizer", "drums", "bass", "strings", "vocal", "fast tempo"
]

SAMPLE_ARTISTS = [
    "Aurora Sound", "Echo Horizon", "Velvet Symphony", "Neon Pulse",
    "Silver Strings", "Midnight Quartet", "Solar Drift", "Rustic Winds",
    "Iron Resonance", "Lucid Dreamers", "Crystal Ensemble", "Urban Beat",
    "Harmonic Tide", "Starlight Rhythm", "Aether Wave", "Golden Era"
]

GENRE_MOOD_MAP = {
    "rock": (["energetic", "electric guitar", "drums", "fast tempo"], (3.5, 7.5)),
    "jazz": (["calm", "romantic", "acoustic guitar", "piano", "bass"], (6.5, 5.0)),
    "classical": (["calm", "melancholic", "piano", "strings"], (5.0, 3.5)),
    "pop": (["happy", "uplifting", "vocal", "synthesizer", "drums"], (7.5, 7.0)),
    "electronic": (["energetic", "synthesizer", "drums", "fast tempo"], (6.0, 8.0)),
    "folk": (["calm", "acoustic guitar", "vocal", "melancholic"], (5.5, 4.0)),
    "ambient": (["calm", "dark", "synthesizer"], (4.5, 2.5)),
    "hip-hop": (["energetic", "drums", "bass", "vocal"], (6.5, 7.5))
}


class MusicDataset(Dataset):
    """
    Multi-modal Music Dataset containing:
    - Audio features (Log-Mel Spectrogram, Chroma)
    - Music Structure Graph (Segment Similarity Graph or Chord Transition Graph)
    - Text context (Natural language caption or concatenated tags)
    - Multi-label context tag targets (Binary vector of length K)
    - Auxiliary continuous emotion targets: (valence, arousal) in range [1, 9]
    """
    def __init__(self, data_list: List[Dict[str, Any]], config: Optional[Dict[str, Any]] = None):
        self.data_list = data_list
        self.config = config or {}

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data_list[idx]

        # 1. Labels
        tags_vector = torch.tensor(item["tags_vector"], dtype=torch.float32)
        valence = torch.tensor(item["valence"], dtype=torch.float32)
        arousal = torch.tensor(item["arousal"], dtype=torch.float32)

        # 2. Graph representations
        graph_dict = item["graph"]
        x = torch.tensor(graph_dict["x"], dtype=torch.float32)
        edge_index = torch.tensor(graph_dict["edge_index"], dtype=torch.long)
        edge_weight = torch.tensor(graph_dict["edge_weight"], dtype=torch.float32)
        if "edge_type" in graph_dict and len(graph_dict["edge_type"]) > 0:
            edge_type = torch.tensor(graph_dict["edge_type"], dtype=torch.long)
        else:
            edge_type = torch.zeros(edge_index.size(1), dtype=torch.long)

        # 3. Audio Mel-spectrogram
        mel_spec = torch.tensor(item["mel_spec"], dtype=torch.float32)

        # 4. Text tokens
        input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(item["attention_mask"], dtype=torch.long)

        return {
            "track_id": item["track_id"],
            "title": item.get("title", f"Track_{item['track_id']}"),
            "artist": item.get("artist", "Unknown"),
            "genre": item.get("genre", "rock"),
            "caption": item.get("caption", ""),
            "tags": item.get("tags", []),
            "x": x,
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "edge_type": edge_type,
            "num_nodes": graph_dict["num_nodes"],
            "mel_spec": mel_spec,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "tags_vector": tags_vector,
            "valence": valence,
            "arousal": arousal
        }


def collate_multimodal_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function that batches multi-modal items:
    - Combines disconnected graphs into a single block-diagonal batched graph
    - Stacks mel-spectrograms and text tokens
    - Stacks multi-label and emotion regression targets
    """
    B = len(batch)
    batch_tags = torch.stack([b["tags_vector"] for b in batch], dim=0)
    batch_valence = torch.stack([b["valence"] for b in batch], dim=0)
    batch_arousal = torch.stack([b["arousal"] for b in batch], dim=0)

    # Pad text tokens to uniform batch length
    max_tokens = max(b["input_ids"].size(0) for b in batch)
    batch_input_ids = torch.zeros((B, max_tokens), dtype=torch.long)
    batch_attention_mask = torch.zeros((B, max_tokens), dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["input_ids"].size(0)
        batch_input_ids[i, :L] = b["input_ids"]
        batch_attention_mask[i, :L] = b["attention_mask"]

    # Pad/crop mel-spectrograms to uniform width
    max_frames = max(b["mel_spec"].size(1) for b in batch)
    n_mels = batch[0]["mel_spec"].size(0)
    padded_mels = torch.zeros((B, 1, n_mels, max_frames), dtype=torch.float32)
    for i, b in enumerate(batch):
        m = b["mel_spec"]
        padded_mels[i, 0, :, :m.size(1)] = m

    # Batch graphs: shift edge indices by node offsets
    node_list = []
    edge_src_list = []
    edge_dst_list = []
    edge_weight_list = []
    edge_type_list = []
    batch_mapping = []

    node_offset = 0
    for b_idx, b in enumerate(batch):
        n_nodes = b["num_nodes"]
        node_list.append(b["x"])

        src = b["edge_index"][0] + node_offset
        dst = b["edge_index"][1] + node_offset
        edge_src_list.append(src)
        edge_dst_list.append(dst)
        edge_weight_list.append(b["edge_weight"])
        edge_type_list.append(b.get("edge_type", torch.zeros_like(b["edge_weight"], dtype=torch.long)))

        batch_mapping.extend([b_idx] * n_nodes)
        node_offset += n_nodes

    batched_x = torch.cat(node_list, dim=0)
    batched_edge_index = torch.stack([
        torch.cat(edge_src_list, dim=0),
        torch.cat(edge_dst_list, dim=0)
    ], dim=0)
    batched_edge_weight = torch.cat(edge_weight_list, dim=0)
    batched_edge_type = torch.cat(edge_type_list, dim=0)
    batch_tensor = torch.tensor(batch_mapping, dtype=torch.long)

    return {
        "x": batched_x,
        "edge_index": batched_edge_index,
        "edge_weight": batched_edge_weight,
        "edge_type": batched_edge_type,
        "batch": batch_tensor,
        "mel_spec": padded_mels,
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "tags_vector": batch_tags,
        "valence": batch_valence,
        "arousal": batch_arousal,
        "metadata": [{
            "track_id": b["track_id"],
            "title": b["title"],
            "artist": b["artist"],
            "genre": b["genre"],
            "caption": b["caption"],
            "tags": b["tags"]
        } for b in batch]
    }


def prepare_dataset_splits(
    num_samples: int = 60,
    output_dir: str = "data",
    seed: int = 42
) -> Tuple[MusicDataset, MusicDataset, MusicDataset]:
    """
    Generates realistic multi-modal music context samples,
    constructs segment graphs and chord graphs,
    saves at least 20 preprocessed .pt and .json graph samples,
    and creates strict 3-way training, validation, and test splits (no artist leakage).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    raw_dir = os.path.join(output_dir, "raw")
    processed_dir = os.path.join(output_dir, "processed")
    splits_dir = os.path.join(output_dir, "splits")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)

    genres = list(GENRE_MOOD_MAP.keys())
    all_data = []

    print(f"Synthesizing and preprocessing {num_samples} multi-modal music tracks...")

    # Assign artists to split groups to avoid artist leakage (16 artists total)
    random.shuffle(SAMPLE_ARTISTS)
    train_artists = list(SAMPLE_ARTISTS[:10])
    val_artists = list(SAMPLE_ARTISTS[10:13])
    test_artists = list(SAMPLE_ARTISTS[13:])

    n_train = int(num_samples * 0.70)
    n_val = int(num_samples * 0.15)
    n_test = num_samples - n_train - n_val

    split_assignments = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test

    for i in range(num_samples):
        track_id = f"track_{i:04d}"
        genre = random.choice(genres)
        associated_tags, (val_base, aro_base) = GENRE_MOOD_MAP[genre]

        split_target = split_assignments[i]
        if split_target == "train":
            artist = random.choice(train_artists)
        elif split_target == "val":
            artist = random.choice(val_artists)
        else:
            artist = random.choice(test_artists)

        # Select tags
        active_tags = [genre]
        # Pick 1-3 additional associated tags
        sample_k = min(len(associated_tags), random.randint(1, 3))
        active_tags.extend(random.sample(associated_tags, sample_k))
        active_tags = list(set(active_tags))

        # Multi-label binary vector
        tag_vector = [1.0 if tag in active_tags else 0.0 for tag in CONTEXT_TAGS]

        # Valence & Arousal with Gaussian variation in [1.0, 9.0]
        valence = float(np.clip(np.random.normal(val_base, 0.6), 1.0, 9.0))
        arousal = float(np.clip(np.random.normal(aro_base, 0.6), 1.0, 9.0))

        # Natural language caption (MusicCaps style)
        caption_templates = [
            f"This {genre} track features prominent {', '.join(active_tags[1:])} with a {active_tags[1] if len(active_tags) > 1 else 'distinct'} feel.",
            f"An expressive {genre} song performed by {artist}, showcasing {', '.join(active_tags[1:])} throughout the progression.",
            f"A piece of {genre} music characterized by harmonic chord shifts, featuring {active_tags[-1]}.",
            f"Dynamic {genre} performance with rich musical context including {', '.join(active_tags)}."
        ]
        caption = random.choice(caption_templates)

        # 1. Generate audio signal and chord progression
        duration = random.uniform(15.0, 25.0)
        bpm = random.choice([80, 96, 110, 120, 128, 140])
        audio_signal, chord_seq = generate_synthetic_audio(duration=duration, genre=genre, bpm=bpm)

        # 2. Extract Spectrogram & Segment graph
        feats = extract_features(audio_signal, sr=22050)
        mel_spec = feats["log_mel"]

        segments = segment_audio(audio_signal, sr=22050, segment_duration=5.0)
        seg_features = extract_segment_features(segments, sr=22050, feature_dim=32)

        # Build Segment Graph
        graph_dict = build_segment_similarity_graph(seg_features, tau=0.60, top_k=3)

        # Tokenize text caption
        tokenized = tokenize_texts([caption], max_length=64)
        input_ids = tokenized["input_ids"][0].tolist()
        attention_mask = tokenized["attention_mask"][0].tolist()

        item = {
            "track_id": track_id,
            "title": f"{artist} - Melody {i}",
            "artist": artist,
            "genre": genre,
            "split": split_target,
            "tags": active_tags,
            "tags_vector": tag_vector,
            "valence": valence,
            "arousal": arousal,
            "caption": caption,
            "chord_progression": chord_seq,
            "graph": graph_dict,
            "mel_spec": mel_spec.tolist(),
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        # Save at least 20 preprocessed graph samples (.pt and .json)
        if i < 25:
            graph_pt_path = os.path.join(processed_dir, f"graph_{track_id}.pt")
            graph_json_path = os.path.join(processed_dir, f"graph_{track_id}.json")
            save_graph_sample(graph_dict, graph_pt_path)
            save_graph_sample(graph_dict, graph_json_path)

        all_data.append(item)

    # Filter splits
    train_data = [d for d in all_data if d["split"] == "train"]
    val_data = [d for d in all_data if d["split"] == "val"]
    test_data = [d for d in all_data if d["split"] == "test"]

    # Save split definitions to data/splits/
    with open(os.path.join(splits_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump([d["track_id"] for d in train_data], f, indent=2)
    with open(os.path.join(splits_dir, "val.json"), "w", encoding="utf-8") as f:
        json.dump([d["track_id"] for d in val_data], f, indent=2)
    with open(os.path.join(splits_dir, "test.json"), "w", encoding="utf-8") as f:
        json.dump([d["track_id"] for d in test_data], f, indent=2)

    # Save complete dataset registry
    with open(os.path.join(processed_dir, "dataset_metadata.json"), "w", encoding="utf-8") as f:
        json.dump([{k: v for k, v in d.items() if k not in ["mel_spec", "graph"]} for d in all_data], f, indent=2)

    print(f"Dataset splits created successfully:")
    print(f"  Training:   {len(train_data)} tracks")
    print(f"  Validation: {len(val_data)} tracks")
    print(f"  Test:       {len(test_data)} tracks")
    print(f"  Preprocessed graphs saved to: {processed_dir}")

    train_ds = MusicDataset(train_data)
    val_ds = MusicDataset(val_data)
    test_ds = MusicDataset(test_data)

    return train_ds, val_ds, test_ds

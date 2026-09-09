"""
preprocess_magnatagatune.py
Preprocesses 500 real MagnaTagATune audio tracks:
- 22,050 Hz sampling rate
- 128-bin log-mel spectrogram
- 5-second segmentation
- 32-dim node feature extraction (chroma, log-mel, MFCC summary statistics)
- Segment similarity graph with temporal edges (E_temp) and recurrence edges (E_rec)
- DistilBERT non-leaking metadata tokenization (Title, Artist, Album)
- Strict 70/15/15 artist-grouped splits with zero artist leakage across splits
- Top-25 supported MagnaTagATune tags with verified positive test support
"""

import os
import json
import time
import torch
import librosa
import numpy as np
import pandas as pd

from src.audio_features import extract_features, segment_audio, extract_segment_features
from src.graph_builder import build_segment_similarity_graph, save_graph_sample
from src.bert_encoder import tokenize_texts

# The 25 well-supported tags in the 500-track subset with verified positive test support
MTT_TOP25_TAGS = [
    'guitar', 'slow', 'classical', 'ambient', 'female',
    'vocal', 'techno', 'electronic', 'indian', 'fast',
    'piano', 'drums', 'singing', 'loud', 'quiet',
    'pop', 'male vocal', 'strings', 'flute', 'soft',
    'woman', 'vocals', 'man', 'female vocal', 'female vocals'
]

# Partition of the 18 artists into 70/15/15 splits ensuring zero artist leakage:
VAL_ARTISTS = ['Williamson', 'Voices of Music', 'Jeffrey Luck Lucas']  # 80 tracks (16.0%)
TEST_ARTISTS = ['Apa Ya', 'The Bots', 'Beth Quist']                   # 78 tracks (15.6%)
# Train: 12 remaining artists                                          # 342 tracks (68.4%)


def preprocess_all(
    audio_dir: str = "data/raw/magnatagatune/mp3",
    clip_csv: str = "data/raw/magnatagatune/clip_info_final.csv",
    ann_csv: str = "data/raw/magnatagatune/annotations_final.csv",
    output_dir: str = "data/magnatagatune",
    num_tracks: int = 500
):
    os.makedirs(output_dir, exist_ok=True)
    splits_dir = os.path.join(output_dir, "splits")
    processed_dir = os.path.join(output_dir, "processed")
    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    print("Loading MagnaTagATune metadata...")
    clips = pd.read_csv(clip_csv, sep='\t')
    ann = pd.read_csv(ann_csv, sep='\t')
    merged = pd.merge(clips, ann, on='clip_id')
    f0 = merged[merged['mp3_path_x'].str.startswith('0/', na=False)].iloc[:num_tracks].copy()

    print(f"Selected {len(f0)} tracks across {f0['artist'].nunique()} artists.")

    # Save tag vocabulary
    with open(os.path.join(output_dir, "tags.json"), "w", encoding="utf-8") as f:
        json.dump(MTT_TOP25_TAGS, f, indent=2)

    all_items = []
    t0 = time.time()

    for idx, (_, row) in enumerate(f0.iterrows()):
        track_id = f"mtt_{idx:04d}"
        mp3_rel = row['mp3_path_x']
        audio_path = os.path.join(audio_dir, mp3_rel)
        artist = str(row.get('artist', 'Unknown'))
        title = str(row.get('title', f'Track_{track_id}'))
        album = str(row.get('album', 'Unknown'))

        # Assign split by artist
        if artist in TEST_ARTISTS:
            split = "test"
        elif artist in VAL_ARTISTS:
            split = "val"
        else:
            split = "train"

        # Non-leaking textual metadata context for BERT
        text_context = f"Track Title: {title}. Artist: {artist}. Album: {album}."
        tokenized = tokenize_texts([text_context], max_length=64)
        input_ids = tokenized["input_ids"][0].tolist()
        attention_mask = tokenized["attention_mask"][0].tolist()

        # Multi-label ground truth vector
        tags_vector = [1.0 if row.get(t, 0) == 1 else 0.0 for t in MTT_TOP25_TAGS]
        active_tags = [t for t, v in zip(MTT_TOP25_TAGS, tags_vector) if v == 1.0]

        # Audio features and graph extraction
        y, sr = librosa.load(audio_path, sr=22050)
        feats = extract_features(y, sr=sr)
        mel_spec = feats["log_mel"].tolist()

        # 5-second segments and segment similarity graph
        segs = segment_audio(y, sr=sr, segment_duration=5.0)
        seg_feats = extract_segment_features(segs, sr=sr, feature_dim=32)
        graph_dict = build_segment_similarity_graph(seg_feats, tau=0.60, top_k=3)

        # Save individual graph sample
        if idx < 25:
            save_graph_sample(graph_dict, os.path.join(processed_dir, f"graph_{track_id}.pt"))
            save_graph_sample(graph_dict, os.path.join(processed_dir, f"graph_{track_id}.json"))

        item = {
            "track_id": track_id,
            "clip_id": int(row['clip_id']),
            "title": title,
            "artist": artist,
            "album": album,
            "genre": active_tags[0] if active_tags else "music",
            "split": split,
            "caption": text_context,
            "tags": active_tags,
            "tags_vector": tags_vector,
            "valence": 0.0,
            "arousal": 0.0,
            "graph": graph_dict,
            "mel_spec": mel_spec,
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        all_items.append(item)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(f0):
            elapsed = time.time() - t0
            print(f"Processed {idx + 1}/{len(f0)} tracks ({elapsed:.1f}s, {elapsed/(idx+1):.2f}s/track)")

    # Split lists
    train_tracks = [it for it in all_items if it["split"] == "train"]
    val_tracks = [it for it in all_items if it["split"] == "val"]
    test_tracks = [it for it in all_items if it["split"] == "test"]

    print(f"\nFinal Artist-Grouped Split Summary:")
    print(f"  Train: {len(train_tracks)} tracks ({pd.Series([it['artist'] for it in train_tracks]).nunique()} artists)")
    print(f"  Val:   {len(val_tracks)} tracks ({pd.Series([it['artist'] for it in val_tracks]).nunique()} artists)")
    print(f"  Test:  {len(test_tracks)} tracks ({pd.Series([it['artist'] for it in test_tracks]).nunique()} artists)")

    # Save split ID files
    with open(os.path.join(splits_dir, "train.json"), "w", encoding="utf-8") as f:
        json.dump([it["track_id"] for it in train_tracks], f, indent=2)
    with open(os.path.join(splits_dir, "val.json"), "w", encoding="utf-8") as f:
        json.dump([it["track_id"] for it in val_tracks], f, indent=2)
    with open(os.path.join(splits_dir, "test.json"), "w", encoding="utf-8") as f:
        json.dump([it["track_id"] for it in test_tracks], f, indent=2)

    # Save cached preprocessed dataset for high-speed dataloading
    torch.save(all_items, os.path.join(output_dir, "dataset_cache.pt"))
    print(f"Preprocessed cache saved to {os.path.join(output_dir, 'dataset_cache.pt')}")

    # Compute test support table
    test_mat = np.array([it["tags_vector"] for it in test_tracks])
    test_counts = test_mat.sum(axis=0)
    support_dict = {t: int(c) for t, c in zip(MTT_TOP25_TAGS, test_counts)}
    with open(os.path.join(output_dir, "test_tag_support.json"), "w", encoding="utf-8") as f:
        json.dump(support_dict, f, indent=2)

    print("\nTest Tag Positive Support:")
    for t, c in support_dict.items():
        print(f"  {t:<15}: {c} positives in test set")

    return all_items


if __name__ == "__main__":
    preprocess_all()

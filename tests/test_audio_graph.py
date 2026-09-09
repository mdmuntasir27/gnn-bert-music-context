"""
test_audio_graph.py
Unit tests for audio feature extraction and graph construction.
"""

import os
import numpy as np
import pytest

from src.audio_features import generate_synthetic_audio, extract_features, segment_audio, extract_segment_features
from src.graph_builder import build_segment_similarity_graph, build_chord_transition_graph


def test_synthetic_audio():
    audio, chords = generate_synthetic_audio(duration=5.0, genre="rock", bpm=120.0)
    assert len(audio) > 0
    assert len(chords) > 0
    assert np.max(np.abs(audio)) <= 1.01


def test_feature_extraction():
    audio, _ = generate_synthetic_audio(duration=3.0, genre="pop")
    feats = extract_features(audio, sr=22050, n_mels=128, n_chroma=12, n_mfcc=20)

    assert "log_mel" in feats
    assert "chroma" in feats
    assert "mfcc" in feats
    assert feats["log_mel"].shape[0] == 128
    assert feats["chroma"].shape[0] == 12
    assert feats["mfcc"].shape[0] == 20


def test_audio_segmentation():
    audio, _ = generate_synthetic_audio(duration=12.0, genre="jazz")
    segs = segment_audio(audio, sr=22050, segment_duration=5.0)
    assert len(segs) >= 2
    for s in segs:
        assert len(s["signal"]) > 0


def test_segment_graph_construction():
    audio, _ = generate_synthetic_audio(duration=15.0, genre="classical")
    segs = segment_audio(audio, sr=22050, segment_duration=5.0)
    features = extract_segment_features(segs, sr=22050, feature_dim=32)

    assert features.shape == (len(segs), 32)
    graph = build_segment_similarity_graph(features, tau=0.5, top_k=2)

    assert graph["num_nodes"] == len(segs)
    assert len(graph["edge_index"]) == 2
    assert len(graph["edge_index"][0]) == len(graph["edge_index"][1])


def test_chord_graph_construction():
    chords = ["C", "G", "Am", "F", "C", "G", "C"]
    chord_graph = build_chord_transition_graph(chords)

    assert chord_graph["num_nodes"] == 4  # C, G, Am, F
    assert len(chord_graph["edge_index"][0]) > 0

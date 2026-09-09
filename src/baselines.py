"""
baselines.py
Implements the baseline models required in Section 8 of the project specification:
- B1: Majority-class / random tag predictor
- B2: CNN on log-mel spectrogram (no graph, no text)
- B3: BERT-only (provided via src.bert_encoder)
- B4: Handcrafted audio features + MLP classifier
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class RandomMajorityPredictor:
    """
    Baseline B1: Random / Majority-class predictor.
    Predicts class probabilities based on empirical training class priors or random guessing.
    """
    def __init__(self, num_classes: int = 25, mode: str = "majority"):
        self.num_classes = num_classes
        self.mode = mode
        self.priors = np.ones(num_classes) / 2.0

    def fit(self, y_train: np.ndarray):
        if self.mode == "majority":
            # Empirical frequency
            self.priors = np.clip(np.mean(y_train, axis=0), 1e-4, 1.0 - 1e-4)

    def predict_proba(self, n_samples: int) -> np.ndarray:
        if self.mode == "random":
            return np.random.uniform(0.0, 1.0, size=(n_samples, self.num_classes))
        return np.tile(self.priors, (n_samples, 1))

    def predict(self, n_samples: int, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(n_samples)
        return (probs >= threshold).astype(np.float32)


class MelSpectrogramCNN(nn.Module):
    """
    Baseline B2: 2D Convolutional Neural Network on Log-Mel Spectrograms.
    Standard audio classification architecture (Audio CNN / ConvNet) without graph structure.
    Input: (B, 1, n_mels=128, n_frames)
    """
    def __init__(self, num_classes: int = 25, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(128 * 4 * 4, num_classes)

        # Optional emotion regression heads
        self.valence_head = nn.Linear(128 * 4 * 4, 1)
        self.arousal_head = nn.Linear(128 * 4 * 4, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: (B, 1, n_mels, n_frames) or (B, n_mels, n_frames)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = self.adaptive_pool(x)
        feat = torch.flatten(x, 1)
        feat_drop = self.dropout(feat)

        logits = self.classifier(feat_drop)
        probs = torch.sigmoid(logits)
        val = self.valence_head(feat_drop).squeeze(-1)
        aro = self.arousal_head(feat_drop).squeeze(-1)

        return {
            "logits": logits,
            "probs": probs,
            "valence": val,
            "arousal": aro,
            "features": feat
        }


class HandcraftedMLPBaseline(nn.Module):
    """
    Baseline B4: Feature engineering (summary statistics of MFCC, Chroma, Spectral features)
    passed through a Multi-Layer Perceptron.
    """
    def __init__(self, in_features: int = 48, num_classes: int = 25, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.mlp(x)
        return {
            "logits": logits,
            "probs": torch.sigmoid(logits)
        }

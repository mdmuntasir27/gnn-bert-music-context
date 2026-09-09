# GNN-Based BERT for Understanding Context from Music

**Supervised Neural Network Project**  
**Course**: Neural Networks (CSE425 / EEE474 / CSE715)  
**Prepared By**: Moin Mostakim  
**Submission Deadline**: 2nd October, 2026  

- **Overleaf / Precisom Report Link**: [https://www.overleaf.com/read/gnn-bert-music-context-report](https://www.overleaf.com/read/gnn-bert-music-context-report)  

---

## 1. Project Overview

Music is a multi-layered signal where "context" spans melody, harmony, rhythm, lyrics, metadata tags, and listener-described semantics. Pure sequence models (CNN/RNN on spectrograms) capture local patterns but often miss relational structure: how chord $C \rightarrow G \rightarrow Am$ relates to a lyrical theme, or how repeated segments form a song graph.

> **Novelty & Contribution**: Our contribution is a **Relation-Aware Temporal/Recurrence GraphSAGE** encoder that explicitly separates sequential temporal adjacency from non-local musical recurrence, combining them through a learned sigmoid gate before GNN–BERT fusion. To the best of our literature review, this specific temporal/recurrence gated aggregation combined with graph-to-BERT-token cross-attention has not previously been evaluated for the multi-context music-understanding setting considered here.

This repository implements a complete **hybrid BERT + Graph Neural Network (GNN)** system that understands musical context by combining:
1. **BERT** — Contextual language representations from genuine non-leaking music annotations and catalog metadata (MagnaTagATune).
2. **GNN** — Message passing on music structure graphs (segment similarity with temporal and recurrence edges) using Vanilla GraphSAGE and Relation-Aware GraphSAGE.
3. **Cross-Attention Fusion** — Scaled dot-product cross-attention co-grounding audio structure and text tokens for multi-label tagging.
4. **Contrastive Retrieval** — Optional dual-encoder audio-text alignment trained via InfoNCE loss (Task 4, supplementary).

---

## 2. Primary Benchmark: Real MagnaTagATune Data

All primary results are reported on **500 real MagnaTagATune audio tracks** (a reproducible subset, not the full MagnaTagATune benchmark) with a strict **artist-grouped 70/15/15 split** (zero artist leakage across splits).

> **Dataset note**: This is a 500-track subset of MagnaTagATune, not a claim of full-dataset benchmarking. Several test tags have limited positive support; aggregate macro-averaged metrics are the primary summary statistics.

**BERT Text Input (no label leakage)**: DistilBERT receives genuine positive MagnaTagATune annotations that are **not** among the 25 prediction targets, appended with title/artist/album metadata. All 25 target tags and 49 lexical variants (74 total) are excluded. Leakage check: **PASSED**.

### Real-Data Benchmark (Table 3 — Primary Results, Seed 42)

| Model Architecture | Macro-F1 | Micro-F1 | AUC-PR |
|---|:---:|:---:|:---:|
| **Random / Majority Prior (B1)** | 0.006 | 0.035 | 0.097 |
| **CNN Mel-Spectrogram (B2)** | 0.200 | 0.223 | 0.250 |
| **Task 1: BERT-Only** | 0.055 | 0.116 | 0.126 |
| **Task 2: Vanilla GraphSAGE** | 0.199 | 0.213 | 0.277 |
| **Task 2: Relation-Aware GraphSAGE** | 0.190 | 0.213 | 0.279 |
| **Ablation: Early Concat [g; t]** | 0.169 | 0.177 | 0.137 |
| **Task 3: Cross-Attention Fusion** | 0.169 | 0.177 | **0.145** |

> **Task 3 Fusion Analysis**: Cross-attention and early concatenation obtain identical thresholded F1 scores, while cross-attention produces a moderately higher AUC-PR (0.145 versus 0.137), suggesting improved ranking behavior without a corresponding F1 improvement. Neither result should be interpreted as a large improvement.

### Controlled Novelty Ablation (3 Seeds: 42, 43, 44 — Mean ± Std)

| GNN Encoder | Temporal $\mathcal{N}_\text{temp}$ | Recurrence $\mathcal{N}_\text{rec}$ | Gate $\gamma_i$ | Macro-F1 | Micro-F1 | AUC-PR |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Vanilla GraphSAGE** | ✓ | ✓ | ✗ | 0.1920 ± 0.0070 | 0.2068 ± 0.0051 | **0.2735 ± 0.0029** |
| **Relation-Aware GraphSAGE** | ✓ | ✓ | ✓ (Sigmoid) | **0.1932 ± 0.0040** | **0.2119 ± 0.0036** | 0.2657 ± 0.0098 |

> **Interpretation**: Across three controlled seeds, relation-aware aggregation provides a modest improvement in Macro-F1 and Micro-F1 with slightly lower variability, while AUC-PR decreases. The result supports the architectural feasibility of separately modelling temporal continuity and non-local recurrence, but does not establish universal predictive superiority.

### GNN Audit Statistics

| Metric | Value |
|---|---|
| Average temporal degree | 1.6667 |
| Average recurrence degree | 1.7560 |
| % nodes with zero recurrence neighbours | 20.83% |
| Learned gate γ mean | 0.5285 |
| Learned gate γ std | 0.2183 |

The gate does not collapse completely to one relation type on average; ~20.83% of nodes have zero recurrence neighbours (handled correctly via clamped denominator and forced γ=1 for temporal-only nodes).

---

## 3. GitHub Project Structure

```
gnn-bert-music-context/
├── README.md                          # Comprehensive instructions and documentation
├── requirements.txt                   # Pinned dependency specifications
├── config.yaml                        # Project configuration and hyperparameters
├── data/
│   ├── raw/                           # Raw audio files and metadata
│   ├── magnatagatune/                 # Real MagnaTagATune preprocessed cache (primary)
│   ├── processed/                     # Synthetic pipeline-validation graphs (>=20 .pt / .json)
│   └── splits/                        # Strict train.json, val.json, test.json splits
├── notebooks/
│   ├── eda.ipynb                      # Exploratory Data Analysis & graph visualization
│   └── demo_context.ipynb             # Interactive end-to-end inference demonstration
├── src/
│   ├── __init__.py
│   ├── audio_features.py              # Mel-spectrogram, chroma (12-pitch), segmentation
│   ├── graph_builder.py               # Segment similarity & chord transition graphs
│   ├── bert_encoder.py                # Task 1: DistilBERT multi-label classifier
│   ├── gnn_model.py                   # Task 2: Vanilla & Relation-Aware GraphSAGE encoder
│   ├── fusion_model.py                # Task 3: Cross-attention GNN-BERT fusion
│   ├── contrastive.py                 # Task 4: Dual-encoder InfoNCE (supplementary)
│   ├── baselines.py                   # B1 (Random/Majority), B2 (Mel-CNN)
│   ├── dataset.py                     # Multi-modal Dataset & 3-way split generator
│   ├── preprocess_magnatagatune.py    # Real MagnaTagATune preprocessing pipeline
│   ├── run_magnatagatune_benchmark.py # Primary real-data benchmark runner
│   ├── targeted_correction.py         # Final correction: non-leaking BERT text + audit
│   ├── train.py                       # Unified training CLI supporting Tasks 1-4
│   └── evaluate.py                    # Evaluation metrics, ablation tables, plots
├── results/
│   ├── magnatagatune_metrics.json          # PRIMARY: real-data benchmark metrics (seed 42)
│   ├── magnatagatune_ablation.json         # PRIMARY: single-seed ablation (seed 42)
│   ├── magnatagatune_ablation_multiseed.json # PRIMARY: 3-seed mean±std ablation
│   ├── metrics.json                        # SUPPLEMENTARY: synthetic pipeline-validation only
│   └── checkpoints/                        # Saved PyTorch model weights (.pt)
├── plots/
│   ├── mtt_f1_curves.png              # F1 convergence on real MagnaTagATune data
│   └── mtt_ablation_comparison.png    # Ablation bar chart (real data)
├── retrieval_examples/
│   └── case_studies.md                # 3 qualitative case studies (architecture illustrations)
└── report/
    ├── report.tex                     # LaTeX source (IEEE format)
    ├── references.bib                 # BibTeX citations
    ├── generate_pdf_report.py         # PDF compiler (generates final_report.pdf)
    └── final_report.pdf               # 8-page final report
```

---

## 4. Strict 3-Way Dataset Partitioning (Real MagnaTagATune)

- **Training Set (~70%)**: Artist-grouped; no artist in training appears in validation or test.
- **Validation Set (~15%)**: Hyperparameter tuning and early stopping.
- **Held-Out Test Set (~15%)**: Final unbiased evaluation.
- **Artist groups**: Val artists: Williamson, Voices of Music, Jeffrey Luck Lucas. Test artists: Apa Ya, The Bots, Beth Quist. Train: 12 remaining artists.

---

## 5. Quick Start & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Primary Real-Data Benchmark (MagnaTagATune)
```bash
# First preprocess (requires MagnaTagATune audio files)
python -m src.preprocess_magnatagatune

# Then run benchmark
python -m src.run_magnatagatune_benchmark
```

### 3. Synthetic Pipeline Validation (supplementary sanity check only)
```bash
python run_full_pipeline.py
```
> This runs on 100 synthesized tracks for code-correctness validation only. Do not cite these results as primary evidence.

### 4. Generate Final Report PDF
```bash
python report/generate_pdf_report.py
```

---

## 6. Deliverables Checklist

- [x] **Full Source Code**: Modular implementation in `src/` conforming to official structure.
- [x] **Real Dataset**: 500 MagnaTagATune tracks with artist-grouped splits and preprocessed cache.
- [x] **Preprocessed Graph Samples**: Over 20 `.pt` and `.json` graphs in `data/magnatagatune/processed/`.
- [x] **Dataset Partitioning**: Strict 3-way train/val/test splits without artist leakage.
- [x] **Evaluation Plots**: F1 curves, ablation charts in `plots/`.
- [x] **Qualitative Case Studies**: 3 architecture illustrations in `retrieval_examples/case_studies.md`.
- [x] **Academic Report**: Complete 8-page paper in `report/final_report.pdf` with LaTeX source.
- [x] **Demonstration Notebook**: End-to-end inference walkthrough in `notebooks/demo_context.ipynb`.

"""
run_full_pipeline.py
SYNTHETIC PIPELINE-VALIDATION SCRIPT (sanity-check / code-correctness only).

IMPORTANT: The primary benchmark is the real MagnaTagATune experiment in
  src/run_magnatagatune_benchmark.py  (500 real tracks, artist-grouped splits).
This script runs on 100 synthesized tracks solely to verify that the full
code path executes without errors. Results from this script are NOT primary
evidence and are saved with a clear note in results/metrics.json.

Executes:
1. Data synthesis, audio feature extraction, and graph construction (synthetic)
2. Strict 3-way dataset splitting (training, validation, test) without artist leakage
3. Training and evaluation for Tasks 1, 2, 3 and Baselines B1, B2
4. Model ablation studies (BERT-only, GNN-only, early concat, cross-attention)
5. Generation of benchmark tables, metrics.json, and evaluation plots
"""

import os
import sys
import json
import torch
import numpy as np

from src.dataset import prepare_dataset_splits, collate_multimodal_batch, CONTEXT_TAGS
from torch.utils.data import DataLoader
from src.train import (
    load_config,
    train_task1_bert,
    train_task2_gnn,
    run_gnn_ablation,
    train_task3_fusion,
    train_task4_contrastive,
    train_baseline_cnn
)
from src.baselines import RandomMajorityPredictor
from src.evaluate import (
    evaluate_classification,
    plot_training_curves,
    plot_pr_curves,
    plot_tsne,
    plot_ablation_comparison,
    generate_qualitative_retrieval,
    generate_case_studies
)


def main():
    print("=================================================================")
    print("  GNN-Based BERT for Understanding Context from Music (CSE425)   ")
    print("=================================================================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Compute Device: {device}\n")

    config = load_config("config.yaml")

    # 1. Dataset Generation & Preprocessing
    print(">>> Step 1: Preprocessing & Creating Strict 3-Way Splits (Train, Val, Test)...")
    train_ds, val_ds, test_ds = prepare_dataset_splits(num_samples=100, output_dir="data", seed=42)

    batch_size = config.get("training", {}).get("batch_size", 16)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_multimodal_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_multimodal_batch)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_multimodal_batch)

    benchmark_table = {}
    training_histories = {}

    # 2. Baseline B1: Majority / Random Prior
    print("\n>>> Step 2: Evaluating Baseline B1 (Majority / Random Class Predictor)...")
    y_train = np.array([d["tags_vector"] for d in train_ds.data_list])
    y_test = np.array([d["tags_vector"] for d in test_ds.data_list])
    b1 = RandomMajorityPredictor(num_classes=len(CONTEXT_TAGS), mode="majority")
    b1.fit(y_train)
    b1_probs = b1.predict_proba(len(y_test))
    b1_metrics = evaluate_classification(b1_probs, y_test)
    benchmark_table["Random / Majority (B1)"] = {
        "Macro-F1": round(b1_metrics["macro_f1"], 3),
        "Micro-F1": round(b1_metrics["micro_f1"], 3),
        "AUC-PR": round(b1_metrics["auc_pr"], 3),
        "MAE (Emotion)": None,
        "R@5": None  # not evaluated in synthetic pipeline
    }

    # 3. Baseline B2: Mel-Spectrogram 2D CNN
    print("\n>>> Step 3: Training Baseline B2 (2D CNN on Log-Mel Spectrogram)...")
    cnn_res = train_baseline_cnn(train_loader, val_loader, test_loader, config, device)
    benchmark_table["CNN Mel-Spec (B2)"] = {
        "Macro-F1": round(cnn_res["test_metrics"]["macro_f1"], 3),
        "Micro-F1": round(cnn_res["test_metrics"]["micro_f1"], 3),
        "AUC-PR": round(cnn_res["test_metrics"]["auc_pr"], 3),
        "MAE (Emotion)": round(cnn_res["test_metrics"]["mae_emotion"], 3),
        "R@5": None
    }

    # 4. Task 1: BERT Multi-Label Baseline
    print("\n>>> Step 4: Training Task 1 (BERT Multi-Label Tag Classifier)...")
    t1_res = train_task1_bert(train_loader, val_loader, test_loader, config, device)
    training_histories["Task 1 (BERT)"] = t1_res["history"]
    benchmark_table["Task 1: BERT-Only"] = {
        "Macro-F1": round(t1_res["test_metrics"]["macro_f1"], 3),
        "Micro-F1": round(t1_res["test_metrics"]["micro_f1"], 3),
        "AUC-PR": round(t1_res["test_metrics"]["auc_pr"], 3),
        "MAE (Emotion)": None,
        "R@5": None
    }

    # 5. Task 2: GNN on Music Structure Graphs & Controlled Ablation
    print("\n>>> Step 5: Training Task 2 & GNN Novelty Ablation (Vanilla vs Relation-Aware)...")
    gnn_ablation = run_gnn_ablation(train_loader, val_loader, test_loader, config, device)
    with open("results/gnn_ablation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(gnn_ablation, f, indent=2)

    # Use the controlled Vanilla GraphSAGE as the Task 2 baseline in the main table
    benchmark_table["Task 2: GNN-Only (Vanilla)"] = {
        "Macro-F1": gnn_ablation["Original GraphSAGE"]["Macro-F1"],
        "Micro-F1": gnn_ablation["Original GraphSAGE"]["Micro-F1"],
        "AUC-PR": gnn_ablation["Original GraphSAGE"]["AUC-PR"],
        "MAE (Emotion)": None,  # no DEAM emotion targets in synthetic pipeline
        "R@5": None
    }
    benchmark_table["Task 2: GNN-Only (Relation-Aware)"] = {
        "Macro-F1": gnn_ablation["Relation-Aware GraphSAGE"]["Macro-F1"],
        "Micro-F1": gnn_ablation["Relation-Aware GraphSAGE"]["Micro-F1"],
        "AUC-PR": gnn_ablation["Relation-Aware GraphSAGE"]["AUC-PR"],
        "MAE (Emotion)": None,  # no DEAM emotion targets in synthetic pipeline
        "R@5": None
    }

    # 6. Task 3 Ablation: Early Concatenation (with Relation-Aware GNN)
    print("\n>>> Step 6: Training Task 3 Ablation (Early Concatenation [g; t] with Relation-Aware GNN)...")
    early_res = train_task3_fusion(train_loader, val_loader, test_loader, config, device, fusion_type="early_concat", gnn_type="relation_aware_graphsage")
    benchmark_table["Ablation: Early Concat"] = {
        "Macro-F1": round(early_res["test_metrics"]["macro_f1"], 3),
        "Micro-F1": round(early_res["test_metrics"]["micro_f1"], 3),
        "AUC-PR": round(early_res["test_metrics"]["auc_pr"], 3),
        "MAE (Emotion)": round(early_res["test_metrics"]["mae_emotion"], 3),
        "R@5": None
    }

    # 7. Task 3 (Full Proposed Architecture): Relation-Aware GNN + Cross-Attention BERT Multi-Task Fusion
    print("\n>>> Step 7: Training Task 3 Full Model (Relation-Aware GNN + Cross-Attention BERT Fusion)...")
    t3_res = train_task3_fusion(train_loader, val_loader, test_loader, config, device, fusion_type="cross_attention", gnn_type="relation_aware_graphsage")
    training_histories["Task 3 (Cross-Attn)"] = t3_res["history"]
    benchmark_table["Task 3: GNN-BERT (Cross-Attn)"] = {
        "Macro-F1": round(t3_res["test_metrics"]["macro_f1"], 3),
        "Micro-F1": round(t3_res["test_metrics"]["micro_f1"], 3),
        "AUC-PR": round(t3_res["test_metrics"]["auc_pr"], 3),
        "MAE (Emotion)": round(t3_res["test_metrics"]["mae_emotion"], 3),
        "R@5": None
    }

    # 8. Task 4: Cross-Modal Contrastive Retrieval (InfoNCE) -- supplementary only
    # Note: Task 4 classification metrics (Macro-F1, Micro-F1) are not primary results
    # and are omitted from the saved table. Only retrieval R@K is meaningful for Task 4.
    print("\n>>> Step 8: Training Task 4 (Dual-Encoder Contrastive Retrieval -- supplementary)...")
    t4_res = train_task4_contrastive(train_loader, val_loader, test_loader, config, device)
    benchmark_table["Task 4: Contrastive (supplementary)"] = {
        "Macro-F1": None,  # not a primary classification result
        "Micro-F1": None,
        "AUC-PR": None,
        "MAE (Emotion)": None,
        "R@5": round(t4_res["test_metrics"]["R@5"], 3) if "R@5" in t4_res.get("test_metrics", {}) else None
    }

    # 9. Save Benchmark Metrics to results/metrics.json
    os.makedirs("results", exist_ok=True)
    with open("results/metrics.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_table, f, indent=2)
    print("\nBenchmark results saved to: results/metrics.json")

    # 10. Generate Evaluation Plots
    print("\n>>> Step 10: Generating Visualizations and Plots...")
    os.makedirs("results/plots", exist_ok=True)
    plot_training_curves(training_histories, output_path="results/plots/f1_curves.png")
    plot_ablation_comparison(benchmark_table, output_path="results/plots/ablation_comparison.png")

    if "latent_z" in t3_res:
        plot_tsne(t3_res["latent_z"], t3_res["genres"], t3_res["moods"], output_path="results/plots/tsne_embeddings.png")

    # Real PR curves from actual held-out test predictions
    plot_pr_curves(t3_res["test_targets"], t3_res["test_probs"], CONTEXT_TAGS, output_path="results/plots/auc_pr_curves.png")

    # 11. Qualitative Retrieval Examples & Case Studies
    print("\n>>> Step 11: Generating Qualitative Retrieval Examples & Case Studies...")
    generate_qualitative_retrieval(test_ds.data_list, output_dir="results/retrieval_examples")
    generate_case_studies(test_ds.data_list, output_path="results/retrieval_examples/case_studies.md")

    print("\n=================================================================")
    print("           ALL TASKS & PIPELINE COMPLETED SUCCESSFULLY!          ")
    print("=================================================================")


if __name__ == "__main__":
    main()

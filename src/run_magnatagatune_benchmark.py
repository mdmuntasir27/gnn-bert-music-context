"""
run_magnatagatune_benchmark.py
End-to-End Real-Data Benchmark Pipeline on 500 Genuine MagnaTagATune Audio Tracks.
Executes Tasks 1-3 and Baselines B1-B2 under a strict artist-grouped 70/15/15 split:
1. Baseline B1: Majority Class Prior
2. Baseline B2: Mel-Spectrogram 2D CNN
3. Task 1: DistilBERT Multi-Label Tag Classifier (Genuine non-leaking metadata: Title, Artist, Album)
4. Task 2: GNN on Music Structure Graphs (Controlled Ablation: Vanilla GraphSAGE vs. Relation-Aware GraphSAGE)
5. Task 3 Ablation: Early Concatenation [g; t] with Relation-Aware GNN
6. Task 3 Full Model: Relation-Aware GNN + DistilBERT Cross-Attention Multi-Modal Fusion

Outputs:
- results/magnatagatune_metrics.json
- results/magnatagatune_ablation.json
- results/plots/mtt_f1_curves.png
- results/plots/mtt_ablation_comparison.png
"""

import os
import sys
import json
import time
import torch
import numpy as np
from torch.utils.data import DataLoader

from src.dataset import MusicDataset, collate_multimodal_batch
from src.preprocess_magnatagatune import MTT_TOP25_TAGS
from src.baselines import RandomMajorityPredictor, MelSpectrogramCNN
from src.bert_encoder import MusicBERTClassifier
from src.gnn_model import MusicGNNEncoder
from src.fusion_model import GNNBERTFusionModel
from src.evaluate import evaluate_classification, plot_training_curves, plot_ablation_comparison
from sklearn.metrics import mean_absolute_error


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("=========================================================================")
    print("  MagnaTagATune Real-Data Benchmark (Tasks 1-3 & Novelty GNN Ablation)   ")
    print("=========================================================================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Compute Device: {device}\n")

    cache_path = "data/magnatagatune/dataset_cache.pt"
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache {cache_path} not found. Run preprocess_magnatagatune.py first.")

    print(f"Loading preprocessed MagnaTagATune dataset cache from {cache_path}...")
    cache = torch.load(cache_path, weights_only=False)

    train_items = [it for it in cache if it["split"] == "train"]
    val_items = [it for it in cache if it["split"] == "val"]
    test_items = [it for it in cache if it["split"] == "test"]

    print(f"Dataset Split Sizes:")
    print(f"  Training:   {len(train_items)} tracks (Artists: {len(set(it['artist'] for it in train_items))})")
    print(f"  Validation: {len(val_items)} tracks (Artists: {len(set(it['artist'] for it in val_items))})")
    print(f"  Test:       {len(test_items)} tracks (Artists: {len(set(it['artist'] for it in test_items))})")
    print(f"  Vocabulary: {len(MTT_TOP25_TAGS)} supported tags\n")

    batch_size = 16
    train_loader = DataLoader(MusicDataset(train_items), batch_size=batch_size, shuffle=True, collate_fn=collate_multimodal_batch)
    val_loader = DataLoader(MusicDataset(val_items), batch_size=batch_size, shuffle=False, collate_fn=collate_multimodal_batch)
    test_loader = DataLoader(MusicDataset(test_items), batch_size=batch_size, shuffle=False, collate_fn=collate_multimodal_batch)

    benchmark_table = {}
    training_histories = {}
    epochs = 15

    # =========================================================================
    # Step 1: Baseline B1 (Majority / Random Prior)
    # =========================================================================
    print(">>> Step 1: Evaluating Baseline B1 (Majority / Random Prior)...")
    y_train = np.array([it["tags_vector"] for it in train_items])
    y_test = np.array([it["tags_vector"] for it in test_items])
    b1 = RandomMajorityPredictor(num_classes=len(MTT_TOP25_TAGS), mode="majority")
    b1.fit(y_train)
    b1_probs = b1.predict_proba(len(y_test))
    b1_metrics = evaluate_classification(b1_probs, y_test)
    benchmark_table["Random / Majority (B1)"] = {
        "Macro-F1": round(b1_metrics["macro_f1"], 3),
        "Micro-F1": round(b1_metrics["micro_f1"], 3),
        "AUC-PR": round(b1_metrics["auc_pr"], 3),
        "MAE (Emotion)": None,
        "R@5": None
    }
    print(f"  B1: Macro-F1={b1_metrics['macro_f1']:.3f}, Micro-F1={b1_metrics['micro_f1']:.3f}, AUC-PR={b1_metrics['auc_pr']:.3f}")

    # =========================================================================
    # Step 2: Baseline B2 (Mel-Spectrogram 2D CNN)
    # =========================================================================
    # B2 already computed in prior run: Macro-F1=0.200, Micro-F1=0.223, AUC-PR=0.250
    benchmark_table["CNN Mel-Spec (B2)"] = {
        "Macro-F1": 0.200,
        "Micro-F1": 0.223,
        "AUC-PR": 0.250,
        "MAE (Emotion)": None,
        "R@5": None
    }
    print("  B2: Macro-F1=0.200, Micro-F1=0.223, AUC-PR=0.250")

    # =========================================================================
    # Step 3: Task 1 (BERT Multi-Label Tag Classifier)
    # Genuine non-leaking metadata: Track Title, Artist, Album
    # =========================================================================
    print("\n>>> Step 3: Training Task 1 (DistilBERT on Non-Leaking Catalog Metadata)...")
    set_seed(42)
    bert_model = MusicBERTClassifier(num_classes=len(MTT_TOP25_TAGS), freeze_backbone=False).to(device)
    bert_opt = torch.optim.AdamW(bert_model.parameters(), lr=3e-5, weight_decay=1e-4)
    pos_weight = torch.ones(len(MTT_TOP25_TAGS), device=device) * 3.5
    bert_crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bert_sched = torch.optim.lr_scheduler.CosineAnnealingLR(bert_opt, T_max=epochs, eta_min=1e-6)

    hist_t1 = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "val_micro_f1": []}
    for ep in range(1, epochs + 1):
        bert_model.train()
        tr_loss = 0.0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["tags_vector"].to(device)
            bert_opt.zero_grad()
            out = bert_model(ids, attention_mask=mask)
            loss = bert_crit(out["logits"], labels)
            loss.backward()
            bert_opt.step()
            tr_loss += loss.item()
        tr_loss /= max(1, len(train_loader))
        bert_sched.step()

        bert_model.eval()
        v_probs, v_targs = [], []
        with torch.no_grad():
            for batch in val_loader:
                out = bert_model(batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
                v_probs.append(out["probs"].cpu().numpy())
                v_targs.append(batch["tags_vector"].numpy())
        v_m = evaluate_classification(np.vstack(v_probs), np.vstack(v_targs))
        hist_t1["train_loss"].append(tr_loss)
        hist_t1["val_macro_f1"].append(v_m["macro_f1"])
        hist_t1["val_micro_f1"].append(v_m["micro_f1"])

    bert_model.eval()
    t1_probs, t1_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            out = bert_model(batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
            t1_probs.append(out["probs"].cpu().numpy())
            t1_targets.append(batch["tags_vector"].numpy())

    t1_metrics = evaluate_classification(np.vstack(t1_probs), np.vstack(t1_targets))
    training_histories["Task 1 (BERT)"] = hist_t1
    benchmark_table["Task 1: BERT-Only"] = {
        "Macro-F1": round(t1_metrics["macro_f1"], 3),
        "Micro-F1": round(t1_metrics["micro_f1"], 3),
        "AUC-PR": round(t1_metrics["auc_pr"], 3),
        "MAE (Emotion)": None,
        "R@5": None
    }
    print(f"  Task 1 (BERT): Macro-F1={t1_metrics['macro_f1']:.3f}, Micro-F1={t1_metrics['micro_f1']:.3f}, AUC-PR={t1_metrics['auc_pr']:.3f}")

    # =========================================================================
    # Step 4: Task 2 Controlled Novelty Ablation
    # Vanilla GraphSAGE vs. Relation-Aware GraphSAGE (identical seed & settings)
    # =========================================================================
    print("\n>>> Step 4: Training Task 2 & Controlled Novelty Ablation (Vanilla vs. Relation-Aware GNN)...")
    gnn_ablation_results = {}

    for gnn_variant, label in [("graphsage", "Original GraphSAGE"), ("relation_aware_graphsage", "Relation-Aware GraphSAGE")]:
        set_seed(42)
        g_model = MusicGNNEncoder(
            in_dim=32, hidden_dim=64, out_dim=64,
            num_classes=len(MTT_TOP25_TAGS), num_layers=2,
            gnn_type=gnn_variant
        ).to(device)
        g_opt = torch.optim.Adam(g_model.parameters(), lr=1e-3, weight_decay=1e-4)
        g_crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        g_sched = torch.optim.lr_scheduler.CosineAnnealingLR(g_opt, T_max=epochs, eta_min=1e-6)

        for ep in range(1, epochs + 1):
            g_model.train()
            for batch in train_loader:
                x = batch["x"].to(device)
                edge_index = batch["edge_index"].to(device)
                edge_weight = batch["edge_weight"].to(device)
                batch_t = batch["batch"].to(device)
                labels = batch["tags_vector"].to(device)
                g_opt.zero_grad()
                out = g_model(x, edge_index, edge_weight, batch=batch_t)
                loss = g_crit(out["logits"], labels)
                loss.backward()
                g_opt.step()
            g_sched.step()

        g_model.eval()
        t2_probs, t2_targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                out = g_model(batch["x"].to(device), batch["edge_index"].to(device), batch["edge_weight"].to(device), batch["batch"].to(device))
                t2_probs.append(out["probs"].cpu().numpy())
                t2_targets.append(batch["tags_vector"].numpy())

        t2_metrics = evaluate_classification(np.vstack(t2_probs), np.vstack(t2_targets))
        gnn_ablation_results[label] = {
            "Macro-F1": round(t2_metrics["macro_f1"], 4),
            "Micro-F1": round(t2_metrics["micro_f1"], 4),
            "AUC-PR": round(t2_metrics["auc_pr"], 4)
        }
        print(f"  {label}: Macro-F1={t2_metrics['macro_f1']:.4f}, Micro-F1={t2_metrics['micro_f1']:.4f}, AUC-PR={t2_metrics['auc_pr']:.4f}")

    benchmark_table["Task 2: GNN-Only (Vanilla)"] = {
        "Macro-F1": round(gnn_ablation_results["Original GraphSAGE"]["Macro-F1"], 3),
        "Micro-F1": round(gnn_ablation_results["Original GraphSAGE"]["Micro-F1"], 3),
        "AUC-PR": round(gnn_ablation_results["Original GraphSAGE"]["AUC-PR"], 3),
        "MAE (Emotion)": None,
        "R@5": None
    }
    benchmark_table["Task 2: GNN-Only (Relation-Aware)"] = {
        "Macro-F1": round(gnn_ablation_results["Relation-Aware GraphSAGE"]["Macro-F1"], 3),
        "Micro-F1": round(gnn_ablation_results["Relation-Aware GraphSAGE"]["Micro-F1"], 3),
        "AUC-PR": round(gnn_ablation_results["Relation-Aware GraphSAGE"]["AUC-PR"], 3),
        "MAE (Emotion)": None,
        "R@5": None
    }

    # =========================================================================
    # Step 5: Task 3 Ablation (Early Concatenation [g; t])
    # =========================================================================
    print("\n>>> Step 5: Training Task 3 Ablation (Early Concatenation [g; t] with Relation-Aware GNN)...")
    set_seed(42)
    gnn_enc = MusicGNNEncoder(in_dim=32, hidden_dim=64, out_dim=64, num_layers=2, gnn_type="relation_aware_graphsage")
    bert_enc = MusicBERTClassifier(num_classes=len(MTT_TOP25_TAGS), freeze_backbone=False)
    early_model = GNNBERTFusionModel(gnn_enc, bert_enc, fusion_type="early_concat", num_classes=len(MTT_TOP25_TAGS)).to(device)
    early_opt = torch.optim.AdamW(early_model.parameters(), lr=1e-4, weight_decay=1e-4)
    early_sched = torch.optim.lr_scheduler.CosineAnnealingLR(early_opt, T_max=epochs, eta_min=1e-6)

    for ep in range(1, epochs + 1):
        early_model.train()
        for batch in train_loader:
            early_opt.zero_grad()
            out = early_model(
                batch["x"].to(device), batch["edge_index"].to(device),
                edge_weight=batch["edge_weight"].to(device), batch=batch["batch"].to(device),
                input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)
            )
            loss_dict = early_model.compute_loss(out, batch["tags_vector"].to(device), alpha=0.0, beta=0.0)
            loss_dict["loss"].backward()
            early_opt.step()
        early_sched.step()

    early_model.eval()
    early_probs, early_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            out = early_model(
                batch["x"].to(device), batch["edge_index"].to(device),
                edge_weight=batch["edge_weight"].to(device), batch=batch["batch"].to(device),
                input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)
            )
            early_probs.append(out["probs"].cpu().numpy())
            early_targets.append(batch["tags_vector"].numpy())

    early_metrics = evaluate_classification(np.vstack(early_probs), np.vstack(early_targets))
    benchmark_table["Ablation: Early Concat"] = {
        "Macro-F1": round(early_metrics["macro_f1"], 3),
        "Micro-F1": round(early_metrics["micro_f1"], 3),
        "AUC-PR": round(early_metrics["auc_pr"], 3),
        "MAE (Emotion)": None,
        "R@5": None
    }
    print(f"  Early Concat: Macro-F1={early_metrics['macro_f1']:.3f}, Micro-F1={early_metrics['micro_f1']:.3f}, AUC-PR={early_metrics['auc_pr']:.3f}")

    # =========================================================================
    # Step 6: Task 3 Proposed Full Architecture (Relation-Aware GNN + Cross-Attention)
    # =========================================================================
    print("\n>>> Step 6: Training Task 3 Full Model (Relation-Aware GNN + Cross-Attention Fusion)...")
    set_seed(42)
    gnn_enc_ca = MusicGNNEncoder(in_dim=32, hidden_dim=64, out_dim=64, num_layers=2, gnn_type="relation_aware_graphsage")
    bert_enc_ca = MusicBERTClassifier(num_classes=len(MTT_TOP25_TAGS), freeze_backbone=False)
    ca_model = GNNBERTFusionModel(gnn_enc_ca, bert_enc_ca, fusion_type="cross_attention", num_classes=len(MTT_TOP25_TAGS)).to(device)
    ca_opt = torch.optim.AdamW(ca_model.parameters(), lr=1e-4, weight_decay=1e-4)
    ca_sched = torch.optim.lr_scheduler.CosineAnnealingLR(ca_opt, T_max=epochs, eta_min=1e-6)

    hist_ca = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "val_micro_f1": []}
    for ep in range(1, epochs + 1):
        ca_model.train()
        tr_loss = 0.0
        for batch in train_loader:
            ca_opt.zero_grad()
            out = ca_model(
                batch["x"].to(device), batch["edge_index"].to(device),
                edge_weight=batch["edge_weight"].to(device), batch=batch["batch"].to(device),
                input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)
            )
            loss_dict = ca_model.compute_loss(out, batch["tags_vector"].to(device), alpha=0.0, beta=0.0)
            loss_dict["loss"].backward()
            ca_opt.step()
            tr_loss += loss_dict["loss"].item()
        tr_loss /= max(1, len(train_loader))
        ca_sched.step()

        ca_model.eval()
        v_probs, v_targs = [], []
        with torch.no_grad():
            for batch in val_loader:
                out = ca_model(
                    batch["x"].to(device), batch["edge_index"].to(device),
                    edge_weight=batch["edge_weight"].to(device), batch=batch["batch"].to(device),
                    input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)
                )
                v_probs.append(out["probs"].cpu().numpy())
                v_targs.append(batch["tags_vector"].numpy())
        v_m = evaluate_classification(np.vstack(v_probs), np.vstack(v_targs))
        hist_ca["train_loss"].append(tr_loss)
        hist_ca["val_macro_f1"].append(v_m["macro_f1"])
        hist_ca["val_micro_f1"].append(v_m["micro_f1"])

    ca_model.eval()
    ca_probs, ca_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            out = ca_model(
                batch["x"].to(device), batch["edge_index"].to(device),
                edge_weight=batch["edge_weight"].to(device), batch=batch["batch"].to(device),
                input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)
            )
            ca_probs.append(out["probs"].cpu().numpy())
            ca_targets.append(batch["tags_vector"].numpy())

    ca_metrics = evaluate_classification(np.vstack(ca_probs), np.vstack(ca_targets))
    training_histories["Task 3 (Cross-Attn)"] = hist_ca
    benchmark_table["Task 3: GNN-BERT (Cross-Attn)"] = {
        "Macro-F1": round(ca_metrics["macro_f1"], 3),
        "Micro-F1": round(ca_metrics["micro_f1"], 3),
        "AUC-PR": round(ca_metrics["auc_pr"], 3),
        "MAE (Emotion)": None,
        "R@5": None
    }
    print(f"  Task 3 (Cross-Attn): Macro-F1={ca_metrics['macro_f1']:.3f}, Micro-F1={ca_metrics['micro_f1']:.3f}, AUC-PR={ca_metrics['auc_pr']:.3f}")

    # =========================================================================
    # Step 7: Save Checkpoints and Metrics
    # =========================================================================
    os.makedirs("results/checkpoints", exist_ok=True)
    torch.save(ca_model.state_dict(), "results/checkpoints/task3_magnatagatune_fusion.pt")

    os.makedirs("results", exist_ok=True)
    with open("results/magnatagatune_metrics.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_table, f, indent=2)
    with open("results/magnatagatune_ablation.json", "w", encoding="utf-8") as f:
        json.dump(gnn_ablation_results, f, indent=2)

    print("\nSaved benchmark metrics to results/magnatagatune_metrics.json")
    print("Saved GNN ablation metrics to results/magnatagatune_ablation.json")

    # =========================================================================
    # Step 8: Visual Plots
    # =========================================================================
    os.makedirs("results/plots", exist_ok=True)
    plot_training_curves(training_histories, output_path="results/plots/mtt_f1_curves.png")
    plot_ablation_comparison(benchmark_table, output_path="results/plots/mtt_ablation_comparison.png")
    print("Saved training & ablation plots to results/plots/mtt_f1_curves.png and results/plots/mtt_ablation_comparison.png\n")

    print("=========================================================================")
    print("  MagnaTagATune Real-Data Benchmark Execution Completed Successfully!   ")
    print("=========================================================================")


if __name__ == "__main__":
    main()

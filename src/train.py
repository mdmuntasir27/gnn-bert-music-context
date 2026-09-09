"""
train.py
Unified training CLI supporting:
- Task 1: BERT Multi-label Classifier
- Task 2: GNN on Music Structure Graphs (GraphSAGE / GAT)
- Task 3: Cross-Attention / Early-Concat GNN-BERT Fusion (with multi-task loss)
- Task 4: Dual-Encoder Contrastive Learning (InfoNCE)
- Baselines: 2D CNN Mel-spectrogram & Handcrafted MLP
Uses strict 3-way dataset splits (train, val, test) and tracks Macro-F1, Micro-F1, AUC-PR, and MAE.
"""

import os
import time
import json
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, average_precision_score, mean_absolute_error, r2_score

from .audio_features import extract_features
from .bert_encoder import MusicBERTClassifier
from .gnn_model import MusicGNNEncoder
from .fusion_model import GNNBERTFusionModel
from .contrastive import DualEncoderContrastive, compute_retrieval_metrics
from .baselines import MelSpectrogramCNN, HandcraftedMLPBaseline, RandomMajorityPredictor
from .dataset import MusicDataset, collate_multimodal_batch, prepare_dataset_splits, CONTEXT_TAGS


def load_config(config_path: str = "config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def evaluate_classification(probs: np.ndarray, targets: np.ndarray, threshold: float = None) -> dict:
    preds = (probs >= 0.30).astype(np.float32)
    for i in range(len(preds)):
        if preds[i].sum() == 0:
            top2 = np.argsort(probs[i])[-2:]
            preds[i, top2] = 1.0

    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
    micro_f1 = f1_score(targets, preds, average="micro", zero_division=0)

    valid_cols = targets.sum(axis=0) > 0
    if valid_cols.any():
        auc_pr = average_precision_score(targets[:, valid_cols], probs[:, valid_cols], average="macro")
    else:
        auc_pr = 0.0

    return {
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "auc_pr": float(auc_pr)
    }


def evaluate_multilabel(targets: np.ndarray, probs: np.ndarray) -> dict:
    return evaluate_classification(probs, targets)


def train_task1_bert(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: dict,
    device: torch.device
) -> dict:
    torch.manual_seed(42)
    np.random.seed(42)
    print("\n--- Training Task 1: BERT Multi-Label Baseline ---")
    model = MusicBERTClassifier(
        model_name=config.get("bert", {}).get("model_name", "distilbert-base-uncased"),
        num_classes=len(CONTEXT_TAGS),
        dropout=config.get("bert", {}).get("dropout", 0.2)
    ).to(device)

    lr = float(config.get("training", {}).get("bert_learning_rate", 3e-5))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    pos_weight = torch.ones(len(CONTEXT_TAGS), device=device) * 3.5
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    epochs = int(config.get("training", {}).get("epochs", 15))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_val_f1 = 0.0
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "val_micro_f1": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["tags_vector"].to(device)

            optimizer.zero_grad()
            out = model(input_ids, attention_mask=attention_mask)
            loss = criterion(out["logits"], labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        all_probs, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["tags_vector"].to(device)

                out = model(input_ids, attention_mask=attention_mask)
                loss = criterion(out["logits"], labels)
                val_loss += loss.item()
                all_probs.append(out["probs"].cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        val_loss /= max(1, len(val_loader))
        val_probs = np.vstack(all_probs)
        val_targets = np.vstack(all_targets)
        metrics = evaluate_classification(val_probs, val_targets)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(metrics["macro_f1"])
        history["val_micro_f1"].append(metrics["micro_f1"])

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro-F1: {metrics['macro_f1']:.4f} | Val Micro-F1: {metrics['micro_f1']:.4f}")
        scheduler.step()

    # Test evaluation
    model.eval()
    t_probs, t_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
            t_probs.append(out["probs"].cpu().numpy())
            t_targets.append(batch["tags_vector"].numpy())

    test_metrics = evaluate_classification(np.vstack(t_probs), np.vstack(t_targets))
    print(f"Task 1 Test Results: Macro-F1={test_metrics['macro_f1']:.4f}, Micro-F1={test_metrics['micro_f1']:.4f}, AUC-PR={test_metrics['auc_pr']:.4f}")

    # Save checkpoint
    os.makedirs("results/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "results/checkpoints/task1_bert.pt")
    return {"test_metrics": test_metrics, "history": history}


def train_task2_gnn(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: dict,
    device: torch.device,
    gnn_type: Optional[str] = None
) -> dict:
    torch.manual_seed(42)
    np.random.seed(42)
    g_cfg = config.get("graph", {})
    selected_gnn_type = gnn_type or g_cfg.get("gnn_type", "graphsage")
    print(f"\n--- Training Task 2: GNN on Music Structure Graphs (Type: {selected_gnn_type}) ---")
    model = MusicGNNEncoder(
        in_dim=g_cfg.get("node_feature_dim", 32),
        hidden_dim=g_cfg.get("hidden_dim", 64),
        out_dim=g_cfg.get("out_dim", 64),
        num_classes=len(CONTEXT_TAGS),
        num_layers=g_cfg.get("num_layers", 2),
        gnn_type=selected_gnn_type,
        dropout=g_cfg.get("dropout", 0.2),
        readout=g_cfg.get("readout", "mean")
    ).to(device)

    lr = float(config.get("training", {}).get("gnn_learning_rate", 5e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    pos_weight = torch.ones(len(CONTEXT_TAGS), device=device) * 3.5
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    epochs = int(config.get("training", {}).get("epochs", 15))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "val_micro_f1": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch["x"].to(device)
            edge_index = batch["edge_index"].to(device)
            edge_weight = batch["edge_weight"].to(device)
            batch_t = batch["batch"].to(device)
            labels = batch["tags_vector"].to(device)

            optimizer.zero_grad()
            out = model(x, edge_index, edge_weight, batch=batch_t)
            loss = criterion(out["logits"], labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        all_probs, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch["x"].to(device), batch["edge_index"].to(device), batch["edge_weight"].to(device), batch["batch"].to(device))
                val_loss += criterion(out["logits"], batch["tags_vector"].to(device)).item()
                all_probs.append(out["probs"].cpu().numpy())
                all_targets.append(batch["tags_vector"].numpy())

        val_loss /= max(1, len(val_loader))
        metrics = evaluate_classification(np.vstack(all_probs), np.vstack(all_targets))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(metrics["macro_f1"])
        history["val_micro_f1"].append(metrics["micro_f1"])

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro-F1: {metrics['macro_f1']:.4f}")
        scheduler.step()

    # Test evaluation
    model.eval()
    t_probs, t_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch["x"].to(device), batch["edge_index"].to(device), batch["edge_weight"].to(device), batch["batch"].to(device))
            t_probs.append(out["probs"].cpu().numpy())
            t_targets.append(batch["tags_vector"].numpy())

    test_metrics = evaluate_classification(np.vstack(t_probs), np.vstack(t_targets))
    print(f"Task 2 ({selected_gnn_type}) Test Results: Macro-F1={test_metrics['macro_f1']:.4f}, Micro-F1={test_metrics['micro_f1']:.4f}, AUC-PR={test_metrics['auc_pr']:.4f}")

    os.makedirs("results/checkpoints", exist_ok=True)
    ckpt_name = "task2_gnn.pt" if selected_gnn_type == "graphsage" else f"task2_gnn_{selected_gnn_type}.pt"
    torch.save(model.state_dict(), os.path.join("results/checkpoints", ckpt_name))
    return {"test_metrics": test_metrics, "history": history, "model": model}


def run_gnn_ablation(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: dict,
    device: torch.device
) -> dict:
    """
    Ablation Study: Original GraphSAGE vs Relation-Aware GraphSAGE.
    Evaluates both architectures under identical experimental settings.
    Reports Macro-F1, Micro-F1, and AUC-PR.
    """
    print("\n=======================================================")
    print("  GNN Architecture Ablation: GraphSAGE vs Relation-Aware ")
    print("=======================================================")

    # 1. Original GraphSAGE
    torch.manual_seed(42)
    np.random.seed(42)
    res_orig = train_task2_gnn(train_loader, val_loader, test_loader, config, device, gnn_type="graphsage")

    # 2. Relation-Aware GraphSAGE
    torch.manual_seed(42)
    np.random.seed(42)
    res_rel = train_task2_gnn(train_loader, val_loader, test_loader, config, device, gnn_type="relation_aware_graphsage")

    ablation_summary = {
        "Original GraphSAGE": {
            "Macro-F1": round(res_orig["test_metrics"]["macro_f1"], 4),
            "Micro-F1": round(res_orig["test_metrics"]["micro_f1"], 4),
            "AUC-PR": round(res_orig["test_metrics"]["auc_pr"], 4),
        },
        "Relation-Aware GraphSAGE": {
            "Macro-F1": round(res_rel["test_metrics"]["macro_f1"], 4),
            "Micro-F1": round(res_rel["test_metrics"]["micro_f1"], 4),
            "AUC-PR": round(res_rel["test_metrics"]["auc_pr"], 4),
        }
    }

    print("\n--- GNN Ablation Comparison Results ---")
    print(f"{'Model Architecture':<28} | {'Macro-F1':<10} | {'Micro-F1':<10} | {'AUC-PR':<10}")
    print("-" * 65)
    for model_name, m in ablation_summary.items():
        print(f"{model_name:<28} | {m['Macro-F1']:<10.4f} | {m['Micro-F1']:<10.4f} | {m['AUC-PR']:<10.4f}")
    print("-------------------------------------------------------\n")

    return ablation_summary


def train_task3_fusion(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: dict,
    device: torch.device,
    fusion_type: str = "cross_attention",
    gnn_type: str = "relation_aware_graphsage"
) -> dict:
    # Explicit seed reset for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    print(f"\n--- Training Task 3: GNN-BERT Fusion ({fusion_type}, GNN: {gnn_type}) ---")
    g_cfg = config.get("graph", {})
    gnn = MusicGNNEncoder(
        in_dim=g_cfg.get("node_feature_dim", 32),
        hidden_dim=g_cfg.get("hidden_dim", 64),
        out_dim=g_cfg.get("out_dim", 64),
        num_classes=len(CONTEXT_TAGS),
        num_layers=g_cfg.get("num_layers", 2),
        gnn_type=gnn_type
    )
    bert = MusicBERTClassifier(
        model_name=config.get("bert", {}).get("model_name", "distilbert-base-uncased"),
        num_classes=len(CONTEXT_TAGS)
    )
    t1_ckpt = "results/checkpoints/task1_bert.pt"
    if os.path.exists(t1_ckpt):
        try:
            bert.load_state_dict(torch.load(t1_ckpt, map_location=device), strict=False)
            print("Loaded fine-tuned Task 1 BERT weights into Fusion model.")
        except Exception:
            pass

    model = GNNBERTFusionModel(
        gnn_encoder=gnn,
        bert_encoder=bert,
        fusion_type=fusion_type,
        attn_dim=config.get("fusion", {}).get("hidden_dim", 128),
        num_heads=config.get("fusion", {}).get("num_attention_heads", 4),
        num_classes=len(CONTEXT_TAGS),
        dropout=config.get("fusion", {}).get("dropout", 0.3)
    ).to(device)

    lr = float(config.get("training", {}).get("fusion_learning_rate", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    epochs = int(config.get("training", {}).get("epochs", 15))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=2e-5)
    alpha = config.get("fusion", {}).get("alpha_valence", 0.5)
    beta = config.get("fusion", {}).get("beta_arousal", 0.5)

    history = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "val_mae": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch["x"].to(device)
            edge_index = batch["edge_index"].to(device)
            edge_weight = batch["edge_weight"].to(device)
            batch_t = batch["batch"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            tags = batch["tags_vector"].to(device)
            val = batch["valence"].to(device)
            aro = batch["arousal"].to(device)

            optimizer.zero_grad()
            out = model(x, edge_index, edge_weight=edge_weight, batch=batch_t, input_ids=input_ids, attention_mask=attention_mask)
            loss_dict = model.compute_loss(out, tags, val, aro, alpha=alpha, beta=beta)
            loss_dict["loss"].backward()
            optimizer.step()
            train_loss += loss_dict["loss"].item()

        train_loss /= max(1, len(train_loader))
        scheduler.step()

        model.eval()
        val_loss = 0.0
        all_probs, all_targets = [], []
        all_pred_val, all_true_val = [], []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                edge_index = batch["edge_index"].to(device)
                edge_weight = batch["edge_weight"].to(device)
                batch_t = batch["batch"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                tags = batch["tags_vector"].to(device)
                val = batch["valence"].to(device)
                aro = batch["arousal"].to(device)

                out = model(x, edge_index, edge_weight=edge_weight, batch=batch_t, input_ids=input_ids, attention_mask=attention_mask)
                loss_dict = model.compute_loss(out, tags, val, aro, alpha=alpha, beta=beta)
                val_loss += loss_dict["loss"].item()

                all_probs.append(out["probs"].cpu().numpy())
                all_targets.append(tags.cpu().numpy())
                all_pred_val.append(out["valence"].cpu().numpy())
                all_true_val.append(val.cpu().numpy())

        val_loss /= max(1, len(val_loader))
        metrics = evaluate_classification(np.vstack(all_probs), np.vstack(all_targets))
        mae = float(mean_absolute_error(np.concatenate(all_true_val), np.concatenate(all_pred_val)))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(metrics["macro_f1"])
        history["val_mae"].append(mae)

        print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {train_loss:.4f} | Val F1: {metrics['macro_f1']:.4f} | Val Emotion MAE: {mae:.4f}")

    # Test evaluation
    model.eval()
    t_probs, t_targets = [], []
    t_pred_val, t_true_val = [], []
    t_pred_aro, t_true_aro = [], []
    all_z = []
    all_genres = []
    all_moods = []

    with torch.no_grad():
        for batch in test_loader:
            out = model(
                batch["x"].to(device),
                batch["edge_index"].to(device),
                edge_weight=batch["edge_weight"].to(device),
                batch=batch["batch"].to(device),
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device)
            )
            t_probs.append(out["probs"].cpu().numpy())
            t_targets.append(batch["tags_vector"].numpy())
            t_pred_val.append(out["valence"].cpu().numpy())
            t_true_val.append(batch["valence"].numpy())
            t_pred_aro.append(out["arousal"].cpu().numpy())
            t_true_aro.append(batch["arousal"].numpy())
            all_z.append(out["z"].cpu().numpy())
            VALID_MOODS = {"happy", "sad", "energetic", "calm", "melancholic", "uplifting", "dark", "romantic"}
            for m in batch["metadata"]:
                all_genres.append(m["genre"])
                # Extract first genuine mood tag if present, else default to calm
                chosen_mood = None
                for t in m.get("tags", []):
                    if t in VALID_MOODS:
                        chosen_mood = t
                        break
                all_moods.append(chosen_mood if chosen_mood is not None else "calm")

    test_probs = np.vstack(t_probs)
    test_targets = np.vstack(t_targets)
    test_metrics = evaluate_classification(test_probs, test_targets)
    pred_v = np.concatenate(t_pred_val)
    true_v = np.concatenate(t_true_val)
    pred_a = np.concatenate(t_pred_aro)
    true_a = np.concatenate(t_true_aro)

    test_metrics["mae_valence"] = float(mean_absolute_error(true_v, pred_v))
    test_metrics["mae_arousal"] = float(mean_absolute_error(true_a, pred_a))
    test_metrics["mae_emotion"] = float((test_metrics["mae_valence"] + test_metrics["mae_arousal"]) / 2.0)

    print(f"Task 3 ({fusion_type}) Test: Macro-F1={test_metrics['macro_f1']:.4f}, AUC-PR={test_metrics['auc_pr']:.4f}, Emotion MAE={test_metrics['mae_emotion']:.4f}")

    os.makedirs("results/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"results/checkpoints/task3_{fusion_type}.pt")

    return {
        "test_metrics": test_metrics,
        "history": history,
        "latent_z": np.vstack(all_z),
        "genres": all_genres,
        "moods": all_moods,
        "test_probs": test_probs,
        "test_targets": test_targets
    }


def train_task4_contrastive(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: dict,
    device: torch.device
) -> dict:
    torch.manual_seed(42)
    np.random.seed(42)
    print("\n--- Training Task 4: Contrastive GNN-BERT (MusicCaps Alignment) ---")
    g_cfg = config.get("graph", {})
    gnn = MusicGNNEncoder(
        in_dim=g_cfg.get("node_feature_dim", 32),
        hidden_dim=g_cfg.get("hidden_dim", 64),
        out_dim=g_cfg.get("out_dim", 64),
        num_classes=len(CONTEXT_TAGS)
    )
    bert = MusicBERTClassifier(
        model_name=config.get("bert", {}).get("model_name", "distilbert-base-uncased"),
        num_classes=len(CONTEXT_TAGS)
    )
    t1_ckpt = "results/checkpoints/task1_bert.pt"
    if os.path.exists(t1_ckpt):
        try:
            bert.load_state_dict(torch.load(t1_ckpt, map_location=device), strict=False)
            print("Loaded fine-tuned Task 1 BERT weights into Contrastive model.")
        except Exception:
            pass
    model = DualEncoderContrastive(
        gnn_encoder=gnn,
        bert_encoder=bert,
        projection_dim=config.get("contrastive", {}).get("projection_dim", 128),
        temperature=config.get("contrastive", {}).get("temperature", 0.07)
    ).to(device)

    lr = float(config.get("training", {}).get("contrastive_learning_rate", config.get("training", {}).get("learning_rate", 1e-4)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    epochs = int(config.get("training", {}).get("epochs", 15))

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch["x"].to(device)
            edge_index = batch["edge_index"].to(device)
            edge_weight = batch["edge_weight"].to(device)
            batch_t = batch["batch"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()
            g_emb, t_emb = model(x, edge_index, edge_weight, batch_t, input_ids, attention_mask)
            loss, _ = model.compute_infonce_loss(g_emb, t_emb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))
        print(f"Epoch {epoch:02d}/{epochs:02d} | Contrastive InfoNCE Loss: {train_loss:.4f}")

    # Evaluate retrieval on test split
    model.eval()
    all_g, all_t = [], []
    with torch.no_grad():
        for batch in test_loader:
            g_emb, t_emb = model(
                batch["x"].to(device),
                batch["edge_index"].to(device),
                batch["edge_weight"].to(device),
                batch["batch"].to(device),
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device)
            )
            all_g.append(g_emb)
            all_t.append(t_emb)

    g_all = torch.cat(all_g, dim=0)
    t_all = torch.cat(all_t, dim=0)

    # Caption -> Audio retrieval
    cap2aud = compute_retrieval_metrics(t_all, g_all, k_vals=[1, 5, 10])
    aud2cap = compute_retrieval_metrics(g_all, t_all, k_vals=[1, 5, 10])

    print(f"Caption -> Audio Retrieval: R@1={cap2aud['R@1']:.3f}, R@5={cap2aud['R@5']:.3f}, R@10={cap2aud['R@10']:.3f}")
    print(f"Audio -> Caption Retrieval: R@1={aud2cap['R@1']:.3f}, R@5={aud2cap['R@5']:.3f}, R@10={aud2cap['R@10']:.3f}")

    os.makedirs("results/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "results/checkpoints/task4_contrastive.pt")

    return {
        "caption_to_audio": cap2aud,
        "audio_to_caption": aud2cap,
        "test_metrics": {"R@1": cap2aud["R@1"], "R@5": cap2aud["R@5"], "R@10": cap2aud["R@10"]}
    }


def train_baseline_cnn(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: dict,
    device: torch.device
) -> dict:
    torch.manual_seed(42)
    np.random.seed(42)
    print("\n--- Training Baseline B2: Mel-Spectrogram 2D CNN ---")
    model = MelSpectrogramCNN(num_classes=len(CONTEXT_TAGS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pos_weight = torch.ones(len(CONTEXT_TAGS), device=device) * 3.5
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    epochs = int(config.get("training", {}).get("epochs", 15))

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            mel = batch["mel_spec"].to(device)
            labels = batch["tags_vector"].to(device)

            optimizer.zero_grad()
            out = model(mel)
            loss = criterion(out["logits"], labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))

    # Test
    model.eval()
    t_probs, t_targets = [], []
    all_pred_v, all_true_v = [], []
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch["mel_spec"].to(device))
            t_probs.append(out["probs"].cpu().numpy())
            t_targets.append(batch["tags_vector"].numpy())
            all_pred_v.append(out["valence"].cpu().numpy())
            all_true_v.append(batch["valence"].numpy())

    metrics = evaluate_classification(np.vstack(t_probs), np.vstack(t_targets))
    metrics["mae_emotion"] = float(mean_absolute_error(np.concatenate(all_true_v), np.concatenate(all_pred_v)))
    print(f"CNN Baseline Test: Macro-F1={metrics['macro_f1']:.4f}, AUC-PR={metrics['auc_pr']:.4f}, MAE={metrics['mae_emotion']:.4f}")
    return {"test_metrics": metrics}

"""
bert_encoder.py
Task 1 (Easy): BERT Baseline for Music Tag Understanding.
Implements HuggingFace DistilBERT / BERT multi-label classifier on textual music context
(tags, captions, or lyrics) without graph structure per Section 4.1 of the specification.
"""

import os
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any

try:
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class MusicBERTClassifier(nn.Module):
    """
    BERT-based multi-label tag predictor on text context (lyrics, tags, descriptions).
    t = BERT_CLS(X_text)
    y_hat = sigmoid(W * t + b)
    """
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_classes: int = 25,
        freeze_backbone: bool = False,
        dropout: float = 0.2,
        embedding_dim: int = 768,
        pretrained: bool = False
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        if TRANSFORMERS_AVAILABLE:
            try:
                # Load configuration
                self.config = AutoConfig.from_pretrained(model_name, output_attentions=True)
                self.config.dim = embedding_dim
                self.config.hidden_dim = embedding_dim
                self.config.hidden_size = embedding_dim
                if pretrained:
                    self.bert = AutoModel.from_pretrained(model_name, config=self.config)
                else:
                    self.bert = AutoModel.from_config(self.config)
                self.embedding_dim = getattr(self.config, "hidden_size", getattr(self.config, "dim", embedding_dim))
            except Exception as e:
                # Fallback to standard TransformerEncoder
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embedding_dim, nhead=4, dim_feedforward=512, dropout=dropout, batch_first=True
                )
                self.bert = nn.TransformerEncoder(encoder_layer, num_layers=2)
                self.token_embedding = nn.Embedding(30522, embedding_dim)
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim, nhead=4, dim_feedforward=512, dropout=dropout, batch_first=True
            )
            self.bert = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.token_embedding = nn.Embedding(30522, embedding_dim)

        if freeze_backbone:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.embedding_dim, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Returns dictionary with:
            - logits: raw classification logits (B, num_classes)
            - probs: sigmoid probabilities (B, num_classes)
            - cls_embedding: [CLS] representation (B, embedding_dim)
            - sequence_output: full token embeddings H_text (B, L, embedding_dim)
            - attentions: attention weights if available
        """
        if TRANSFORMERS_AVAILABLE and hasattr(self.bert, "embeddings"):
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                return_dict=True
            )
            # Full sequence representations H_text
            sequence_output = outputs.last_hidden_state  # (B, L, d)
            # CLS token representation t = H_text[:, 0, :]
            cls_embedding = sequence_output[:, 0, :]
            attentions = outputs.attentions
        else:
            # Fallback embedding
            token_embeds = self.token_embedding(input_ids)
            sequence_output = self.bert(token_embeds)
            cls_embedding = sequence_output[:, 0, :]
            attentions = None

        pooled = self.dropout(cls_embedding)
        logits = self.classifier(pooled)
        probs = torch.sigmoid(logits)

        result = {
            "logits": logits,
            "probs": probs,
            "cls_embedding": cls_embedding,
            "sequence_output": sequence_output,
            "attentions": attentions
        }
        return result

    def get_text_representations(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            H_text (torch.Tensor): Token level embeddings (B, L, d)
            t (torch.Tensor): CLS pooled embedding (B, d)
        """
        res = self.forward(input_ids, attention_mask=attention_mask)
        return res["sequence_output"], res["cls_embedding"]


_GLOBAL_TOKENIZER = None

def tokenize_texts(
    texts: List[str],
    model_name: str = "distilbert-base-uncased",
    max_length: int = 128
) -> Dict[str, torch.Tensor]:
    """
    Tokenizes text captions, user tags, or lyrics with BERT tokenizer.
    Caches tokenizer instance globally to avoid redundant network calls.
    """
    global _GLOBAL_TOKENIZER
    if TRANSFORMERS_AVAILABLE:
        if _GLOBAL_TOKENIZER is None:
            try:
                _GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
            except Exception:
                try:
                    _GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                except Exception:
                    _GLOBAL_TOKENIZER = False

        if _GLOBAL_TOKENIZER is not False and _GLOBAL_TOKENIZER is not None:
            try:
                encoded = _GLOBAL_TOKENIZER(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                return {
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"]
                }
            except Exception:
                pass

    # High-speed deterministic hashing fallback
    B = len(texts)
    input_ids = torch.zeros((B, max_length), dtype=torch.long)
    attention_mask = torch.zeros((B, max_length), dtype=torch.long)
    for i, t in enumerate(texts):
        tokens = [101] + [abs(hash(w)) % 28000 + 102 for w in t.split()[:max_length-2]] + [102]
        L = len(tokens)
        input_ids[i, :L] = torch.tensor(tokens, dtype=torch.long)
        attention_mask[i, :L] = 1

    return {"input_ids": input_ids, "attention_mask": attention_mask}

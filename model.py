from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
from transformers import AutoModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------- #
#  Helper type aliases
# --------------------------------------------------------------------------- #
Span = Tuple[int, int]                 # (token_start, token_end) inclusive
Relation = Tuple[int, int, int]        # (cause_idx, effect_idx, polarity_id)


# --------------------------------------------------------------------------- #
#  Joint model
# --------------------------------------------------------------------------- #
class JointCausalModel(nn.Module):
    """
    One Transformer encoder + three light heads.

    Parameters
    ----------
    encoder_name : str
        HF checkpoint (e.g. "google-bert/bert-base-uncased").
    num_cls_labels : int
        Number of classes for sentence-level classification.
    num_span_labels : int
        Number of classes for BIO tagging. (e.g., O, B-Cause, I-Cause, B-Effect, I-Effect, B-CA_EF, I-CA_EF for 7 labels)
    num_rel_labels : int
        Number of classes for relation prediction.
    """

    def __init__(self, encoder_name, num_cls_labels=2, num_span_labels=7, num_rel_labels=5): # Updated default num_span_labels
        super().__init__()
        self.enc = AutoModel.from_pretrained(encoder_name)
        self.hidden_size = self.enc.config.hidden_size
        self.dropout_rate = 0.1
        self.num_cls_labels = num_cls_labels
        self.num_span_labels = num_span_labels
        self.num_rel_labels = num_rel_labels
        
        # Dropout for regularization
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # Task 1: Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size // 2, num_cls_labels)
        )
        
        # Task 2: Span prediction head with improved architecture
        self.span_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size // 2, num_span_labels)
        )
        
        # Task 3: Relation prediction head

        self.rel_head = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size // 2, num_rel_labels)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training stability."""
        for module in [self.cls_head, self.span_head, self.rel_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(self, input_ids, attention_mask, pair_batch=None, cause_starts=None, 
                cause_ends=None, effect_starts=None, effect_ends=None):
        """Forward pass of the model.

        Args:
            input_ids: Input token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            pair_batch: Batch indices for relation pairs. Each index maps a candidate relation pair
                       to its original sentence within the input batch. Shape: (total_num_pairs_in_batch,)
            cause_starts: Start indices for cause spans. Shape: (total_num_pairs_in_batch,)
            cause_ends: End indices for cause spans. Shape: (total_num_pairs_in_batch,)
            effect_starts: Start indices for effect spans. Shape: (total_num_pairs_in_batch,)
            effect_ends: End indices for effect spans. Shape: (total_num_pairs_in_batch,)
            
        Returns:
            tuple: (cls_logits, span_logits, rel_logits)
                - cls_logits: Classification logits (batch_size, num_cls_labels)
                - span_logits: Span prediction logits (batch_size, seq_len, num_span_labels)
                - rel_logits: Relation prediction logits (total_num_pairs_in_batch, num_rel_labels) or None
        """
        # Get encoder outputs
        enc_out = self.enc(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state
        
        # Apply dropout and layer normalization
        enc_out = self.dropout(enc_out)
        enc_out = self.layer_norm(enc_out)
        
        # Task 1: Classification
        cls_logits = self.cls_head(enc_out[:, 0, :])  # Use [CLS] token
        
        # Task 2: BIO tagging
        span_logits = self.span_head(enc_out)
        
        # Task 3: Relation prediction
        rel_logits = None
        if pair_batch is not None:
            # Get span representations
            span_states = enc_out[pair_batch]  # select per-pair batch rows
            N, T, H = span_states.size()
            
            # Build position masks
            pos = torch.arange(T, device=span_states.device).unsqueeze(0)  # (1, T)
            cause_mask = (cause_starts.unsqueeze(1) <= pos) & (pos <= cause_ends.unsqueeze(1))  # (N, T)
            effect_mask = (effect_starts.unsqueeze(1) <= pos) & (pos <= effect_ends.unsqueeze(1))  # (N, T)
            
            # Get mean-pooled representations for cause/effect spans
            cause_mask = cause_mask.unsqueeze(2)  # (N, T, 1)
            effect_mask = effect_mask.unsqueeze(2)  # (N, T, 1)
            cause_vec = (span_states * cause_mask).sum(1) / (cause_mask.sum(1) + 1e-6)  # (N, H)
            effect_vec = (span_states * effect_mask).sum(1) / (effect_mask.sum(1) + 1e-6)  # (N, H)
            
            # Concatenate cause and effect vectors
            rel_input = torch.cat([cause_vec, effect_vec], dim=1)  # (N, 2H)
            
            # Predict relation
            rel_logits = self.rel_head(rel_input)
        
        return cls_logits, span_logits, rel_logits
"""
Joint causal extraction model + inference predictor (HF‑*friendly* mix‑in).

This file keeps the **exact same forward logic** you wrote, but mixes in
`PyTorchModelHubMixin` so you still get `save_pretrained`, `from_pretrained`,
and `push_to_hub` without rewriting everything around a `PreTrainedModel`.

Dependencies
------------
• transformers ≥ 4.40  
• torch ≥ 2.1  
• huggingface_hub ≥ 0.20

Label utilities expected in your project (import wherever appropriate):
```
from config import (
    label2id_span,
    id2label_span,
    id2label_rel,
    label2id_cls,
)
```
"""

from __future__ import annotations

from typing import List, Dict, Tuple
import itertools

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import PyTorchModelHubMixin

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------- #
# Type aliases
# --------------------------------------------------------------------------- #
Span = Tuple[int, int]            # (start_idx, end_idx) inclusive
Relation = Tuple[Span, Span, int] # (cause_span, effect_span, label_id)

# --------------------------------------------------------------------------- #
# Neural network (unchanged forward, HF mix‑in added)
# --------------------------------------------------------------------------- #
class JointCausalModel(nn.Module, PyTorchModelHubMixin):
    """Joint model with three heads + HF Hub mix‑in.

    * Inherits from **`nn.Module`** to keep training code untouched.
    * Inherits from **`PyTorchModelHubMixin`** to gain
      `save_pretrained`, `from_pretrained`, and `push_to_hub` helpers.
    * All computational logic (encoder → heads) is identical to your original
      implementation; only minimal bookkeeping attributes are added so the
      mix‑in can rebuild the model later.
    """

    # ------------------------------------------------------------------ #
    # Constructor
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        encoder_name: str,
        num_cls_labels: int = 2,
        num_bio_labels: int = 7,
        num_rel_labels: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # ---------- store hyper‑params so we can write them to config.json
        self.encoder_name = encoder_name
        self.num_cls_labels = num_cls_labels
        self.num_bio_labels = num_bio_labels
        self.num_rel_labels = num_rel_labels
        self.dropout_rate = dropout
        # --------------------------------------------------------------

        # ---------- backbone + heads (unchanged) ----------------------
        self.enc = AutoModel.from_pretrained(encoder_name)
        self.hidden_size = self.enc.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.hidden_size)

        self.cls_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, num_cls_labels),
        )

        self.bio_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, num_bio_labels),
        )

        self.rel_head = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, num_rel_labels),
        )

        self._init_weights()

    # ------------------------------------------------------------------ #
    # Required for PyTorchModelHubMixin
    # ------------------------------------------------------------------ #
    def get_config_dict(self) -> Dict:
        """Return *serialisable* hyper‑parameters for config.json."""
        return dict(
            encoder_name=self.encoder_name,
            num_cls_labels=self.num_cls_labels,
            num_bio_labels=self.num_bio_labels,
            num_rel_labels=self.num_rel_labels,
            dropout=self.dropout_rate,
        )

    @classmethod
    def from_config_dict(cls, config: Dict) -> "JointCausalModel":
        """Rebuild a model from config.json (used by `from_pretrained`)."""
        return cls(**config)

    # ------------------------------------------------------------------ #
    # Helper: weight initialisation (unchanged)
    # ------------------------------------------------------------------ #
    def _init_weights(self) -> None:
        for module in (self.cls_head, self.bio_head, self.rel_head):
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    # ------------------------------------------------------------------ #
    # Shared encoder pass (unchanged)
    # ------------------------------------------------------------------ #
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """One encoder forward pass + dropout + LN."""
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = self.dropout(h)
        h = self.layer_norm(h)
        return h

    # ------------------------------------------------------------------ #
    # Training/validation forward (unchanged)
    # ------------------------------------------------------------------ #
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pair_batch: torch.Tensor | None = None,
        cause_starts: torch.Tensor | None = None,
        cause_ends: torch.Tensor | None = None,
        effect_starts: torch.Tensor | None = None,
        effect_ends: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor | None]:
        """Compute logits for all three tasks."""
        hidden = self.encode(input_ids, attention_mask)

        outputs = {
            "cls_logits": self.cls_head(hidden[:, 0]),
            "bio_logits": self.bio_head(hidden),
            "rel_logits": None,
        }

        # relation head (only if candidate pairs provided)
        if pair_batch is not None:
            bio_states = hidden[pair_batch]  # (N, T, H)
            seq_len = bio_states.size(1)
            pos = torch.arange(seq_len, device=bio_states.device).unsqueeze(0)
            c_mask = ((cause_starts.unsqueeze(1) <= pos) & (pos <= cause_ends.unsqueeze(1))).unsqueeze(2)
            e_mask = ((effect_starts.unsqueeze(1) <= pos) & (pos <= effect_ends.unsqueeze(1))).unsqueeze(2)
            c_vec = (bio_states * c_mask).sum(1) / (c_mask.sum(1) + 1e-6)
            e_vec = (bio_states * e_mask).sum(1) / (e_mask.sum(1) + 1e-6)
            outputs["rel_logits"] = self.rel_head(torch.cat([c_vec, e_vec], dim=1))

        return outputs

# # --------------------------------------------------------------------------- #
# # Inference cascade (logic unchanged)
# # --------------------------------------------------------------------------- #
# class JointCausalPredictor:
#     """Runs the 3‑step cascade on *raw sentences* using a trained model."""

#     def __init__(
#         self,
#         model: JointCausalModel,
#         tokenizer_name: str | None = None,
#         device: str | torch.device = DEVICE,
#         cls_threshold: float = 0.5,
#     ) -> None:
#         self.model = model.to(device).eval()
#         tok_name = tokenizer_name or model.encoder_name  # fallback to same encoder
#         self.tokenizer = AutoTokenizer.from_pretrained(tok_name)
#         self.device = device
#         self.cls_threshold = cls_threshold

#     # ------------------------------------------------------------------ #
#     # Public API
#     # ------------------------------------------------------------------ #
#     @torch.no_grad()
#     def __call__(self, texts: List[str] | str) -> List[Dict]:
#         """Analyse a sentence or list of sentences."""
#         if isinstance(texts, str):
#             texts = [texts]

#         batch = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
#         hidden = self.model.encode(**batch)

#         # 1) sentence‑level decision
#         cls_logits = self.model.cls_head(hidden[:, 0])
#         cls_probs = torch.softmax(cls_logits, dim=-1)
#         causal_flags = (cls_probs[:, 1] > self.cls_threshold).tolist()

#         # 2) BIO prediction
#         bio_pred = self.model.bio_head(hidden).argmax(-1)

#         results: List[Dict] = []
#         pair_records: list[Tuple[int, int, int, int, int]] = []

#         for i, is_causal in enumerate(causal_flags):
#             sample = {"causal_prob": cls_probs[i, 1].item(), "is_causal": is_causal, "spans": [], "relations": []}
#             if is_causal:
#                 spans = self._decode_spans(bio_pred[i])
#                 sample["spans"] = spans
#                 cause_ids = [j for j, (lab, _) in enumerate(spans) if lab in {"cause", "internal_CE"}]
#                 effect_ids = [j for j, (lab, _) in enumerate(spans) if lab in {"effect", "internal_CE"}]
#                 for c_id, e_id in itertools.product(cause_ids, effect_ids):
#                     if c_id == e_id and spans[c_id][0] != "internal_CE":
#                         continue
#                     c_s, c_e = spans[c_id][1]
#                     e_s, e_e = spans[e_id][1]
#                     pair_records.append((i, c_s, c_e, e_s, e_e))
#             results.append(sample)

#         # 3) relation classification
#         if pair_records:
#             rel_logits = self._run_rel_head(pair_records, hidden)
#             rel_pred = rel_logits.argmax(-1).tolist()
#             for rec, lab in zip(pair_records, rel_pred):
#                 sent_idx, c_s, c_e, e_s, e_e = rec
#                 results[sent_idx]["relations"].append({
#                     "cause_span": (c_s, c_e),
#                     "effect_span": (e_s, e_e),
#                     "label_id": lab,
#                     # map id→str with id2label_rel if desired
#                 })
#         return results

#     # ------------------------------------------------------------------ #
#     # Helper functions
#     # ------------------------------------------------------------------ #
#     def _decode_spans(self, tag_seq: torch.Tensor) -> List[Tuple[str, Span]]:
#         """Greedy BIO decoding; adjust if you prefer a CRF/FSA."""
#         spans, cur_lab, cur_start = [], None

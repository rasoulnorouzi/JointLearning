"""Joint Causal Extraction Model (switchable CRF / soft‑max)
============================================================================

A *single* PyTorch module that can be instantiated in two modes:

* **CRF mode** (`use_crf=True`, default):  BIO tagging is decoded by a linear–
  chain CRF and trained with negative log‑likelihood (no class weights).
* **Soft‑max mode** (`use_crf=False`):  BIO tagging is trained with standard
  per‑token cross‑entropy; you can pass **class weights** for the rare "B-CE" /
  "I-CE" tags, which is impossible with `torchcrf`.

```python
>>> model = JointCausalModel(use_crf=False)        # soft‑max baseline
>>> crf_model = JointCausalModel(use_crf=True)     # CRF variant
```

The flag is saved in `config.json`, so `from_pretrained()` restores the right
variant automatically.

---------------------------------------------------------------------------
Usage overview
---------------------------------------------------------------------------

**Training**
~~~~~~~~~~~~
(Training code example omitted for brevity, see previous versions)

**Inference**
~~~~~~~~~~~~~

```python
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # Or your model's tokenizer
# model.eval() # Call eval() on the model instance
# text = "Because interest rates rose, housing prices fell."
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Define DEVICE
# encoding  = tokenizer(text, return_tensors="pt").to(DEVICE)
#
# # Basic prediction (rules off by default now)
# base_pred = model.predict_with_rules(original_text=text, encoding=encoding, tokenizer=tokenizer, device=DEVICE)
# import json
# print(json.dumps(base_pred, indent=2))
#
# # Prediction with specific rules activated:
# processed_pred_rule3_active = model.predict_with_rules(
#                               original_text=text,
#                               encoding=encoding,
#                               tokenizer=tokenizer,
#                               device=DEVICE,
#                               apply_rule_enforce_spans_insufficient=True
#                           )
# print(json.dumps(processed_pred_rule3_active, indent=2))
```

The `predict(...)` method provides raw model outputs, while `predict_with_rules(...)`
allows for layered application of heuristic post-processing rules.

---------------------------------------------------------------------------
Implementation
---------------------------------------------------------------------------
"""
from __future__ import annotations

from typing import Dict, Tuple, List, Optional, Any 
import itertools

import torch
import torch.nn as nn
from torchcrf import CRF 
from transformers import AutoModel, PreTrainedTokenizerBase 
from huggingface_hub import PyTorchModelHubMixin

# Attempt to import from local config, provide fallbacks for standalone execution/testing
try:
    from .config import MODEL_CONFIG, id2label_bio, id2label_rel, INFERENCE_CONFIG, DEVICE
except ImportError:
    print("Warning: Could not import from .config. Using fallback configurations for crf_model.py.")
    MODEL_CONFIG = {
        "encoder_name": "bert-base-uncased",
        "num_cls_labels": 2,
        "num_bio_labels": 7, 
        "num_rel_labels": 2, 
        "dropout": 0.1,
    }
    id2label_bio = {
        0: "O", 1: "B-C", 2: "I-C", 3: "B-E", 4: "I-E", 5: "B-CE", 6: "I-CE"
    }
    id2label_rel = {
        0: "NoRel", 1: "Rel_CE"
    }
    INFERENCE_CONFIG = {"cls_threshold": 0.5}
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Type aliases & label maps
# ---------------------------------------------------------------------------
Span = Tuple[int, int]  # inclusive indices (token indices)
label2id_bio = {v: k for k, v in id2label_bio.items()}
label2id_rel = {v: k for k, v in id2label_rel.items()}


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------
class JointCausalModel(nn.Module, PyTorchModelHubMixin):
    """Encoder + three heads with **optional CRF** BIO decoder."""

    # ------------------------------------------------------------------
    # constructor
    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        encoder_name: str = MODEL_CONFIG["encoder_name"],
        num_cls_labels: int = MODEL_CONFIG["num_cls_labels"],
        num_bio_labels: int = MODEL_CONFIG["num_bio_labels"],
        num_rel_labels: int = MODEL_CONFIG["num_rel_labels"],
        dropout: float = MODEL_CONFIG["dropout"],
        use_crf: bool = True,
    ) -> None:
        super().__init__()

        self.encoder_name = encoder_name
        self.num_cls_labels = num_cls_labels
        self.num_bio_labels = num_bio_labels
        self.num_rel_labels = num_rel_labels
        self.dropout_rate = dropout
        self.use_crf = use_crf
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
        self.crf: CRF | None = CRF(num_bio_labels, batch_first=True) if use_crf else None
        self.rel_head = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, num_rel_labels),
        )
        self._init_new_layer_weights()

    def get_config_dict(self) -> Dict:
        return {
            "encoder_name": self.encoder_name,
            "num_cls_labels": self.num_cls_labels,
            "num_bio_labels": self.num_bio_labels,
            "num_rel_labels": self.num_rel_labels,
            "dropout": self.dropout_rate,
            "use_crf": self.use_crf,
        }

    @classmethod
    def from_config_dict(cls, config: Dict) -> "JointCausalModel":
        return cls(**config)

    def _init_new_layer_weights(self):
        for mod in [self.cls_head, self.bio_head, self.rel_head]:
            for sub_module in mod.modules():
                if isinstance(sub_module, nn.Linear):
                    nn.init.xavier_uniform_(sub_module.weight)
                    if sub_module.bias is not None:
                        nn.init.zeros_(sub_module.bias)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.layer_norm(self.dropout(hidden_states))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        bio_labels: torch.Tensor | None = None, 
        pair_batch: torch.Tensor | None = None,
        cause_starts: torch.Tensor | None = None,
        cause_ends: torch.Tensor | None = None,
        effect_starts: torch.Tensor | None = None,
        effect_ends: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor | None]:
        hidden = self.encode(input_ids, attention_mask)
        cls_logits = self.cls_head(hidden[:, 0])
        emissions  = self.bio_head(hidden)
        tag_loss: Optional[torch.Tensor] = None
        if bio_labels is not None and self.use_crf and self.crf is not None:
            active_mask = attention_mask.bool() & (bio_labels != -100)
            if active_mask.shape[1] > 0 : 
                 active_mask[:, 0] = attention_mask[:, 0].bool()
            safe_bio_labels = bio_labels.clone()
            safe_bio_labels[safe_bio_labels == -100] = label2id_bio.get("O", 0)
            if torch.any(active_mask): 
                tag_loss = -self.crf(emissions, safe_bio_labels, mask=active_mask, reduction="mean")
            else: 
                tag_loss = torch.tensor(0.0, device=emissions.device)
        elif bio_labels is not None and not self.use_crf:
            tag_loss = torch.tensor(0.0, device=emissions.device) 
        rel_logits: torch.Tensor | None = None
        if pair_batch is not None and cause_starts is not None and cause_ends is not None \
           and effect_starts is not None and effect_ends is not None:
            bio_states_for_rel = hidden[pair_batch] 
            seq_len_rel = bio_states_for_rel.size(1)
            pos_rel = torch.arange(seq_len_rel, device=bio_states_for_rel.device).unsqueeze(0)
            c_mask = ((cause_starts.unsqueeze(1) <= pos_rel) & (pos_rel <= cause_ends.unsqueeze(1))).unsqueeze(2)
            e_mask = ((effect_starts.unsqueeze(1) <= pos_rel) & (pos_rel <= effect_ends.unsqueeze(1))).unsqueeze(2)
            c_vec = (bio_states_for_rel * c_mask).sum(1) / c_mask.sum(1).clamp(min=1) 
            e_vec = (bio_states_for_rel * e_mask).sum(1) / e_mask.sum(1).clamp(min=1) 
            rel_logits = self.rel_head(torch.cat([c_vec, e_vec], dim=1))
        return {
            "cls_logits": cls_logits,
            "bio_emissions": emissions,
            "tag_loss": tag_loss, 
            "rel_logits": rel_logits, 
        }

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

```python
from torch.utils.data import DataLoader
from causal_data import CausalDataset, CausalDatasetCollator

train_ds = CausalDataset(train_df, tokenizer_name="bert-base-uncased")
collator = CausalDatasetCollator(train_ds.tokenizer, num_rel_labels_model_expects=3)
loader   = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collator)

model     = JointCausalModel(use_crf=False)   # soft‑max + weighted CE
model.to(DEVICE)

# weighted CE only for the soft‑max baseline
class_weights = torch.tensor([1.0, 3.0, 3.0, 3.0, 6.0, 6.0, 0.5], device=DEVICE)
ce_loss_fn   = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)
optimizer    = torch.optim.AdamW(model.parameters(), lr=2e-5)

for epoch in range(num_epochs):
    model.train()
    for batch in loader:
        batch = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()}
        outs  = model(**batch)

        # sentence‑level CLS loss
        cls_loss = torch.nn.functional.cross_entropy(
            outs["cls_logits"], batch["cls_labels"], reduction="mean")

        # BIO tag loss – already returned when use_crf=True.
        if model.use_crf:
            tag_loss = outs["tag_loss"]
        else:
            tag_loss = ce_loss_fn(
                outs["bio_emissions"].view(-1, model.num_bio_labels),
                batch["bio_labels"].view(-1),
            )

        # relation loss (if relation labels present)
        rel_loss = torch.tensor(0.0, device=DEVICE)
        if outs["rel_logits"] is not None and batch["rel_labels"] is not None:
            rel_loss = torch.nn.functional.cross_entropy(
                outs["rel_logits"], batch["rel_labels"], reduction="mean")

        loss = cls_loss + tag_loss + rel_loss
        loss.backward()
        optimizer.step(); optimizer.zero_grad()
```

The **CRF version** uses the negative log‑likelihood returned in `outs["tag_loss"]`; no
weights possible here, so omit `class_weights`.

**Inference**
~~~~~~~~~~~~~

```python
model.eval()
text = "Because interest rates rose, housing prices fell."
enc  = tokenizer(text, return_tensors="pt").to(DEVICE)
print(model.predict(enc))
```

The `predict(...)` method works transparently with both modes: in CRF mode it
runs Viterbi; otherwise it takes the arg‑max of the emissions.

---------------------------------------------------------------------------
Implementation
---------------------------------------------------------------------------
"""
from __future__ import annotations

from typing import Dict, Tuple, List, Optional, Iterable
import itertools

import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import AutoModel
from huggingface_hub import PyTorchModelHubMixin

from jointlearning.config import MODEL_CONFIG, id2label_bio, id2label_rel, INFERENCE_CONFIG, DEVICE

# ---------------------------------------------------------------------------
# Type aliases & label maps
# ---------------------------------------------------------------------------
Span = Tuple[int, int]  # inclusive indices
label2id_bio = {v: k for k, v in id2label_bio.items()}

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

        # hyper‑params kept for Hub config
        self.encoder_name = encoder_name
        self.num_cls_labels = num_cls_labels
        self.num_bio_labels = num_bio_labels
        self.num_rel_labels = num_rel_labels
        self.dropout_rate = dropout
        self.use_crf = use_crf

        # backbone
        self.enc = AutoModel.from_pretrained(encoder_name)
        self.hidden_size = self.enc.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.hidden_size)

        # heads ---------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Hub helpers
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # weight init (task‑specific layers only)
    # ------------------------------------------------------------------
    def _init_new_layer_weights(self):
        for mod in [self.cls_head, self.bio_head, self.rel_head]:
            for sub in mod.modules():
                if isinstance(sub, nn.Linear):
                    nn.init.xavier_uniform_(sub.weight)
                    if sub.bias is not None:
                        nn.init.zeros_(sub.bias)

    # ------------------------------------------------------------------
    # helper
    # ------------------------------------------------------------------
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.layer_norm(self.dropout(h))

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
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
        if bio_labels is not None:
            if self.use_crf:
                active_mask = attention_mask.bool() & (bio_labels != -100)
                safe_lab    = bio_labels.clone()
                safe_lab[safe_lab == -100] = label2id_bio["O"]
                tag_loss = -self.crf(emissions, safe_lab, mask=active_mask, reduction="mean")  # type: ignore[arg-type]
            else:
                tag_loss = torch.tensor(0.0, device=emissions.device)  # computed externally to allow weighting

        rel_logits: torch.Tensor | None = None
        if pair_batch is not None:
            bio_states = hidden[pair_batch]
            seq_len = bio_states.size(1)
            pos = torch.arange(seq_len, device=bio_states.device).unsqueeze(0)
            c_mask = ((cause_starts.unsqueeze(1) <= pos) & (pos <= cause_ends.unsqueeze(1))).unsqueeze(2)
            e_mask = ((effect_starts.unsqueeze(1) <= pos) & (pos <= effect_ends.unsqueeze(1))).unsqueeze(2)
            c_vec = (bio_states * c_mask).sum(1) / c_mask.sum(1).clamp(min=1)
            e_vec = (bio_states * e_mask).sum(1) / e_mask.sum(1).clamp(min=1)
            rel_logits = self.rel_head(torch.cat([c_vec, e_vec], dim=1))

        return {
            "cls_logits": cls_logits,
            "bio_emissions": emissions,
            "tag_loss": tag_loss,
            "rel_logits": rel_logits,
        }

    # ------------------------------------------------------------------
    # Inference API
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def predict(
        self,
        encoding: Dict[str, torch.Tensor],
        *,
        cls_threshold: float | None = None,
        device: str | torch.device | None = None,
    ) -> Dict:
        """End‑to‑end single‑sentence prediction.

        Parameters
        ----------
        encoding : dict returned by a tokenizer (must include *input_ids* and
            *attention_mask*)
        cls_threshold : float, optional – probability threshold for the
            sentence‑level causal decision.  Defaults to
            ``INFERENCE_CONFIG['cls_threshold']`` or 0.5.
        device : torch device, optional – model / tensors will be moved here.

        Returns
        -------
        A dict with keys ::
            cls_prob   float    – P(sentence is causal)
            is_causal  bool
            spans      dict     – role → list[(start, end)]
            relations  list     – detected relations (arg1/arg2 spans, roles, label id)
        """
        if cls_threshold is None:
            cls_threshold = INFERENCE_CONFIG.get("cls_threshold", 0.5)
        if device is None:
            device = DEVICE

        encoding = {k: v.to(device) for k, v in encoding.items()}
        input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

        hidden = self.encode(input_ids, attention_mask)  # (1, T, H)

        # sentence‑level causal prob
        cls_logits   = self.cls_head(hidden[:, 0])
        cls_prob     = torch.softmax(cls_logits, dim=1)[0, 1].item()
        is_causal    = cls_prob >= cls_threshold

        # BIO tagging ----------------------------------------------------
        emissions = self.bio_head(hidden)
        if self.use_crf:
            tag_seq = self.crf.decode(emissions, mask=attention_mask.bool())[0]  # type: ignore[arg-type]
        else:
            tag_seq = emissions.argmax(-1)[0].tolist()
        spans = self._bio_to_spans(tag_seq)

        # relation classification ---------------------------------------
        relations: List[Dict] = []
        if is_causal:
            cand_pairs = self._generate_candidate_pairs(spans)
            for role1, span1, role2, span2 in cand_pairs:
                vec1 = hidden[0, span1[0] : span1[1] + 1].mean(0)
                vec2 = hidden[0, span2[0] : span2[1] + 1].mean(0)
                logits = self.rel_head(torch.cat([vec1, vec2]).unsqueeze(0))
                label_id = int(torch.argmax(logits, dim=1))
                relations.append(
                    {
                        "arg1_span": span1,
                        "arg1_role": role1,
                        "arg2_span": span2,
                        "arg2_role": role2,
                        "label": id2label_rel[label_id],
                        "logits": logits.squeeze(0).cpu().tolist(),
                    }
                )

        return {
            "cls_prob": cls_prob,
            "is_causal": is_causal,
            "spans": spans,
            "relations": relations,
        }

    # ------------------------------------------------------------------
    # Utilities (BIO decoding + pair generation)
    # ------------------------------------------------------------------
    @staticmethod
    def _bio_to_spans(tags: List[int]) -> Dict[str, List[Span]]:
        spans = {"cause": [], "effect": [], "ce": []}
        i = 0
        while i < len(tags):
            lbl = id2label_bio.get(tags[i], "O")
            if lbl.startswith("B-"):
                role = lbl[2:]
                start = i
                i += 1
                while i < len(tags) and id2label_bio.get(tags[i], "O") == f"I-{role}":
                    i += 1
                end = i - 1
                if role == "C":
                    spans["cause"].append((start, end))
                elif role == "E":
                    spans["effect"].append((start, end))
                elif role == "CE":
                    spans["ce"].append((start, end))
            else:
                i += 1
        return spans

    @staticmethod
    def _generate_candidate_pairs(spans: Dict[str, List[Span]]) -> List[Tuple[str, Span, str, Span]]:
        pairs: List[Tuple[str, Span, str, Span]] = []
        causes, effects, ces = spans["cause"], spans["effect"], spans["ce"]
        # cause ↔ effect
        for c in causes:
            for e in effects:
                pairs.append(("cause", c, "effect", e))
        # CE acts as either role
        for ce in ces:
            for e in effects:
                pairs.append(("cause", ce, "effect", e))
            for c in causes:
                pairs.append(("cause", c, "effect", ce))
        # deduplicate (start,end tuple key)
        seen = set(); uniq: List[Tuple[str, Span, str, Span]] = []
        for tpl in pairs:
            key = (tpl[1], tpl[3])
            if key not in seen:
                seen.add(key); uniq.append(tpl)
        return uniq

# ---------------------------------------------------------------------------
# Minimal sanity test & sample training loop runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick forward pass sanity check -------------------------------
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    sample = tok("Interest rates rose, therefore housing prices fell.", return_tensors="pt")

    model = JointCausalModel(use_crf=True)
    out   = model(**sample)
    print("OK — shapes:", out["cls_logits"].shape, out["bio_emissions"].shape)

    # Sample predict ------------------------------------------------
    pred = model.predict(sample)
    print(pred)

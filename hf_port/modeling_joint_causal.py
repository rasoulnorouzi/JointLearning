from __future__ import annotations
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel
from dataclasses import dataclass
try:
    from .config import id2label_bio, id2label_rel, id2label_cls
except ImportError:
    from config import id2label_bio, id2label_rel, id2label_cls

try:
    from .configuration_joint_causal import JointCausalConfig
except ImportError:
    from configuration_joint_causal import JointCausalConfig

# ---------------------------------------------------------------------------
# Type aliases & label maps
# ---------------------------------------------------------------------------
label2id_bio = {v: k for k, v in id2label_bio.items()}
label2id_rel = {v: k for k, v in id2label_rel.items()}
label2id_cls = {v: k for k, v in id2label_cls.items()}

# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------
"""Joint Causal Extraction Model (softmax)
============================================================================

A PyTorch module for joint causal extraction using softmax decoding for BIO tagging.
The model supports class weights for handling imbalanced data.

```python
>>> model = JointCausalModel()        # softmax-based model
"""


# ---------------------------------------------------------------------------
# Span dataclass
# ---------------------------------------------------------------------------
@dataclass
class Span:
    role: str
    start_tok: int
    end_tok: int
    text: str
    is_virtual: bool = False


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------

class JointCausalModel(PreTrainedModel):

    """Encoder + three heads with **optional CRF** BIO decoder.

    This model integrates a pre-trained transformer encoder with three distinct
    heads for:
    1. Classification (cls_head): Predicts a global label for the input.
    2. BIO tagging (bio_head): Performs sequence tagging using BIO scheme.
       Can operate with a CRF layer or standard softmax.
    3. Relation extraction (rel_head): Identifies relations between entities
       detected by the BIO tagging head.
    """
    # Link the model to its config class, as shown in the tutorial.
    config_class = JointCausalConfig
  
    # ------------------------------------------------------------------
    # constructor
    # -----------------------------------------------------------
    def __init__(self, config: JointCausalConfig):

        """Initializes the JointCausalModel.

        Args:
            encoder_name: Name of the pre-trained transformer model to use
                (e.g., "bert-base-uncased").
            num_cls_labels: Number of labels for the classification task.
            num_bio_labels: Number of labels for the BIO tagging task.
            num_rel_labels: Number of labels for the relation extraction task.
            dropout: Dropout rate for regularization.
        """

        super().__init__(config)
        self.config = config

        self.enc = AutoModel.from_pretrained(config.encoder_name)
        self.hidden_size = self.enc.config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
   

        self.cls_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_size // 2, config.num_cls_labels),
        )
        self.bio_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_size // 2, config.num_bio_labels),
        )
        self.rel_head = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_size // 2, config.num_rel_labels),
        )
        self._init_new_layer_weights()

    def get_config_dict(self) -> Dict:
        """Returns the model's configuration as a dictionary."""
        return {
            "encoder_name": self.encoder_name,
            "num_cls_labels": self.num_cls_labels,
            "num_bio_labels": self.num_bio_labels,
            "num_rel_labels": self.num_rel_labels,
            "dropout": self.dropout_rate,
        }

    @classmethod
    def from_config_dict(cls, config: Dict) -> "JointCausalModel":
        """Creates a JointCausalModel instance from a configuration dictionary."""
        return cls(**config)

    def _init_new_layer_weights(self):
        """Initializes the weights of the newly added linear layers.

        Uses Xavier uniform initialization for weights and zeros for biases.
        """
        for mod in [self.cls_head, self.bio_head, self.rel_head]:
            for sub_module in mod.modules():
                if isinstance(sub_module, nn.Linear):
                    nn.init.xavier_uniform_(sub_module.weight)
                    if sub_module.bias is not None:
                        nn.init.zeros_(sub_module.bias)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encodes the input using the transformer model.

        Args:
            input_ids: Tensor of input token IDs.
            attention_mask: Tensor indicating which tokens to attend to.

        Returns:
            Tensor of hidden states from the encoder, passed through dropout
            and layer normalization.
        """
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
        """Performs a forward pass through the model.

        Args:
            input_ids: Tensor of input token IDs.
            attention_mask: Tensor indicating which tokens to attend to.
            bio_labels: Optional tensor of BIO labels for training.
            pair_batch: Optional tensor indicating which hidden states to use
                for relation extraction.
            cause_starts: Optional tensor of start indices for cause spans.
            cause_ends: Optional tensor of end indices for cause spans.
            effect_starts: Optional tensor of start indices for effect spans.
            effect_ends: Optional tensor of end indices for effect spans.

        Returns:
            A dictionary containing:
                - "cls_logits": Logits for the classification task.
                - "bio_emissions": Emissions from the BIO tagging head.
                - "tag_loss": Loss for the BIO tagging task (if bio_labels provided).
                - "rel_logits": Logits for the relation extraction task (if
                  relation extraction inputs provided).
        """
        # Encode input
        hidden = self.encode(input_ids, attention_mask)

        # Classification head
        cls_logits = self.cls_head(hidden[:, 0])  # Use [CLS] token representation

        # BIO tagging head
        emissions  = self.bio_head(hidden)
        tag_loss: Optional[torch.Tensor] = None

        # Calculate BIO tagging loss if labels are provided
        if bio_labels is not None:
            # Softmax loss (typically handled by the training loop's loss function, e.g., CrossEntropyLoss)
            # Here, we initialize it to 0.0 as a placeholder.
            # The actual loss calculation for softmax would compare emissions with bio_labels.
            tag_loss = torch.tensor(0.0, device=emissions.device)

        # Relation extraction head
        rel_logits: torch.Tensor | None = None
        if pair_batch is not None and cause_starts is not None and cause_ends is not None \
           and effect_starts is not None and effect_ends is not None:
            # Select hidden states corresponding to the pairs for relation extraction
            bio_states_for_rel = hidden[pair_batch] 
            seq_len_rel = bio_states_for_rel.size(1)
            pos_rel = torch.arange(seq_len_rel, device=bio_states_for_rel.device).unsqueeze(0)

            # Create masks for cause and effect spans
            c_mask = ((cause_starts.unsqueeze(1) <= pos_rel) & (pos_rel <= cause_ends.unsqueeze(1))).unsqueeze(2)
            e_mask = ((effect_starts.unsqueeze(1) <= pos_rel) & (pos_rel <= effect_ends.unsqueeze(1))).unsqueeze(2)

            # Compute mean-pooled representations for cause and effect spans
            c_vec = (bio_states_for_rel * c_mask).sum(1) / c_mask.sum(1).clamp(min=1) # Average pooling, clamp to avoid div by zero
            e_vec = (bio_states_for_rel * e_mask).sum(1) / e_mask.sum(1).clamp(min=1) # Average pooling, clamp to avoid div by zero
            
            # Concatenate cause and effect vectors and pass through relation head
            rel_logits = self.rel_head(torch.cat([c_vec, e_vec], dim=1))

        return {
            "cls_logits": cls_logits,
            "bio_emissions": emissions,
            "tag_loss": tag_loss, 
            "rel_logits": rel_logits, 
        }
    


# ---------------------------------------------------------------------------
# Refactored prediction & post‑processing utilities for JointCausalModel
# ---------------------------------------------------------------------------

    def predict(
        self,
        sents: List[str],
        tokenizer=None,
        *,
        rel_mode: str = "auto",
        rel_threshold: float = 0.55,
        cause_decision: str = "cls+span",
    ) -> List[dict]:
        """End‑to‑end inference for causal sentence extraction.

        *   **Sub‑token stitching** – word‑pieces are merged so we never output
            truncated forms like "ins" or "##pse".
        *   **Connector bridging** – spans may jump over a single function word
            (``of``, ``to`` …) when that yields a cleaner phrase.
        *   **Self‑loops removed** – we never return relations where *cause == effect*.
        *   **Probability threshold raised** to 0.55 for relation head to
            reduce spurious pairs.
        """
        # ------------------------------------------------------------------
        # 0. Tokeniser & device
        # ------------------------------------------------------------------
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.encoder_name, use_fast=True)

        device = next(self.parameters()).device
        to_dev = lambda d: {k: v.to(device) for k, v in d.items()}

        outputs: List[dict] = []

        # ------------------------------------------------------------------
        # 1. Iterate through sentences
        # ------------------------------------------------------------------
        for sent in sents:
            enc = tokenizer(sent, return_tensors="pt", truncation=True, max_length=512)
            enc = to_dev(enc)

            with torch.no_grad():
                base = self(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])

            cls_logits = base["cls_logits"].squeeze(0)                    # (2,)
            bio_ids    = base["bio_emissions"].squeeze(0).argmax(-1).tolist()
            tokens     = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze(0))
            bio_labels = [id2label_bio[i] for i in bio_ids]

            fixed_labels = self._apply_bio_rules(tokens, bio_labels)
            spans        = self._merge_spans(tokens, fixed_labels, tokenizer)

            # sentence‑level causal flag
            is_causal = self._decide_causal(cls_logits, spans, cause_decision)

            # ------------------------------------------------------------------
            # 2. Relation extraction
            # ------------------------------------------------------------------
            rels: List[dict] = []
            cause_spans  = [s for s in spans if s.role in {"C", "CE"}]
            effect_spans = [s for s in spans if s.role in {"E", "CE"}]

            if cause_spans and effect_spans:
                # --- quick heuristic (single‑side singleton) ----------------
                if rel_mode == "auto" and (len(cause_spans) == 1 or len(effect_spans) == 1):
                    if len(cause_spans) == 1:  # one cause – many effects
                        for e in effect_spans:
                            if cause_spans[0].text.lower() != e.text.lower():
                                rels.append({"cause": cause_spans[0].text, "effect": e.text, "type": "Rel_CE"})
                    else:                       # many causes – one effect
                        for c in cause_spans:
                            if c.text.lower() != effect_spans[0].text.lower():
                                rels.append({"cause": c.text, "effect": effect_spans[0].text, "type": "Rel_CE"})
                else:
                    # --- relation head scoring ------------------------------
                    pair_meta = [
                        (c, e) for c in cause_spans for e in effect_spans
                        if not (c.start_tok == e.start_tok and c.end_tok == e.end_tok)
                    ]
                    if pair_meta:
                        pair_batch    = torch.zeros(len(pair_meta), dtype=torch.long, device=device)
                        cause_starts  = torch.tensor([c.start_tok for c, _ in pair_meta], device=device)
                        cause_ends    = torch.tensor([c.end_tok   for c, _ in pair_meta], device=device)
                        effect_starts = torch.tensor([e.start_tok for _, e in pair_meta], device=device)
                        effect_ends   = torch.tensor([e.end_tok   for _, e in pair_meta], device=device)

                        rel_logits = self(
                            input_ids=enc["input_ids"],
                            attention_mask=enc["attention_mask"],
                            pair_batch=pair_batch,
                            cause_starts=cause_starts,
                            cause_ends=cause_ends,
                            effect_starts=effect_starts,
                            effect_ends=effect_ends,
                        )["rel_logits"]

                        probs = torch.softmax(rel_logits, dim=-1)[:, 1].tolist()  # Rel_CE column
                        for (c, e), p in zip(pair_meta, probs):
                            if p >= rel_threshold and c.text.lower() != e.text.lower():
                                rels.append({"cause": c.text, "effect": e.text, "type": "Rel_CE"})

            # deduplicate while preserving order
            seen = set()
            uniq = []
            for r in rels:
                key = (r["cause"].lower(), r["effect"].lower())
                if key not in seen:
                    seen.add(key)
                    uniq.append(r)
            rels = uniq

            outputs.append({
                "text": sent,
                "causal": is_causal,
                "relations": rels,
            })

        return outputs

    # ------------------------------------------------------------------
    # BIO utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_bio_rules(tok: List[str], lab: List[str]) -> List[str]:
        """Light‑touch BIO sanitiser that fixes **intra‑span role clashes** and
        common WordPiece artefacts while deferring to model probabilities.

        Added rule (R‑6)
        ----------------
        When a contiguous non‑O block mixes **C** and **E** roles (e.g.
        ``B‑C I‑C I‑E I‑C``) we collapse the entire block to the *majority*
        role (ties prefer **C**).  Only the first token keeps the ``B‑`` prefix.
        """
        n = len(tok)
        out = lab.copy()

        # R‑1 propagate to ## -------------------------------------------------
        for i in range(1, n):
            if tok[i].startswith("##") and out[i] == "O" and out[i-1] != "O":
                role = out[i-1].split("-")[-1]
                out[i] = f"I-{role}"

        # R‑2 stray I‑tags → B ----------------------------------------------
        for i in range(n):
            if out[i].startswith("I-") and (i == 0 or out[i-1] == "O"):
                out[i] = out[i].replace("I-", "B-", 1)

        # R‑3 merge adjacent B blocks of same role ---------------------------
        for i in range(1, n):
            if out[i].startswith("B-") and out[i-1] != "O":
                role_prev = out[i-1].split("-")[-1]
                role_curr = out[i].split("-")[-1]
                if role_prev == role_curr:
                    out[i] = out[i].replace("B-", "I-", 1)

        # R‑4 (removed): We no longer force punctuation tokens to O
        # This keeps apostrophes/hyphens inside spans when the model labels them.

        # R‑5 CE disambiguation --------------------------------------------- CE disambiguation ---------------------------------------------
        roles_present = {tag.split("-")[-1] for tag in out if tag != "O"}
        if "CE" in roles_present and ("C" not in roles_present or "E" not in roles_present):
            target = "C" if "C" not in roles_present else "E"
            for i, tag in enumerate(out):
                if tag.endswith("CE"):
                    out[i] = tag[:-2] + target

        # R‑6 intra‑span role clash fix -------------------------------------
        i = 0
        while i < n:
            if out[i] == "O":
                i += 1
                continue
            start = i
            role_counts = {"C": 0, "E": 0}
            while i < n and out[i] != "O" and not (i > start and out[i].startswith("B-")):
                role = out[i].split("-")[-1]
                if role == "CE":
                    role_counts["C"] += 1
                    role_counts["E"] += 1
                else:
                    role_counts[role] += 1
                i += 1
            maj = "C" if role_counts["C"] >= role_counts["E"] else "E"
            j = start
            first = True
            while j < i:
                out[j] = ("B-" if first else "I-") + maj
                first = False
                j += 1

        # R‑7 connector & punctuation bridge ----------------------------------
        CONNECT = {"of", "to", "with", "for", "and", "or", "but", "in"}
        for k in range(1, n - 1):
            left_role  = out[k - 1].split("-")[-1] if out[k - 1] != "O" else None
            right_role = out[k + 1].split("-")[-1] if out[k + 1] != "O" else None
            if not left_role or left_role != right_role:
                continue
            # 7a: connector word originally tagged O
            if out[k] == "O" and tok[k].lower() in CONNECT:
                out[k] = "I-" + left_role
            # 7b: single‑char punctuation / hyphen / apostrophe bridge
            elif out[k] == "O" and len(tok[k]) == 1 and not tok[k].isalnum():
                out[k] = "I-" + left_role
            # 7c: mis‑role single token sandwiched by same role
            elif out[k].startswith("I-") and out[k].split("-")[-1] != left_role:
                out[k] = "I-" + left_role

        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _merge_spans(tok: List[str], lab: List[str], tokenizer) -> List["Span"]:
        """Turn cleaned BIO labels into Span objects.

        Policy:
        1. **First pass** – assemble raw spans, letting them bridge a single
           connector (of, to, with, for, and, or, but, in).
        2. **Trim** leading/trailing connectors & punctuation.
        3. **Normalise** hyphen spacing & strip quotes.
        4. **Role‑wise pruning** – if a role has ≥1 span with *≥2 words*, drop
           *all* its 1‑word spans.  This removes stray nouns like "choices"
           while preserving them when they are the *only* cause/effect.
        """
        CONNECT = {"of", "to", "with", "for", "and", "or", "but", "in"}

        spans: List[Span] = []
        i, n = 0, len(tok)
        while i < n:
            if lab[i] == "O":
                i += 1; continue
            role = lab[i].split("-")[-1]
            s = i
            i += 1
            while i < n:
                if lab[i].startswith("I-"):
                    i += 1; continue
                if tok[i].lower() in CONNECT and lab[i] == "O" and i+1 < n and lab[i+1].startswith("I-"):
                    i += 1; continue
                break
            e = i - 1
            text = tokenizer.convert_tokens_to_string(tok[s:e+1])
            # basic cleanup
            text = text.replace(" - ", "-").replace(" -", "-").replace("- ", "-")
            text = text.strip("\"'”’““”")
            words = text.split()
            while words and words[0].lower() in CONNECT:
                words.pop(0)
            while words and words[-1].lower() in CONNECT:
                words.pop()
            if not words:
                continue
            clean_text = " ".join(words)
            spans.append(Span(role, s, e, clean_text))

        # role‑wise pruning --------------------------------------------------
        from collections import defaultdict, OrderedDict
        import re
        by_role = defaultdict(list)
        for sp in spans:
            by_role[sp.role].append(sp)
        final: List[Span] = []
        for role, group in by_role.items():
            has_multi = any((g.end_tok - g.start_tok) >= 1 for g in group)
            for sp in group:
                single_tok = (sp.end_tok - sp.start_tok) == 0
                # Remove verb/pruning logic: do not check for looks_verb
                if single_tok:
                    if role == "C":
                        if has_multi:
                            continue
                    elif role == "E":
                        if has_multi:
                            continue
                final.append(sp)
        final.sort(key=lambda s: s.start_tok)
        # second pass: merge over *pure punctuation* gaps only -----------------
        merged: List[Span] = []
        def is_punct(tok):
            return len(tok) == 1 and not tok.isalnum()
        for sp in final:
            if merged and sp.role == merged[-1].role:
                gap_tokens = tok[merged[-1].end_tok + 1 : sp.start_tok]
                if gap_tokens and all(is_punct(t) for t in gap_tokens):
                    # safe to merge across punctuation (e.g., apostrophe or hyphen)
                    combined_text = tokenizer.convert_tokens_to_string(tok[merged[-1].start_tok: sp.end_tok + 1]).strip("\"'”’““”")
                    merged[-1] = Span(sp.role, merged[-1].start_tok, sp.end_tok, combined_text)
                    continue
            merged.append(sp)
        return merged



    def _decide_causal(self, cls_logits, spans, cause_decision):
        """Determine if a sentence is causal based on classification logits and spans.
        
        Args:
            cls_logits: Tensor of classification logits
            spans: List of extracted spans
            cause_decision: Strategy for determining causality ('cls_only', 'span_only', or 'cls+span')
            
        Returns:
            bool: True if the sentence is determined to be causal
        """
        prob_causal = torch.softmax(cls_logits, dim=-1)[1].item()
        has_spans = bool(spans)
        
        if cause_decision == "cls_only":
            return prob_causal >= 0.5
        elif cause_decision == "span_only":
            return has_spans
        else:  # "cls+span" - default behavior
            return prob_causal >= 0.5 and has_spans
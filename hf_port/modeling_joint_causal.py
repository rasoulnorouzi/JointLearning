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
        """End‑to‑end inference for causal sentence extraction (batched).

        Args:
            sents: List of input sentences for causal extraction.
            tokenizer: Tokenizer instance for encoding sentences. If None, a default tokenizer is initialized.
            rel_mode: Strategy for relation extraction. "auto" mode simplifies relations when spans are limited.
            rel_threshold: Probability threshold for relation head to reduce spurious pairs.
            cause_decision: Strategy for determining causality ('cls_only', 'span_only', or 'cls+span').

        Returns:
            List of dictionaries containing:
                - "text": Original sentence.
                - "causal": Boolean indicating if the sentence is causal.
                - "relations": List of extracted causal relations.
        """
        # ------------------------------------------------------------------
        # 0. Tokeniser & device
        # ------------------------------------------------------------------
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.encoder_name, use_fast=True)

        device = next(self.parameters()).device
        to_dev = lambda d: {k: v.to(device) for k, v in d.items()}  # Move tensors to the model's device

        outputs: List[dict] = []

        # ------------------------------------------------------------------
        # 1. Batch tokenize all sentences
        # ------------------------------------------------------------------
        enc = tokenizer(sents, return_tensors="pt", truncation=True, max_length=512, padding=True)
        enc = to_dev(enc)  # Ensure tensors are on the correct device

        with torch.no_grad():
            base = self(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])

        cls_logits_batch = base["cls_logits"]              # Sentence-level classification logits
        bio_emissions_batch = base["bio_emissions"]        # BIO tagging emissions
        input_ids_batch = enc["input_ids"]                 # Token IDs for each sentence
        attention_mask_batch = enc["attention_mask"]       # Attention mask for each sentence

        batch_size = input_ids_batch.size(0)

        for i in range(batch_size):
            seq_len = attention_mask_batch[i].sum().item()  # Determine the actual sequence length
            input_ids = input_ids_batch[i][:seq_len]        # Trim padding tokens
            bio_emissions = bio_emissions_batch[i][:seq_len]  # Trim emissions to sequence length
            tokens = tokenizer.convert_ids_to_tokens(input_ids)  # Convert token IDs to actual tokens
            bio_ids = bio_emissions.argmax(-1).tolist()    # Get predicted BIO label indices
            bio_labels = [id2label_bio[j] for j in bio_ids]  # Map indices to label names

            # Apply BIO rules to clean up predictions
            fixed_labels = self._apply_bio_rules(tokens, bio_labels)
            spans = self._merge_spans(tokens, fixed_labels, tokenizer)  # Merge spans based on cleaned labels

            # Determine if the sentence is causal based on classification logits and spans
            is_causal = self._decide_causal(cls_logits_batch[i], spans, cause_decision)

            # ------------------------------------------------------------------
            # 2. Relation extraction (per sentence, as before)
            # ------------------------------------------------------------------
            rels: List[dict] = []
            pure_cause_spans = [s for s in spans if s.role == "C"]  # Extract pure cause spans
            pure_effect_spans = [s for s in spans if s.role == "E"]  # Extract pure effect spans
            ce_spans = [s for s in spans if s.role == "CE"]  # Extract combined cause-effect spans
            cause_spans = pure_cause_spans + ce_spans
            effect_spans = pure_effect_spans + ce_spans

            if cause_spans and effect_spans:
                # Check for presence of pure causes/effects and combined spans
                has_pure_causes = len(pure_cause_spans) > 0
                has_pure_effects = len(pure_effect_spans) > 0
                has_ce_spans = len(ce_spans) > 0

                if has_ce_spans and not (has_pure_causes or has_pure_effects):
                    # Skip relation extraction if only combined spans exist
                    pass
                elif rel_mode == "auto" and (len(cause_spans) == 1 or len(effect_spans) == 1):
                    # Simplified relation extraction for single spans
                    if len(cause_spans) == 1:
                        for e in effect_spans:
                            if (cause_spans[0].text.lower() != e.text.lower() or 
                                (cause_spans[0].role == "CE" and e.role != "CE")):
                                rels.append({"cause": cause_spans[0].text, "effect": e.text, "type": "Rel_CE"})
                    else:
                        for c in cause_spans:
                            if (c.text.lower() != effect_spans[0].text.lower() or 
                                (c.role == "CE" and effect_spans[0].role != "CE")):
                                rels.append({"cause": c.text, "effect": effect_spans[0].text, "type": "Rel_CE"})
                else:
                    # Full relation extraction for multiple spans
                    pair_meta = []
                    for c in cause_spans:
                        for e in effect_spans:
                            if (not (c.start_tok == e.start_tok and c.end_tok == e.end_tok) or
                                (c.role == "CE" and e.role in {"C", "E"}) or
                                (c.role in {"C", "E"} and e.role == "CE")):
                                pair_meta.append((c, e))
                    if pair_meta:
                        # Prepare tensors for this sentence only
                        pair_batch = torch.zeros(len(pair_meta), dtype=torch.long, device=device)
                        cause_starts = torch.tensor([c.start_tok for c, _ in pair_meta], device=device)
                        cause_ends = torch.tensor([c.end_tok for c, _ in pair_meta], device=device)
                        effect_starts = torch.tensor([e.start_tok for _, e in pair_meta], device=device)
                        effect_ends = torch.tensor([e.end_tok for _, e in pair_meta], device=device)
                        rel_logits = self(
                            input_ids=input_ids.unsqueeze(0),
                            attention_mask=attention_mask_batch[i][:seq_len].unsqueeze(0),
                            pair_batch=pair_batch,
                            cause_starts=cause_starts,
                            cause_ends=cause_ends,
                            effect_starts=effect_starts,
                            effect_ends=effect_ends,
                        )["rel_logits"]
                        probs = torch.softmax(rel_logits, dim=-1)[:, 1].tolist()  # Extract probabilities for relation type
                        for (c, e), p in zip(pair_meta, probs):
                            if p >= rel_threshold and c.text.lower() != e.text.lower():
                                rels.append({"cause": c.text, "effect": e.text, "type": "Rel_CE"})
            # Remove duplicate relations
            seen = set()
            uniq = []
            for r in rels:
                key = (r["cause"].lower(), r["effect"].lower())
                if key not in seen:
                    seen.add(key)
                    uniq.append(r)
            rels = uniq

            # Append results for this sentence
            outputs.append({
                "text": sents[i],
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
        # This keeps apostrophes/hyphens inside spans when the model labels them.        # R‑5 CE disambiguation - only convert CE if no other roles present
        roles_present = {tag.split("-")[-1] for tag in out if tag != "O"}
        if "CE" in roles_present and "C" not in roles_present and "E" not in roles_present:
            # Only CE tags present - convert all to C (arbitrary choice)
            for i, tag in enumerate(out):
                if tag.endswith("CE"):
                    out[i] = tag[:-2] + "C"

        # R‑6 intra‑span role clash fix - preserve CE spans when meaningful
        i = 0
        while i < n:
            if out[i] == "O":
                i += 1
                continue
            start = i
            role_counts = {"C": 0, "E": 0, "CE": 0}
            has_mixed_roles = False
            
            # Count roles in this span and check for mixing
            while i < n and out[i] != "O" and not (i > start and out[i].startswith("B-")):
                role = out[i].split("-")[-1]
                role_counts[role] += 1
                i += 1
            
            # Check if span has mixed C/E roles (not including CE)
            non_ce_roles = set()
            j = start
            while j < i:
                role = out[j].split("-")[-1]
                if role in {"C", "E"}:
                    non_ce_roles.add(role)
                j += 1
            
            if len(non_ce_roles) > 1:
                # Mixed C and E tags - resolve to majority
                maj = "C" if role_counts["C"] >= role_counts["E"] else "E"
                j = start
                first = True
                while j < i:
                    out[j] = ("B-" if first else "I-") + maj
                    first = False
                    j += 1
            elif role_counts["CE"] > 0 and len(non_ce_roles) == 0:
                # Pure CE span - keep as CE
                j = start
                first = True
                while j < i:
                    out[j] = ("B-" if first else "I-") + "CE"
                    first = False
                    j += 1
            elif role_counts["CE"] > 0 and len(non_ce_roles) == 1:
                # CE mixed with single pure role - check if CE is meaningful
                # If we have other pure spans of different types, keep CE
                other_roles = {tag.split("-")[-1] for tag in out if tag != "O"}
                pure_role = list(non_ce_roles)[0]
                
                if (pure_role == "C" and "E" in other_roles) or (pure_role == "E" and "C" in other_roles):
                    # CE is meaningful - keep it
                    j = start
                    first = True
                    while j < i:
                        out[j] = ("B-" if first else "I-") + "CE"
                        first = False
                        j += 1
                else:
                    # CE not meaningful - convert to pure role
                    j = start
                    first = True
                    while j < i:
                        out[j] = ("B-" if first else "I-") + pure_role
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
                out[k] = "I-" + left_role            # 7c: mis‑role single token sandwiched by same role
            elif out[k].startswith("I-") and out[k].split("-")[-1] != left_role:
                out[k] = "I-" + left_role

        # R‑8 gap‑tolerant B‑tag merging ------------------------------------
        # Merge B- tags of the same type separated by small gaps (≤1 O tokens)
        # This reduces span fragmentation like "B-E O B-E" -> "B-E I-E I-E"
        b_positions = {}
        for i, label in enumerate(out):
            if label.startswith("B-"):
                role = label.split("-")[1]
                if role not in b_positions:
                    b_positions[role] = []
                b_positions[role].append(i)
        
        for role, positions in b_positions.items():
            if len(positions) < 2:
                continue
                
            # Group positions that are close together (gap ≤ 1)
            groups = []
            current_group = [positions[0]]
            
            for i in range(1, len(positions)):
                prev_pos = positions[i-1]
                curr_pos = positions[i]
                gap_size = curr_pos - prev_pos - 1
                
                if gap_size <= 1:  # Allow gaps of 0 or 1 O tokens
                    gap_labels = out[prev_pos + 1:curr_pos]
                    if all(label == "O" for label in gap_labels):
                        current_group.append(curr_pos)
                    else:
                        groups.append(current_group)
                        current_group = [curr_pos]
                else:
                    groups.append(current_group)
                    current_group = [curr_pos]
            
            groups.append(current_group)
            
            # Merge groups with multiple B- tags
            for group in groups:
                if len(group) > 1:
                    first_pos = group[0]
                    last_pos = group[-1]
                    
                    for pos in range(first_pos + 1, last_pos + 1):
                        if pos in group[1:]:  # B- tag to convert
                            out[pos] = f"I-{role}"
                        elif out[pos] == "O":  # Fill gap
                            out[pos] = f"I-{role}"

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
            spans.append(Span(role, s, e, clean_text))        # role‑wise pruning --------------------------------------------------
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
                # Only remove single-token spans if they look like artifacts
                # Keep all meaningful single-token spans like "depression", "cancer", etc.
                if single_tok:
                    # Check if the span text looks like a meaningful entity
                    is_meaningful = (
                        len(sp.text) > 2 and  # Longer than 2 characters
                        sp.text.isalpha() and  # Only alphabetic characters
                        not sp.text.lower() in {"this", "that", "it", "they", "them", "he", "she", "we", "i", "you"}  # Not pronouns
                    )
                    if not is_meaningful and has_multi:
                        # Only skip single-token spans that seem like artifacts when multi-token spans exist
                        if role == "C" or role == "E":
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
        
        # Check for presence of both cause and effect spans (CE spans count as both)
        has_cause_spans = any(x.role in ("C", "CE") for x in spans)
        has_effect_spans = any(x.role in ("E", "CE") for x in spans)
        has_both_spans = has_cause_spans and has_effect_spans
        
        if cause_decision == "cls_only":
            return prob_causal >= 0.5
        elif cause_decision == "span_only":
            return has_both_spans
        else:  # "cls+span" - default behavior
            return prob_causal >= 0.5 and has_both_spans
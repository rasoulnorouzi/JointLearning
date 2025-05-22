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

    @torch.inference_mode()
    def predict(
        self,
        encoding: Dict[str, torch.Tensor],
        *,
        cls_threshold: float | None = None,
        device: str | torch.device | None = None,
    ) -> Dict[str, Any]:
        if cls_threshold is None:
            cls_threshold = INFERENCE_CONFIG.get("cls_threshold", 0.5)
        current_device = device if device is not None else DEVICE
        self.to(current_device) 
        self.eval() 
        encoding = {k: v.to(current_device) for k, v in encoding.items() if torch.is_tensor(v)}
        input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
        hidden = self.encode(input_ids, attention_mask) 
        cls_logits   = self.cls_head(hidden[:, 0]) 
        cls_probs    = torch.softmax(cls_logits, dim=1)
        cls_prob_causal = cls_probs[0, 1].item() if cls_probs.size(1) > 1 else cls_probs[0,0].item()
        is_causal    = cls_prob_causal >= cls_threshold
        emissions = self.bio_head(hidden)
        tag_seq: List[int]
        if self.use_crf and self.crf is not None:
            crf_mask = attention_mask.bool()
            if crf_mask.shape[1] == 0: 
                tag_seq = []
            else:
                tag_sequence_list = self.crf.decode(emissions, mask=crf_mask)
                tag_seq = tag_sequence_list[0] 
        else:
            tag_seq = emissions.argmax(-1)[0].tolist() 
        spans = self._bio_to_spans(tag_seq, attention_mask.squeeze(0) if attention_mask.ndim > 1 else attention_mask)
        relations: List[Dict] = []
        has_any_span = any(s_list for s_list in spans.values())
        if has_any_span: 
            candidate_relations = self._generate_candidate_pairs(spans)
            for role1, span1, role2, span2 in candidate_relations:
                seq_len_from_hidden = hidden.size(1)
                if not (0 <= span1[0] <= span1[1] < seq_len_from_hidden and \
                        0 <= span2[0] <= span2[1] < seq_len_from_hidden):
                    continue
                vec1 = hidden[0, span1[0] : span1[1] + 1].mean(0) 
                vec2 = hidden[0, span2[0] : span2[1] + 1].mean(0) 
                relation_prediction_logits = self.rel_head(torch.cat([vec1, vec2]).unsqueeze(0))
                predicted_label_id = int(torch.argmax(relation_prediction_logits, dim=1))
                relations.append(
                    {
                        "arg1_span": span1, 
                        "arg1_role": role1, 
                        "arg2_span": span2,
                        "arg2_role": role2, 
                        "label": id2label_rel.get(predicted_label_id, "UnknownRel"), 
                        "logits": relation_prediction_logits.squeeze(0).cpu().tolist(), 
                    }
                )
        return {
            "cls_prob": cls_prob_causal,
            "is_causal": is_causal,
            "spans": spans,
            "relations": relations,
        }

    def _get_text_from_span(self, input_ids_squeezed: torch.Tensor, span: Span, tokenizer: PreTrainedTokenizerBase) -> str:
        if not (0 <= span[0] <= span[1] < len(input_ids_squeezed)):
            return "[Error: Invalid Span Indices]"

        original_start_idx, original_end_idx = span

        # Expand to the left to find the true start of the word
        current_start = original_start_idx
        while current_start > 0:
            token_at_current_start_str = tokenizer.convert_ids_to_tokens(input_ids_squeezed[current_start].item())
            
            # If the token at current_start is not a subword, it's the beginning of a word.
            if not token_at_current_start_str.startswith("##"):
                break 

            # If token_at_current_start_str is a subword (e.g., "##usa"), check the token to its left.
            token_to_left_str = tokenizer.convert_ids_to_tokens(input_ids_squeezed[current_start - 1].item())
            
            # If the token to the left is a special token, stop. current_start is the beginning of this subword sequence.
            if token_to_left_str in tokenizer.all_special_tokens:
                break
            
            # If the token to the left is part of the same word (either a word start like "caus" 
            # or another subword like "##pre"), move current_start left.
            current_start -= 1
            # If token_to_left_str was a word start (e.g., "caus"), the next iteration's check 
            # on token_at_current_start_str (which would be "caus") will not start with "##", 
            # causing the loop to break, with current_start correctly pointing to "caus".
        
        # Expand to the right to find the true end of the word
        current_end = original_end_idx
        while current_end < len(input_ids_squeezed) - 1:
            # Check the token to the RIGHT of current_end
            token_to_right_str = tokenizer.convert_ids_to_tokens(input_ids_squeezed[current_end + 1].item())

            # If the token to the right is a special token, stop.
            if token_to_right_str in tokenizer.all_special_tokens:
                break 
            
            # If it's a subword continuation (starts with "##"), expand current_end.
            if token_to_right_str.startswith("##"):
                current_end += 1
            else: 
                # Not a subword continuation, so current_end is the actual end of the word.
                break
            
        final_span_token_ids = input_ids_squeezed[current_start : current_end + 1]
        
        text = tokenizer.decode(
            token_ids=final_span_token_ids.tolist(), # Ensure it's a list of Python ints
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True  # Important for joining subwords
        ).strip()
        
        return text

    @torch.inference_mode()
    def predict_with_rules(
        self,
        original_text: str,
        encoding: Dict[str, torch.Tensor], 
        tokenizer: PreTrainedTokenizerBase, 
        *,
        cls_threshold: float | None = None, 
        device: str | torch.device | None = None,
        apply_rule_promote_sentence: bool = False, 
        apply_rule_heuristic_connect: bool = False, 
        apply_rule_enforce_spans_insufficient: bool = False, 
        apply_rule_enforce_no_relations: bool = False, 
        no_relation_label: str = "NoRel",
        default_causal_relation_type: str = "Rel_CE"
    ) -> Dict[str, Any]:
        """
        Performs end-to-end single-sentence causal extraction with configurable post-processing rules,
        outputting results in a specified JSON-like format.

        This method first calls the base `predict()` method to get initial model outputs
        (sentence causality, spans, and relations). Then, it applies a sequence of
        optional heuristic rules to refine these predictions.

        Parameters
        ----------
        original_text : str
            The raw input sentence string. This is used for the "text" field in the output.
        encoding : Dict[str, torch.Tensor]
            A dictionary returned by a Hugging Face tokenizer, containing at least
            "input_ids" and "attention_mask" for the `original_text`.
            Expected shape for tensors: (1, seq_len).
        tokenizer : PreTrainedTokenizerBase
            The tokenizer instance that was used to create the `encoding`.
            This is required to decode token spans back to text.
        cls_threshold : float, optional
            The probability threshold (0.0 to 1.0) for classifying a sentence as causal
            by the sentence classification head. If None, uses the value from
            `INFERENCE_CONFIG['cls_threshold']` or defaults to 0.5.
        device : str or torch.device, optional
            The device (e.g., "cpu", "cuda") on which to perform computations.
            If None, uses the globally configured `DEVICE`.
        apply_rule_promote_sentence : bool, default False
            Activates Rule 1: `PromoteSentenceToCausalIfSpansAndRelationsFound`.
            If True and the sentence is initially classified as non-causal, this rule
            attempts to promote it to causal if strong span and relation evidence exists.
        apply_rule_heuristic_connect : bool, default False
            Activates Rule 2: `HeuristicRelationConnectSingleToAll`.
            If True and the sentence is causal, this rule may change "NoRel" relation labels
            to `default_causal_relation_type` in specific "single-source" span configurations.
        apply_rule_enforce_spans_insufficient : bool, default False
            Activates Rule 3: `EnforceSentenceNonCausalIfSpansInsufficient`.
            If True and the sentence is causal, this rule may demote it to non-causal
            if the detected spans are deemed insufficient (e.g., no cause-type or no effect-type spans).
        apply_rule_enforce_no_relations : bool, default False
            Activates Rule 4: `EnforceSentenceNonCausalIfNoRelations`.
            If True and the sentence is causal with sufficient spans, this rule may
            demote it to non-causal if no actual relations (all are `no_relation_label`) are found.
        no_relation_label : str, default "NoRel"
            The string label used in your `id2label_rel` mapping that signifies "no relation".
            This is crucial for Rules 1, 2, and 4 to correctly identify non-relations.
        default_causal_relation_type : str, default "Rel_CE"
            The string label (from your `id2label_rel`) to assign when Rule 2
            (`HeuristicRelationConnectSingleToAll`) heuristically creates or modifies a relation.

        Returns
        -------
        Dict[str, Any]
            A dictionary representing the processed causal extraction result:
            {
                "text": "[exact input sentence]",
                "causal": bool,  // Final causality status after rules
                "relations": [
                    {
                        "cause": "[exact cause text string]",
                        "effect": "[exact effect text string]",
                        "type": "[relation type string, e.g., Rel_CE]"
                    },
                    // ... more relations if found and sentence is causal
                ]
            }
            If "causal" is false, "relations" will be an empty list.

        Post-Processing Rules Logic (Applied in order):
        -------------------------------------------------
        1.  **`PromoteSentenceToCausalIfSpansAndRelationsFound` (Rule 1):**
            -   If `apply_rule_promote_sentence` is True:
            -   **Condition:** Sentence initially predicted as non-causal by the model.
            -   **Action:** If sufficient cause-type and effect-type spans are found, AND at least one
                model-predicted relation is NOT `no_relation_label`, the sentence's causality status
                is promoted to "causal".
            -   **Purpose:** Corrects false negatives from the sentence classifier using strong downstream evidence.

        2.  **`HeuristicRelationConnectSingleToAll` (Rule 2):**
            -   If `apply_rule_heuristic_connect` is True and sentence is currently "causal":
            -   **Condition:** Exactly one cause-type span exists with one or more effect-type spans
                (or vice-versa).
            -   **Action:** For pairs involving this "single source" span, if the model predicted
                `no_relation_label`, their relation label is changed to `default_causal_relation_type`.
            -   **Purpose:** Captures obvious relations in simple one-to-many or many-to-one structures.

        3.  **`EnforceSentenceNonCausalIfSpansInsufficient` (Rule 3):**
            -   If `apply_rule_enforce_spans_insufficient` is True and sentence is currently "causal":
            -   **Condition:** No cause-type spans are found, OR no effect-type spans are found,
                OR no spans are found at all. (A cause-type span can be 'C' or 'CE'; an effect-type can be 'E' or 'CE').
            -   **Action:** The sentence's causality status is demoted to "non-causal".
            -   **Purpose:** Ensures a minimal level of evidence (both cause and effect components based on BIO tagging).

        4.  **`EnforceSentenceNonCausalIfNoRelations` (Rule 4):**
            -   If `apply_rule_enforce_no_relations` is True and sentence is currently "causal" (and presumed to have sufficient spans):
            -   **Condition:** After considering model predictions and Rule 2, if all identified
                relations for candidate pairs still have the `no_relation_label`.
            -   **Action:** The sentence's causality status is demoted to "non-causal".
            -   **Purpose:** Enforces that a causal sentence must have identifiable links.

        5.  **`FinalizeOutputsBasedOnSentenceCausality` (Implicit Final Output Formatting Logic):**
            -   **Logic:** This step is always performed to structure the output. Based on the final determined
                causality status (after all active rules have been applied):
                - If "non-causal": The output "relations" list is an empty list.
                - If "causal": The "relations" list is populated with actual relations (where label is not
                  `no_relation_label` AND both cause and effect text are non-empty and valid).
                  If this filtering results in no actual relations being formatted,
                  the causality status is set to "non-causal" as a final consistency check.
            -   **Purpose:** Ensures the output structure is consistent with the final causal decision and data quality.
        """
        current_device = device if device is not None else DEVICE
        self.to(current_device) 
        self.eval()

        initial_pred = self.predict(encoding, cls_threshold=cls_threshold, device=current_device)
        
        current_is_causal: bool = initial_pred["is_causal"]
        current_spans: Dict[str, List[Span]] = initial_pred["spans"]
        current_relations: List[Dict] = initial_pred["relations"]

        # Rule 1: PromoteSentenceToCausalIfSpansAndRelationsFound
        if apply_rule_promote_sentence and not current_is_causal:
            has_cause_source = bool(current_spans.get("cause")) or bool(current_spans.get("ce"))
            has_effect_source = bool(current_spans.get("effect")) or bool(current_spans.get("ce"))
            if has_cause_source and has_effect_source:
                if any(r["label"] != no_relation_label for r in current_relations):
                    current_is_causal = True
        
        # Rule 2: HeuristicRelationConnectSingleToAll
        if apply_rule_heuristic_connect and current_is_causal:
            all_potential_cause_spans = current_spans.get("cause", []) + current_spans.get("ce", [])
            all_potential_effect_spans = current_spans.get("effect", []) + current_spans.get("ce", [])
            is_single_cause_source = (len(all_potential_cause_spans) == 1 and len(all_potential_effect_spans) >= 1)
            is_single_effect_source = (len(all_potential_effect_spans) == 1 and len(all_potential_cause_spans) >= 1)

            if is_single_cause_source or is_single_effect_source:
                modified_relations_list = []
                for rel_candidate in current_relations:
                    is_heuristic_target = False
                    # Check if the relation involves the "single" source and a "multiple" counterpart
                    if is_single_cause_source:
                        # arg1_span is the single cause, arg2_span is one of the effects
                        if rel_candidate["arg1_role"] == "cause" and rel_candidate["arg1_span"] == all_potential_cause_spans[0] and \
                           rel_candidate["arg2_role"] == "effect" and rel_candidate["arg2_span"] in all_potential_effect_spans:
                            is_heuristic_target = True
                    elif is_single_effect_source:
                         # arg2_span is the single effect, arg1_span is one of the causes
                         if rel_candidate["arg2_role"] == "effect" and rel_candidate["arg2_span"] == all_potential_effect_spans[0] and \
                            rel_candidate["arg1_role"] == "cause" and rel_candidate["arg1_span"] in all_potential_cause_spans:
                            is_heuristic_target = True
                    
                    if is_heuristic_target and rel_candidate["label"] == no_relation_label:
                        modified_rel = rel_candidate.copy()
                        modified_rel["label"] = default_causal_relation_type
                        modified_relations_list.append(modified_rel)
                    else:
                        modified_relations_list.append(rel_candidate)
                current_relations = modified_relations_list
        
        # Rule 3: EnforceSentenceNonCausalIfSpansInsufficient
        if apply_rule_enforce_spans_insufficient and current_is_causal:
            # This rule checks for the *existence of BIO-tagged span types*.
            # It does not check the textual content of these spans.
            has_cause_source_tag = bool(current_spans.get("cause")) or bool(current_spans.get("ce"))
            has_effect_source_tag = bool(current_spans.get("effect")) or bool(current_spans.get("ce"))
            if not (has_cause_source_tag and has_effect_source_tag): # Must have at least one of each type based on BIO tags
                current_is_causal = False
            elif not any(s_list for s_list in current_spans.values()): # No BIO-tagged spans at all
                 current_is_causal = False

        # Rule 4: EnforceSentenceNonCausalIfNoRelations
        if apply_rule_enforce_no_relations and current_is_causal:
            actual_relations_found = [r for r in current_relations if r["label"] != no_relation_label]
            if not actual_relations_found:
                current_is_causal = False

        # Rule 5: FinalizeOutputsBasedOnSentenceCausality (Formatting and final consistency check based on text quality)
        output_formatted_relations = []
        if current_is_causal: 
            squeezed_input_ids = encoding["input_ids"].squeeze(0) if encoding["input_ids"].ndim > 1 else encoding["input_ids"]
            at_least_one_fully_valid_textual_relation_found = False 

            for rel_data in current_relations:
                if rel_data["label"] != no_relation_label: 
                    cause_text_str = self._get_text_from_span(squeezed_input_ids, rel_data["arg1_span"], tokenizer)
                    effect_text_str = self._get_text_from_span(squeezed_input_ids, rel_data["arg2_span"], tokenizer)
                    
                    if "[Error: Invalid Span Indices]" in cause_text_str or \
                       "[Error: Invalid Span Indices]" in effect_text_str:
                        continue
                    
                    # A relation is only considered valid if BOTH its cause and effect text are non-empty after stripping,
                    # AND each must be at least 2 characters long.
                    if cause_text_str.strip() and len(cause_text_str.strip()) >= 2 and \
                       effect_text_str.strip() and len(effect_text_str.strip()) >= 2:
                        output_formatted_relations.append({
                            "cause": cause_text_str,
                            "effect": effect_text_str,
                            "type": rel_data["label"]
                        })
                        at_least_one_fully_valid_textual_relation_found = True
            
            if not at_least_one_fully_valid_textual_relation_found:
                 current_is_causal = False

        return {
            "text": original_text,
            "causal": current_is_causal,
            "relations": output_formatted_relations if current_is_causal else [],
        }

    @staticmethod
    def _bio_to_spans(tags: List[int], attention_mask_1d: Optional[torch.Tensor] = None) -> Dict[str, List[Span]]:
        spans_dict = {"cause": [], "effect": [], "ce": []}
        effective_length = len(tags)
        if attention_mask_1d is not None:
            try:
                # Ensure attention_mask_1d is a 1D tensor of integers or booleans
                if attention_mask_1d.ndim > 1: # Should already be squeezed by caller if from batch
                    attention_mask_1d = attention_mask_1d.squeeze(0)
                if attention_mask_1d.dtype != torch.bool: # Ensure boolean for sum if it's 0/1
                     attention_mask_1d = attention_mask_1d.bool()
                true_seq_len = torch.sum(attention_mask_1d).item()
                effective_length = min(len(tags), int(true_seq_len))
            except Exception as e: 
                # print(f"Warning: Could not process attention_mask_1d in _bio_to_spans: {e}. Using full tags length: {len(tags)}")
                pass
        i = 0
        while i < effective_length: 
            label_str = id2label_bio.get(tags[i], "O")
            if label_str.startswith("B-"): 
                role_type = label_str[2:] 
                start_index = i
                i += 1
                while i < effective_length and id2label_bio.get(tags[i], "O") == f"I-{role_type}":
                    i += 1
                end_index = i - 1
                if role_type == "C":
                    spans_dict["cause"].append((start_index, end_index))
                elif role_type == "E":
                    spans_dict["effect"].append((start_index, end_index))
                elif role_type == "CE": 
                    spans_dict["ce"].append((start_index, end_index))
            else: 
                i += 1
        return spans_dict

    @staticmethod
    def _generate_candidate_pairs(spans_data: Dict[str, List[Span]]) -> List[Tuple[str, Span, str, Span]]:
        candidate_pairs: List[Tuple[str, Span, str, Span]] = []
        cause_spans = spans_data.get("cause", [])
        effect_spans = spans_data.get("effect", [])
        ce_spans = spans_data.get("ce", [])
        for c_span in cause_spans:
            for e_span in effect_spans:
                candidate_pairs.append(("cause", c_span, "effect", e_span))
        for ce_as_c_span in ce_spans:
            for e_span in effect_spans:
                candidate_pairs.append(("cause", ce_as_c_span, "effect", e_span))
        for c_span in cause_spans:
            for ce_as_e_span in ce_spans:
                candidate_pairs.append(("cause", c_span, "effect", ce_as_e_span))
        if len(ce_spans) >= 2: 
            for i in range(len(ce_spans)):
                for j in range(len(ce_spans)):
                    if i == j: 
                        continue
                    candidate_pairs.append(("cause", ce_spans[i], "effect", ce_spans[j]))
        seen_span_pairs = set()
        unique_candidate_pairs: List[Tuple[str, Span, str, Span]] = []
        for role1, span1, role2, span2 in candidate_pairs:
            pair_key = (span1, span2) # Key based on the tuple of span indices
            if pair_key not in seen_span_pairs:
                seen_span_pairs.add(pair_key)
                unique_candidate_pairs.append((role1, span1, role2, span2))
        return unique_candidate_pairs

# ---------------------------------------------------------------------------
# Minimal sanity test & sample training loop runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from transformers import AutoTokenizer
    import json

    if not id2label_rel: 
        print("CRITICAL: id2label_rel is empty. Populating with minimal defaults for testing.")
        id2label_rel.update({0: "NoRel", 1: "Rel_CE", 2: "Rel_EC"}) 
        label2id_rel.update({v:k for k,v in id2label_rel.items()})
    if not id2label_bio:
        print("CRITICAL: id2label_bio is empty. Populating with minimal defaults for testing.")
        id2label_bio.update({0: "O", 1: "B-C", 2: "I-C", 3: "B-E", 4: "I-E", 5: "B-CE", 6: "I-CE"})
        label2id_bio.update({v:k for k,v in id2label_bio.items()})

    actual_no_rel_label = "NoRel" 
    if 0 in id2label_rel and id2label_rel[0].lower() == "norel": 
        actual_no_rel_label = id2label_rel[0]
    elif "NoRel" not in label2id_rel: 
        for k_rel, v_rel in id2label_rel.items():
            if "no" in v_rel.lower() or "none" in v_rel.lower(): # Broader check for NoRel
                actual_no_rel_label = v_rel
                print(f"Info: Using '{actual_no_rel_label}' as no_relation_label based on content.")
                break
        else: 
             print(f"Warning: 'NoRel' not clearly identified in id2label_rel. Using default '{actual_no_rel_label}'.")

    actual_default_causal_label = "Rel_CE" 
    found_causal_rel_for_default = False
    for k_rel, v_rel in id2label_rel.items():
        if v_rel != actual_no_rel_label and "no" not in v_rel.lower() and "none" not in v_rel.lower(): 
            actual_default_causal_label = v_rel 
            found_causal_rel_for_default = True
            break
    if not found_causal_rel_for_default:
         print(f"Warning: No clear causal relation type found for heuristic. Defaulting to '{actual_default_causal_label}'.")

    tokenizer_name = MODEL_CONFIG.get("encoder_name", "bert-base-uncased")
    tok = AutoTokenizer.from_pretrained(tokenizer_name)

    test_sentences = [
        "The heavy rain caused the river to flood.", 
        "Housing prices fell because interest rates rose.", 
        "The plant is green.", 
        "The new policy led to increased efficiency, which boosted profits.", 
        "The server crashed.",
        "The evidence suggests a correlation, but no direct causation was established.", # Non-causal by cls, but might have C/E like terms
        "A led to B and B resulted in C." # Chained
    ]
    
    sample_text_for_rules_test = test_sentences[0] 
    # sample_text_for_rules_test = test_sentences[2] # "The plant is green." - good for testing Rule 1 (Promote) if model says non-causal
    # sample_text_for_rules_test = test_sentences[4] # "The server crashed." - good for testing Rule 3 (Insufficient Spans)

    print(f"\n--- Testing Rule Combinations for: \"{sample_text_for_rules_test}\" ---")
    encoding = tok(sample_text_for_rules_test, return_tensors="pt")

    model_under_test = JointCausalModel(use_crf=False,
                                 num_bio_labels=len(id2label_bio), 
                                 num_rel_labels=len(id2label_rel))
    model_under_test.to(DEVICE)

    rule_configurations = [
        {"name": "All Rules OFF (Current Default)", "flags": {}}, # Default is all False
        {"name": "Rule 1 (Promote Sentence) ON", "flags": {"apply_rule_promote_sentence": True}},
        {"name": "Rule 2 (Heuristic Connect) ON", "flags": {"apply_rule_heuristic_connect": True}},
        {"name": "Rule 3 (Enforce Spans Insufficient) ON", "flags": {"apply_rule_enforce_spans_insufficient": True}},
        {"name": "Rule 4 (Enforce No Relations) ON", "flags": {"apply_rule_enforce_no_relations": True}},
        {"name": "Rules 3 & 4 ON", "flags": {"apply_rule_enforce_spans_insufficient": True, "apply_rule_enforce_no_relations": True}},
        {"name": "All Rules ON", "flags": {
            "apply_rule_promote_sentence": True,
            "apply_rule_heuristic_connect": True,
            "apply_rule_enforce_spans_insufficient": True,
            "apply_rule_enforce_no_relations": True
        }},
    ]

    for config_test in rule_configurations:
        print(f"\nTesting Configuration: {config_test['name']}")
        base_params = {
            "original_text": sample_text_for_rules_test,
            "encoding": encoding,
            "tokenizer": tok,
            "device": DEVICE,
            "no_relation_label": actual_no_rel_label,
            "default_causal_relation_type": actual_default_causal_label
        }
        current_params = {**base_params, **config_test["flags"]}
        prediction = model_under_test.predict_with_rules(**current_params)
        print(json.dumps(prediction, indent=2))

    print("\n--- Sanity Check for All Test Sentences (All Rules OFF by Default) ---")
    for idx, sample_text in enumerate(test_sentences):
        print(f"\nSentence {idx+1}: \"{sample_text}\" (Rules OFF by default)")
        encoding = tok(sample_text, return_tensors="pt")
        # Re-initialize model for each sentence if testing untrained behavior consistently
        # model_instance = JointCausalModel(use_crf=True, num_bio_labels=len(id2label_bio), num_rel_labels=len(id2label_rel))
        # model_instance.to(DEVICE)
        prediction_all_off = model_under_test.predict_with_rules( # Using same model_under_test instance
            original_text=sample_text,
            encoding=encoding,
            tokenizer=tok,
            device=DEVICE,
            no_relation_label=actual_no_rel_label,
            default_causal_relation_type=actual_default_causal_label
        )
        print(json.dumps(prediction_all_off, indent=2))

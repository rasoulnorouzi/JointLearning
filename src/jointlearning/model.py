"""Joint Causal Extraction Model (softmax)
============================================================================

A PyTorch module for joint causal extraction using softmax decoding for BIO tagging.
The model supports class weights for handling imbalanced data.

```python
>>> model = JointCausalModel()        # softmax-based model
```

---------------------------------------------------------------------------
Usage overview
---------------------------------------------------------------------------

**Training**
~~~~~~~~~~~~
(Training code example omitted for brevity, see previous versions)
---------------------------------------------------------------------------
Implementation
---------------------------------------------------------------------------
"""
from __future__ import annotations
from typing import Dict, Tuple, Optional, Any, List
import torch
import torch.nn as nn
from transformers import AutoModel
from huggingface_hub import PyTorchModelHubMixin
# Assuming config.py is in the same directory or accessible via PYTHONPATH
try:
    # Attempt relative import for package structure
    from .config import MODEL_CONFIG, id2label_bio, id2label_rel, id2label_cls
except ImportError:
    # Fallback for standalone script execution (if config.py is present)
    from config import MODEL_CONFIG, id2label_bio, id2label_rel, id2label_cls


# ---------------------------------------------------------------------------
# Type aliases & label maps
# ---------------------------------------------------------------------------
Span = Tuple[int, int]  # inclusive indices (token indices)
label2id_bio = {v: k for k, v in id2label_bio.items()}
label2id_rel = {v: k for k, v in id2label_rel.items()}
label2id_cls = {v: k for k, v in id2label_cls.items()}


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------
class JointCausalModel(nn.Module, PyTorchModelHubMixin):
    """Encoder + three heads with **optional CRF** BIO decoder.

    This model integrates a pre-trained transformer encoder with three distinct
    heads for:
    1. Classification (cls_head): Predicts a global label for the input.
    2. BIO tagging (bio_head): Performs sequence tagging using BIO scheme.
       Can operate with a CRF layer or standard softmax.
    3. Relation extraction (rel_head): Identifies relations between entities
       detected by the BIO tagging head.
    """

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
    ) -> None:
        """Initializes the JointCausalModel."""
        super().__init__()

        self.encoder_name = encoder_name
        self.num_cls_labels = num_cls_labels
        self.num_bio_labels = num_bio_labels
        self.num_rel_labels = num_rel_labels
        self.dropout_rate = dropout
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
        self._init_new_layer_weights()

        # Store label maps for easy access
        self.id2label_bio = id2label_bio
        self.id2label_rel = id2label_rel
        self.id2label_cls = id2label_cls


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
        """Initializes the weights of the newly added linear layers."""
        for mod in [self.cls_head, self.bio_head, self.rel_head]:
            for sub_module in mod.modules():
                if isinstance(sub_module, nn.Linear):
                    nn.init.xavier_uniform_(sub_module.weight)
                    if sub_module.bias is not None:
                        nn.init.zeros_(sub_module.bias)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encodes the input using the transformer model."""
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
        """Performs a forward pass through the model."""
        hidden = self.encode(input_ids, attention_mask)
        cls_logits = self.cls_head(hidden[:, 0])
        emissions  = self.bio_head(hidden)
        tag_loss: Optional[torch.Tensor] = None
        if bio_labels is not None: tag_loss = torch.tensor(0.0, device=emissions.device)

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
            "cls_logits": cls_logits, "bio_emissions": emissions, "tag_loss": tag_loss, 
            "rel_logits": rel_logits, "hidden_states": hidden
        }

    # ------------------------------------------------------------------
    # Prediction & Helper Methods
    # ------------------------------------------------------------------

    def _align_and_get_word_tags(self, token_bio_ids: List[int], token_attention_mask: List[int], 
                                 offset_mapping: List[Tuple[int, int]], text: str) -> List[Dict[str, Any]]:
        """Aligns token BIO tags to words using 'first subword' & offset_mapping."""
        word_level_tags_with_info = []
        current_word_tokens_indices = []
        current_word_first_token_bio_id = -1
        current_word_char_start = -1
        prev_token_end_char = -1

        for token_idx, ((offset_start, offset_end), bio_id_num) in enumerate(zip(offset_mapping, token_bio_ids)):
            if token_attention_mask[token_idx] == 0: continue # Skip padding
            
            is_special_token = offset_start == 0 and offset_end == 0
            is_new_word = True # Assume new unless proven otherwise

            if is_special_token:
                if current_word_tokens_indices: # Finalize word before special token
                    actual_end_char = offset_mapping[current_word_tokens_indices[-1]][1]
                    word_level_tags_with_info.append({
                        'text': text[current_word_char_start:actual_end_char],
                        'tag': self.id2label_bio[current_word_first_token_bio_id],
                        'start_char': current_word_char_start, 'end_char': actual_end_char,
                        'start_token_idx': current_word_tokens_indices[0], 
                        'end_token_idx': current_word_tokens_indices[-1]
                    })
                    current_word_tokens_indices = []
                prev_token_end_char = offset_end 
                continue # Skip processing special token further

            if current_word_tokens_indices:
                if offset_start == prev_token_end_char: is_new_word = False # Is subword if starts immediately after prev

            if is_new_word and current_word_tokens_indices: # Finalize previous word
                actual_end_char = offset_mapping[current_word_tokens_indices[-1]][1]
                word_level_tags_with_info.append({
                    'text': text[current_word_char_start:actual_end_char],
                    'tag': self.id2label_bio[current_word_first_token_bio_id],
                    'start_char': current_word_char_start, 'end_char': actual_end_char,
                    'start_token_idx': current_word_tokens_indices[0], 
                    'end_token_idx': current_word_tokens_indices[-1]
                })
                current_word_tokens_indices = []
            
            if not current_word_tokens_indices: # Start a new word
                current_word_first_token_bio_id = bio_id_num
                current_word_char_start = offset_start
            
            current_word_tokens_indices.append(token_idx)
            prev_token_end_char = offset_end

        if current_word_tokens_indices: # Finalize any remaining word
            actual_end_char = offset_mapping[current_word_tokens_indices[-1]][1]
            word_level_tags_with_info.append({
                'text': text[current_word_char_start:actual_end_char],
                'tag': self.id2label_bio[current_word_first_token_bio_id],
                'start_char': current_word_char_start, 'end_char': actual_end_char,
                'start_token_idx': current_word_tokens_indices[0], 
                'end_token_idx': current_word_tokens_indices[-1]
            })
        return word_level_tags_with_info

    def _correct_bio_word_sequence(self, word_tags_list: List[str]) -> List[str]:
        """Corrects invalid I- tags in a BIO sequence."""
        corrected_tags = []
        for i, tag in enumerate(word_tags_list):
            if tag.startswith("I-"):
                entity_type = tag.split("-", 1)[1]
                if i == 0: corrected_tags.append(f"B-{entity_type}")
                else:
                    prev_tag = corrected_tags[-1]
                    if prev_tag == "O" or (prev_tag[2:] != entity_type):
                        corrected_tags.append(f"B-{entity_type}")
                    else: corrected_tags.append(tag)
            else: corrected_tags.append(tag)
        return corrected_tags

    def _finalize_span(self, span_tokens_info: List[Dict], label: str, original_text: str) -> Dict[str, Any]:
        """Creates a span dictionary from a list of word info dicts."""
        return {
            'label': label,
            'text': original_text[span_tokens_info[0]['start_char']:span_tokens_info[-1]['end_char']],
            'start_char': span_tokens_info[0]['start_char'],
            'end_char': span_tokens_info[-1]['end_char'],
            'start_token_idx': span_tokens_info[0]['start_token_idx'],
            'end_token_idx': span_tokens_info[-1]['end_token_idx']
        }

    def _group_word_tags_to_spans(self, word_level_tags_with_info: List[Dict], original_text: str) -> List[Dict[str, Any]]:
        """Groups word-level BIO tags into spans (C, E, CE)."""
        spans = []
        current_span_word_info_list = []
        current_span_label_type = None

        for i, word_info in enumerate(word_level_tags_with_info):
            tag = word_info['tag']
            prefix = "O"
            entity_type_from_tag = None
            if tag != "O":
                try: prefix, entity_type_from_tag = tag.split("-", 1)
                except ValueError: prefix = "O"

            if prefix == "B":
                if current_span_word_info_list: spans.append(self._finalize_span(current_span_word_info_list, current_span_label_type, original_text))
                current_span_word_info_list = [word_info]
                current_span_label_type = entity_type_from_tag
            elif prefix == "I" and current_span_word_info_list and entity_type_from_tag == current_span_label_type:
                current_span_word_info_list.append(word_info)
            else: # O-tag or invalid/mismatched I-tag
                if current_span_word_info_list: spans.append(self._finalize_span(current_span_word_info_list, current_span_label_type, original_text))
                current_span_word_info_list = []
                current_span_label_type = None
                if prefix == "I": # Mismatched I-tag, treat as B-tag
                    current_span_word_info_list = [word_info]
                    current_span_label_type = entity_type_from_tag
        
        if current_span_word_info_list: spans.append(self._finalize_span(current_span_word_info_list, current_span_label_type, original_text))
        return spans

    def predict_batch(
        self,
        texts: List[str],
        tokenized_batch: Dict[str, torch.Tensor],
        device: str = "cpu",
        use_heuristic: bool = False,
        override_cls_if_spans_found: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Performs batch prediction with optional heuristics and CLS override.

        This method processes a batch of texts to extract causal relations. 
        It involves several steps:
        1.  **Classification:** Predicts if each text is causal or not.
        2.  **BIO Tagging:** Identifies Cause (C), Effect (E), and Cause+Effect (CE) spans.
        3.  **Span Processing:** Aligns tokens to words, validates BIO tags, 
            and groups them into spans. Handles CE spans.
        4.  **Mandatory Span Check:** Ensures at least one Cause and one Effect 
            (potential or actual) exist; otherwise, marks as non-causal.
        5.  **Relation Extraction:**
            * **Heuristic (Optional):** If `use_heuristic` is True and the 
                sentence has exactly one Cause (and >=1 Effect) or exactly 
                one Effect (and >=1 Cause), it assumes all possible pairs are 
                related. **Crucially, in this mode, CE spans *can* be paired 
                with themselves.**
            * **Standard (`rel_head`):** If heuristics aren't used or don't 
                apply, it uses the trained `rel_head` to predict relations between 
                all valid C-E pairs. **In this mode, CE spans *cannot* be paired 
                with themselves.**
        6.  **CLS Override (Optional):** If `override_cls_if_spans_found` is True, 
            a 'non-causal' classification can be changed to 'causal' if valid 
            spans and relations are ultimately found.
        7.  **Output Formatting:** Returns results in the specified JSON format.

        Args:
            texts: List of raw input text strings.
            tokenized_batch: A dictionary from Hugging Face tokenizer (batched).
            device: 'cpu' or 'cuda'.
            use_heuristic: Enable heuristic relation extraction for simple cases.
            override_cls_if_spans_found: Enable overriding initial CLS prediction.

        Returns:
            A list of prediction dictionaries.
        """
        self.eval()
        self.to(device)

        b_input_ids = tokenized_batch['input_ids'].to(device)
        b_attention_mask = tokenized_batch['attention_mask'].to(device)
        offset_mapping_batch_tensor = tokenized_batch['offset_mapping']

        batch_size = b_input_ids.size(0)
        results = []

        with torch.no_grad():
            outputs_pass1 = self.forward(input_ids=b_input_ids, attention_mask=b_attention_mask)
            hidden_states_batch = outputs_pass1["hidden_states"] 
            cls_logits_batch = outputs_pass1["cls_logits"]
            bio_emissions_batch = outputs_pass1["bio_emissions"]

        for i in range(batch_size):
            text = texts[i]
            current_offset_mapping = offset_mapping_batch_tensor[i].cpu().tolist()
            cls_logits = cls_logits_batch[i]
            bio_emissions = bio_emissions_batch[i]
            current_hidden_states = hidden_states_batch[i]
            current_attention_mask_list = b_attention_mask[i].cpu().tolist()

            # --- 1. Classification ---
            predicted_cls_id = torch.argmax(cls_logits).item()
            cls_label = self.id2label_cls.get(predicted_cls_id, "non-causal")
            is_causal_pred_initial = (cls_label == "causal")

            # Early exit ONLY if initial is non-causal AND override is OFF
            if not is_causal_pred_initial and not override_cls_if_spans_found:
                results.append({"text": text, "causal": False, "relations": []})
                continue

            # --- 2 & 3. BIO Tagging and Span Processing ---
            predicted_bio_tag_ids_list = torch.argmax(bio_emissions, dim=-1).cpu().tolist()
            word_level_tags = self._align_and_get_word_tags(
                predicted_bio_tag_ids_list, current_attention_mask_list, current_offset_mapping, text
            )
            if not word_level_tags:
                 results.append({"text": text, "causal": False, "relations": []})
                 continue

            corrected_tags = self._correct_bio_word_sequence([item['tag'] for item in word_level_tags])
            for idx, item in enumerate(word_level_tags): item['tag'] = corrected_tags[idx]
            extracted_spans = self._group_word_tags_to_spans(word_level_tags, text)

            potential_causes, potential_effects = [], []
            span_id_counter = 0
            for span in extracted_spans:
                span['id'] = span_id_counter
                span_id_counter += 1
                if span['label'] == "C": potential_causes.append(span)
                elif span['label'] == "E": potential_effects.append(span)
                elif span['label'] == "CE":
                    potential_causes.append({**span, 'original_label': 'CE'}) 
                    potential_effects.append({**span, 'original_label': 'CE'})

            # --- 4. Mandatory Span Check ---
            if not (potential_causes and potential_effects):
                results.append({"text": text, "causal": False, "relations": []})
                continue
            
            # --- 5. Relation Extraction ---
            relations = []
            heuristic_applied = False

            if use_heuristic:
                unique_cause_ids = set(s['id'] for s in potential_causes)
                unique_effect_ids = set(s['id'] for s in potential_effects)
                Nc, Ne = len(unique_cause_ids), len(unique_effect_ids)

                if Nc >= 1 and Ne == 1:
                    effect_span = next(s for s in potential_effects if s['id'] == list(unique_effect_ids)[0])
                    for cause_span in potential_causes:
                        # Heuristic *allows* self-pairing: No check needed here.
                        relations.append({"cause": cause_span['text'], "effect": effect_span['text'], "type": "Rel_CE"})
                    heuristic_applied = bool(relations)
                
                elif Nc == 1 and Ne >= 1:
                    cause_span = next(s for s in potential_causes if s['id'] == list(unique_cause_ids)[0])
                    for effect_span in potential_effects:
                        # Heuristic *allows* self-pairing: No check needed here.
                        relations.append({"cause": cause_span['text'], "effect": effect_span['text'], "type": "Rel_CE"})
                    heuristic_applied = bool(relations)

            if not heuristic_applied: # Standard rel_head path
                rel_pairs_to_evaluate = []
                for c_span in potential_causes:
                    for e_span in potential_effects:
                        # Standard path *prevents* CE self-pairing.
                        if c_span.get('original_label') == 'CE' and e_span.get('original_label') == 'CE' and c_span['id'] == e_span['id']:
                            continue
                        rel_pairs_to_evaluate.append((c_span, e_span))
                
                if rel_pairs_to_evaluate:
                    all_c_vecs, all_e_vecs = [], []
                    seq_len = current_hidden_states.size(0)
                    pos_rel_base = torch.arange(seq_len, device=device)
                    expanded_hidden_states = current_hidden_states.unsqueeze(0) 

                    for c_span, e_span in rel_pairs_to_evaluate:
                        c_start, c_end = max(0, c_span['start_token_idx']), min(seq_len - 1, c_span['end_token_idx'])
                        e_start, e_end = max(0, e_span['start_token_idx']), min(seq_len - 1, e_span['end_token_idx'])
                        c_mask = ((c_start <= pos_rel_base) & (pos_rel_base <= c_end)).unsqueeze(0).unsqueeze(2)
                        e_mask = ((e_start <= pos_rel_base) & (pos_rel_base <= e_end)).unsqueeze(0).unsqueeze(2)
                        c_vec = (expanded_hidden_states * c_mask).sum(1) / c_mask.sum(1).clamp(min=1)
                        e_vec = (expanded_hidden_states * e_mask).sum(1) / e_mask.sum(1).clamp(min=1)
                        all_c_vecs.append(c_vec)
                        all_e_vecs.append(e_vec)
                    
                    if all_c_vecs:
                        pair_rel_logits_batch = self.rel_head(torch.cat([torch.cat(all_c_vecs, dim=0), torch.cat(all_e_vecs, dim=0)], dim=1))
                        predicted_rel_ids_batch = torch.argmax(pair_rel_logits_batch, dim=-1)
                        for pair_idx, rel_id in enumerate(predicted_rel_ids_batch):
                            if self.id2label_rel.get(rel_id.item()) == "Rel_CE":
                                c_s, e_s = rel_pairs_to_evaluate[pair_idx]
                                relations.append({"cause": c_s['text'], "effect": e_s['text'], "type": "Rel_CE"})
            
            # --- 6 & 7. Final Output Formatting ---
            # If we have relations, it's causal (handles override). Otherwise, non-causal.
            if not relations: 
                results.append({"text": text, "causal": False, "relations": []})
            else:
                results.append({"text": text, "causal": True, "relations": relations})
        
        return results


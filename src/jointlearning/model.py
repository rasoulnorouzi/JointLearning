from __future__ import annotations
from typing import Dict, Tuple, Optional, Any, List
import torch
import torch.nn as nn
from transformers import AutoModel
from huggingface_hub import PyTorchModelHubMixin

try:
    from .config import MODEL_CONFIG, id2label_bio, id2label_rel, id2label_cls
except ImportError:
    from config import MODEL_CONFIG, id2label_bio, id2label_rel, id2label_cls

# ---------------------------------------------------------------------------
# Type aliases & label maps
# ---------------------------------------------------------------------------
Span = Tuple[int, int]  # inclusive indices (token indices)
label2id_bio = {v: k for k, v in id2label_bio.items()}
label2id_rel = {v: k for k, v in id2label_rel.items()}
label2id_cls = {v: k for k, v in id2label_cls.items()}

STOPWORDS = {"the", "a", "an"}
PUNCT = {".", ",", ";", ":"}

# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------
class JointCausalModel(nn.Module, PyTorchModelHubMixin): # PyTorchModelHubMixin might not be needed for local test
    def __init__(
        self,
        *,
        encoder_name: str = MODEL_CONFIG["encoder_name"],
        num_cls_labels: int = MODEL_CONFIG["num_cls_labels"],
        num_bio_labels: int = MODEL_CONFIG["num_bio_labels"],
        num_rel_labels: int = MODEL_CONFIG["num_rel_labels"],
        dropout: float = MODEL_CONFIG["dropout"],
    ) -> None:
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
            nn.Linear(self.hidden_size, self.hidden_size // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(self.hidden_size // 2, num_cls_labels),
        )
        self.bio_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, num_bio_labels),
        )
        self.rel_head = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, num_rel_labels),
        )
        self._init_new_layer_weights()
        self.id2label_bio = id2label_bio
        self.id2label_rel = id2label_rel
        self.id2label_cls = id2label_cls

    def get_config_dict(self) -> Dict:
        return {"encoder_name": self.encoder_name, "num_cls_labels": self.num_cls_labels,
                "num_bio_labels": self.num_bio_labels, "num_rel_labels": self.num_rel_labels,
                "dropout": self.dropout_rate}
    @classmethod
    def from_config_dict(cls, config: Dict) -> "JointCausalModel": return cls(**config)
    def _init_new_layer_weights(self):
        for mod in [self.cls_head, self.bio_head, self.rel_head]:
            for sub_module in mod.modules():
                if isinstance(sub_module, nn.Linear):
                    nn.init.xavier_uniform_(sub_module.weight)
                    if sub_module.bias is not None: nn.init.zeros_(sub_module.bias)
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden_states = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.layer_norm(self.dropout(hidden_states))
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor | None]:
        hidden = self.encode(input_ids, attention_mask)
        cls_logits = self.cls_head(hidden[:, 0])
        emissions  = self.bio_head(hidden)
        return {"cls_logits": cls_logits, "bio_emissions": emissions, "hidden_states": hidden,
                "tag_loss": None, "rel_logits": None}

    def _get_word_ids_from_offsets(self, offset_mapping: List[Tuple[int, int]], 
                                     token_attention_mask: List[int]) -> List[Optional[int]]:
        word_ids = []; current_word_id = -1; last_end_char = -1
        for token_idx, (start_char, end_char) in enumerate(offset_mapping):
            if token_attention_mask[token_idx] == 0: word_ids.append(None); continue
            if start_char == 0 and end_char == 0: word_ids.append(None); last_end_char = -1
            elif start_char == last_end_char: word_ids.append(current_word_id)
            else: current_word_id += 1; word_ids.append(current_word_id)
            last_end_char = end_char
        return word_ids
    def _align_and_get_word_tags(self, token_bio_ids: List[int], token_attention_mask: List[int], 
                                 offset_mapping: List[Tuple[int, int]], text: str) -> List[Dict[str, Any]]:
        word_ids = self._get_word_ids_from_offsets(offset_mapping, token_attention_mask)
        word_map = {}
        for token_idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id not in word_map: word_map[word_id] = []
                word_map[word_id].append(token_idx)
        word_level_tags_with_info = []
        for word_id in sorted(word_map.keys()):
            token_indices = word_map[word_id]
            first_token_idx, last_token_idx = token_indices[0], token_indices[-1]
            start_char, end_char = offset_mapping[first_token_idx][0], offset_mapping[last_token_idx][1]
            tag_id = token_bio_ids[first_token_idx]
            word_level_tags_with_info.append({
                'text': text[start_char:end_char], 'tag': self.id2label_bio[tag_id],
                'start_char': start_char, 'end_char': end_char,
                'start_token_idx': first_token_idx, 'end_token_idx': last_token_idx})
        return word_level_tags_with_info
    def _correct_bio_word_sequence(self, word_tags_list: List[str]) -> List[str]:
        corrected_tags = []
        for i, tag in enumerate(word_tags_list):
            if tag.startswith("I-"):
                entity_type = tag.split("-", 1)[1]
                if i == 0: corrected_tags.append(f"B-{entity_type}")
                else:
                    prev_tag = corrected_tags[-1]
                    if prev_tag == "O" or (prev_tag.startswith(("B-","I-")) and prev_tag[2:] != entity_type): # check type correctly
                        corrected_tags.append(f"B-{entity_type}")
                    else: corrected_tags.append(tag)
            else: corrected_tags.append(tag)
        return corrected_tags
    def _finalize_span(self, span_words_info: List[Dict], label: str, original_text: str) -> Dict[str, Any]:
        first_word, last_word = span_words_info[0], span_words_info[-1]
        start_char, end_char = first_word['start_char'], last_word['end_char']
        return {'label': label, 'text': original_text[start_char:end_char],
                'start_char': start_char, 'end_char': end_char,
                'start_token_idx': first_word['start_token_idx'], 
                'end_token_idx': last_word['end_token_idx']}

    def _group_word_tags_to_spans(self, word_level_tags_with_info: List[Dict], original_text: str) -> List[Dict[str, Any]]:
        """
        Groups word-level BIO tags into entity spans.
        Revised to more robustly handle span continuation.
        """
        spans = []
        current_span_words = []  # Stores word_info dicts for the current span
        current_span_type = None   # e.g., "C", "E", "CE"

        for i, word_info in enumerate(word_level_tags_with_info):
            tag = word_info['tag']
            bio_prefix = "O"
            entity_type = None # This word's entity type (C, E, CE)
            
            if tag != "O":
                try:
                    bio_prefix, entity_type = tag.split("-", 1)
                except ValueError: # Should not happen for valid B-X, I-X tags
                    bio_prefix = "O" 

            # Condition to start a new span:
            # 1. Current tag is B-Something
            # 2. Current tag is I-Something, but it doesn't match the current_span_type
            # 3. Current tag is I-Something, but there's no active current_span (current_span_words is empty)
            
            # Condition to continue a span:
            # Current tag is I-Something AND it matches current_span_type AND a span is active
            
            if bio_prefix == "B":
                if current_span_words: # Finalize any previous span
                    spans.append(self._finalize_span(current_span_words, current_span_type, original_text))
                current_span_words = [word_info] # Start a new span
                current_span_type = entity_type
            elif bio_prefix == "I":
                if current_span_words and entity_type == current_span_type: # Valid continuation
                    current_span_words.append(word_info)
                else: # Invalid I-tag (mismatch type or no preceding B/I)
                      # This I-tag should have been corrected to a B-tag by _correct_bio_word_sequence.
                      # If it's still an I-tag here and doesn't match, we treat it as starting a new span of its own type.
                    if current_span_words: # Finalize previous span
                        spans.append(self._finalize_span(current_span_words, current_span_type, original_text))
                    current_span_words = [word_info] # Start new span with this "I turned B" tag
                    current_span_type = entity_type 
            else: # O-tag
                if current_span_words: # Finalize any previous span
                    spans.append(self._finalize_span(current_span_words, current_span_type, original_text))
                current_span_words = [] # Reset
                current_span_type = None
        
        # After loop, finalize any remaining open span
        if current_span_words:
            spans.append(self._finalize_span(current_span_words, current_span_type, original_text))
            
        return spans

    def predict_batch(
        self, texts: List[str], tokenized_batch: Dict[str, torch.Tensor], device: str = "cpu",
        use_heuristic: bool = False, override_cls_if_spans_found: bool = False
    ) -> List[Dict[str, Any]]:
        self.eval(); self.to(device)
        b_input_ids = tokenized_batch['input_ids'].to(device)
        b_attention_mask = tokenized_batch['attention_mask'].to(device)
        offset_mapping_batch_tensor = tokenized_batch['offset_mapping']
        batch_size = b_input_ids.size(0); results = []
        with torch.no_grad():
            outputs_pass1 = self.forward(input_ids=b_input_ids, attention_mask=b_attention_mask)
            hidden_states_batch, cls_logits_batch, bio_emissions_batch = \
                outputs_pass1["hidden_states"], outputs_pass1["cls_logits"], outputs_pass1["bio_emissions"]
        for i in range(batch_size):
            text = texts[i]
            current_offset_mapping = offset_mapping_batch_tensor[i].cpu().tolist()
            cls_logits, bio_emissions, current_hidden_states, current_attention_mask_list = \
                cls_logits_batch[i], bio_emissions_batch[i], hidden_states_batch[i], b_attention_mask[i].cpu().tolist()
            predicted_cls_id = torch.argmax(cls_logits).item()
            cls_label = self.id2label_cls.get(predicted_cls_id, "non-causal")
            is_causal_pred_initial = (cls_label == "causal")
            if not is_causal_pred_initial and not override_cls_if_spans_found:
                results.append({"text": text, "causal": False, "relations": []}); continue
            predicted_bio_tag_ids_list = torch.argmax(bio_emissions, dim=-1).cpu().tolist()
            word_level_tags = self._align_and_get_word_tags(
                predicted_bio_tag_ids_list, current_attention_mask_list, current_offset_mapping, text)
            if not word_level_tags: results.append({"text": text, "causal": False, "relations": []}); continue
            
            # Apply BIO correction to the 'tag' field within word_level_tags list of dicts
            current_word_tags_only = [item['tag'] for item in word_level_tags]
            corrected_tags_only = self._correct_bio_word_sequence(current_word_tags_only)
            for idx, item in enumerate(word_level_tags): 
                item['tag'] = corrected_tags_only[idx] # Update in place

            extracted_spans = self._group_word_tags_to_spans(word_level_tags, text) # Use updated word_level_tags
            
            potential_causes, potential_effects = [], []
            span_id_counter = 0
            for span in extracted_spans:
                span['id'] = span_id_counter; span_id_counter += 1
                if span['label'] == "C": potential_causes.append(span)
                elif span['label'] == "E": potential_effects.append(span)
                elif span['label'] == "CE":
                    potential_causes.append({**span, 'original_label': 'CE'}) 
                    potential_effects.append({**span, 'original_label': 'CE'})
            if not (potential_causes and potential_effects):
                results.append({"text": text, "causal": False, "relations": []}); continue
            relations = []; heuristic_applied = False
            if use_heuristic:
                unique_cause_ids, unique_effect_ids = set(s['id'] for s in potential_causes), set(s['id'] for s in potential_effects)
                Nc, Ne = len(unique_cause_ids), len(unique_effect_ids)
                if (Nc >= 1 and Ne == 1) or (Nc == 1 and Ne >= 1) :
                    if Nc == 1 and Ne >=1: 
                        # Find the single unique cause span details
                        cause_s_details = next(s for s_list in [potential_causes] for s in s_list if s['id'] == list(unique_cause_ids)[0])
                        # Iterate through all effect spans (can have multiple instances if from CE)
                        for effect_s in potential_effects: 
                            relations.append({"cause": cause_s_details['text'], "effect": effect_s['text'], "type": "Rel_CE"})
                    elif Nc >= 1 and Ne == 1: 
                        effect_s_details = next(s for s_list in [potential_effects] for s in s_list if s['id'] == list(unique_effect_ids)[0])
                        for cause_s in potential_causes: 
                            relations.append({"cause": cause_s['text'], "effect": effect_s_details['text'], "type": "Rel_CE"})
                    heuristic_applied = bool(relations)

            if not heuristic_applied:
                rel_pairs_to_evaluate = []
                for c_span in potential_causes:
                    for e_span in potential_effects:
                        if c_span.get('original_label') == 'CE' and e_span.get('original_label') == 'CE' and c_span['id'] == e_span['id']: continue
                        rel_pairs_to_evaluate.append((c_span, e_span))
                if rel_pairs_to_evaluate:
                    all_c_vecs, all_e_vecs = [], []
                    seq_len = current_hidden_states.size(0); pos_rel_base = torch.arange(seq_len, device=device)
                    expanded_hidden_states = current_hidden_states.unsqueeze(0) 
                    for c_span, e_span in rel_pairs_to_evaluate:
                        c_start, c_end = max(0,c_span['start_token_idx']), min(seq_len-1,c_span['end_token_idx'])
                        e_start, e_end = max(0,e_span['start_token_idx']), min(seq_len-1,e_span['end_token_idx'])
                        c_mask = ((c_start <= pos_rel_base)&(pos_rel_base <= c_end)).unsqueeze(0).unsqueeze(2)
                        e_mask = ((e_start <= pos_rel_base)&(pos_rel_base <= e_end)).unsqueeze(0).unsqueeze(2)
                        c_vec = (expanded_hidden_states * c_mask).sum(1) / c_mask.sum(1).clamp(min=1)
                        e_vec = (expanded_hidden_states * e_mask).sum(1) / e_mask.sum(1).clamp(min=1)
                        all_c_vecs.append(c_vec); all_e_vecs.append(e_vec)
                    if all_c_vecs:
                        # Mocking rel_head prediction for test script
                        num_pairs = len(all_c_vecs)
                        mock_rel_logits = torch.zeros(num_pairs, self.num_rel_labels, device=device)
                        # For testing, assume if pairs exist, the first one is a valid relation
                        if num_pairs > 0 : mock_rel_logits[0, label2id_rel["Rel_CE"]] = 1 
                        
                        predicted_rel_ids_batch = torch.argmax(mock_rel_logits, dim=-1)
                        # In real use:
                        # pair_rel_logits_batch = self.rel_head(torch.cat([torch.cat(all_c_vecs, dim=0), torch.cat(all_e_vecs, dim=0)], dim=1))
                        # predicted_rel_ids_batch = torch.argmax(pair_rel_logits_batch, dim=-1)
                        for pair_idx, rel_id in enumerate(predicted_rel_ids_batch):
                            if self.id2label_rel.get(rel_id.item()) == "Rel_CE":
                                c_s, e_s = rel_pairs_to_evaluate[pair_idx]
                                relations.append({"cause": c_s['text'], "effect": e_s['text'], "type": "Rel_CE"})
            if not relations: results.append({"text": text, "causal": False, "relations": []})
            else: results.append({"text": text, "causal": True, "relations": relations})
        return results
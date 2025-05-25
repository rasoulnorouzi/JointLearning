from __future__ import annotations
import torch
import torch.nn as nn
import string
import traceback
import types # Added import for types.MethodType
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Tuple, Optional, Any, List
from huggingface_hub import PyTorchModelHubMixin # Ensure this is imported

try:
    from .config import MODEL_CONFIG, id2label_bio, id2label_rel, id2label_cls
except ImportError:
    from config import MODEL_CONFIG, id2label_bio, id2label_rel, id2label_cls

# --- Configuration (Simplified from config.py for this test script) ---
id2label_bio = {
    0: "B-C", 1: "I-C", 2: "B-E", 3: "I-E",
    4: "B-CE", 5: "I-CE", 6: "O"
}
label2id_bio = {v: k for k, v in id2label_bio.items()}
id2label_rel = {0: "Rel_None", 1: "Rel_CE"}
label2id_rel = {v: k for k, v in id2label_rel.items()}
id2label_cls = {0: "non-causal", 1: "causal"}
label2id_cls = {v: k for k, v in id2label_cls.items()}

MODEL_CONFIG = {
    "encoder_name": "bert-base-uncased",
    "num_cls_labels": 2,
    "num_bio_labels": 7,
    "num_rel_labels": 2,
    "dropout": 0.1,
}


# --- JointCausalModel Class ---
class JointCausalModel(nn.Module, PyTorchModelHubMixin):
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
            elif start_char == last_end_char and last_end_char != -1 : # Ensure last_end_char was valid
                word_ids.append(current_word_id)
            else: current_word_id += 1; word_ids.append(current_word_id)
            last_end_char = end_char
        return word_ids
        
    def _align_and_get_word_tags(self, token_bio_ids: List[int], token_attention_mask: List[int], 
                                 offset_mapping: List[Tuple[int, int]], text: str) -> List[Dict[str, Any]]:
        word_ids = self._get_word_ids_from_offsets(offset_mapping, token_attention_mask)
        word_map = {}
        for token_idx, word_id in enumerate(word_ids):
            if word_id is not None: # Process only tokens part of a word
                if word_id not in word_map: word_map[word_id] = []
                word_map[word_id].append(token_idx)
        
        word_level_tags_with_info = []
        if not word_map: return word_level_tags_with_info # Handle cases with no words found

        for word_id in sorted(word_map.keys()): # Ensure words are processed in order
            token_indices = word_map[word_id]
            if not token_indices: continue # Should not happen if word_id exists in map

            first_token_idx = token_indices[0]
            last_token_idx = token_indices[-1]
            
            # Ensure offsets are valid before slicing
            start_char_offset, end_char_offset = offset_mapping[first_token_idx][0], offset_mapping[last_token_idx][1]
            
            # Ensure start_char is not negative and end_char is not beyond text length (or less than start_char)
            start_char = max(0, start_char_offset)
            end_char = min(len(text), end_char_offset)
            if start_char >= end_char and not (start_char == 0 and end_char == 0 and first_token_idx == last_token_idx) : # Allow (0,0) for special tokens if they become a word (should not happen)
                 actual_text = "" # Or handle as an error / skip
            else:
                actual_text = text[start_char:end_char]

            tag_id = token_bio_ids[first_token_idx] # First subword strategy
            
            word_level_tags_with_info.append({
                'text': actual_text, 
                'tag': self.id2label_bio.get(tag_id, "O"), # Default to O if tag_id is invalid
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
                    if prev_tag == "O" or (prev_tag.startswith(("B-","I-")) and prev_tag[2:] != entity_type):
                        corrected_tags.append(f"B-{entity_type}")
                    else: corrected_tags.append(tag)
            else: corrected_tags.append(tag)
        return corrected_tags

    def _finalize_span(self, span_words_info: List[Dict], label: str, original_text: str) -> Dict[str, Any]:
        if not span_words_info: # Should not happen if called correctly
            return {'label': label, 'text': "", 'start_char': -1, 'end_char': -1, 
                    'start_token_idx': -1, 'end_token_idx': -1}

        first_word, last_word = span_words_info[0], span_words_info[-1]
        start_char, end_char = first_word['start_char'], last_word['end_char']
        span_text = original_text[start_char:end_char]
        
        trailing_punctuation = string.punctuation
        # Strip only if span_text is not empty
        while span_text and span_text[-1] in trailing_punctuation:
            span_text = span_text[:-1]
            # end_char = max(start_char, end_char -1) # Optional: adjust end_char for stripped text

        return {'label': label, 'text': span_text,
                'start_char': start_char, 'end_char': end_char, # end_char refers to original offset
                'start_token_idx': first_word['start_token_idx'], 
                'end_token_idx': last_word['end_token_idx']}

    def _group_word_tags_to_spans(self, word_level_tags_with_info: List[Dict], original_text: str) -> List[Dict[str, Any]]:
        spans = []
        current_span_words = [] 
        current_span_type = None  

        for i, word_info in enumerate(word_level_tags_with_info):
            tag = word_info['tag']
            bio_prefix = "O"
            entity_type = None 
            
            if tag != "O" and "-" in tag: # Ensure tag is in B-X or I-X format
                try:
                    bio_prefix, entity_type = tag.split("-", 1)
                except ValueError:
                    bio_prefix = "O" # Fallback for malformed tags
            
            if bio_prefix == "B":
                if current_span_words: 
                    spans.append(self._finalize_span(current_span_words, current_span_type, original_text))
                current_span_words = [word_info] 
                current_span_type = entity_type
            elif bio_prefix == "I":
                if current_span_words and entity_type == current_span_type: 
                    current_span_words.append(word_info)
                else: # I-tag is inconsistent with current span or no active span
                    if current_span_words: 
                        spans.append(self._finalize_span(current_span_words, current_span_type, original_text))
                    # This I-tag should have been corrected to B-tag by _correct_bio_word_sequence.
                    # Start a new span with this word, assuming it's now effectively a B-tag.
                    current_span_words = [word_info] 
                    current_span_type = entity_type 
            else: # O-tag
                if current_span_words: 
                    spans.append(self._finalize_span(current_span_words, current_span_type, original_text))
                current_span_words = [] 
                current_span_type = None
        
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
            current_word_tags_only = [item['tag'] for item in word_level_tags]
            corrected_tags_only = self._correct_bio_word_sequence(current_word_tags_only)
            for idx, item in enumerate(word_level_tags): item['tag'] = corrected_tags_only[idx]
            extracted_spans = self._group_word_tags_to_spans(word_level_tags, text)
            potential_causes, potential_effects = [], []
            span_id_counter = 0
            for span in extracted_spans:
                span['id'] = span_id_counter; span_id_counter += 1
                if span.get('label') == "C": potential_causes.append(span) # Use .get for safety
                elif span.get('label') == "E": potential_effects.append(span)
                elif span.get('label') == "CE":
                    potential_causes.append({**span, 'original_label': 'CE'}) 
                    potential_effects.append({**span, 'original_label': 'CE'})
            if not (potential_causes and potential_effects):
                results.append({"text": text, "causal": False, "relations": []}); continue
            relations = []; heuristic_applied = False
            if use_heuristic:
                unique_cause_ids, unique_effect_ids = set(s['id'] for s in potential_causes), set(s['id'] for s in potential_effects)
                Nc, Ne = len(unique_cause_ids), len(unique_effect_ids)
                if (Nc >= 1 and Ne == 1) or (Nc == 1 and Ne >= 1) :
                    if Nc == 1 and Ne >=1 and unique_cause_ids and potential_causes : 
                        cause_s_id = list(unique_cause_ids)[0]
                        cause_s_details = next((s for s in potential_causes if s['id'] == cause_s_id), None)
                        if cause_s_details:
                            for effect_s in potential_effects: 
                                relations.append({"cause": cause_s_details['text'], "effect": effect_s['text'], "type": "Rel_CE"})
                    elif Nc >= 1 and Ne == 1 and unique_effect_ids and potential_effects: 
                        effect_s_id = list(unique_effect_ids)[0]
                        effect_s_details = next((s for s in potential_effects if s['id'] == effect_s_id), None)
                        if effect_s_details:
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
                        num_pairs = len(all_c_vecs)
                        mock_rel_logits = torch.zeros(num_pairs, self.num_rel_labels, device=device)
                        if num_pairs > 0 : mock_rel_logits[0, label2id_rel["Rel_CE"]] = 1 
                        predicted_rel_ids_batch = torch.argmax(mock_rel_logits, dim=-1)
                        for pair_idx, rel_id in enumerate(predicted_rel_ids_batch):
                            if self.id2label_rel.get(rel_id.item()) == "Rel_CE":
                                c_s, e_s = rel_pairs_to_evaluate[pair_idx]
                                relations.append({"cause": c_s['text'], "effect": e_s['text'], "type": "Rel_CE"})
            if not relations: results.append({"text": text, "causal": False, "relations": []})
            else: results.append({"text": text, "causal": True, "relations": relations})
        return results

# --- Mocking Utilities (Revised) ---
def get_mock_forward_fn(tokenized_inputs_for_mock: Dict[str, torch.Tensor], 
                        test_case_mock_data: Dict[str, Any], 
                        tokenizer_for_mock: AutoTokenizer,
                        device="cpu"):
    def mock_forward(self_arg_ignored_by_mock, input_ids, attention_mask, **kwargs):
        batch_size, seq_len = input_ids.shape
        cls_id = test_case_mock_data["cls_id"]
        mock_cls_logits = torch.full((batch_size, MODEL_CONFIG["num_cls_labels"]), -10.0, device=device)
        mock_cls_logits[0, cls_id] = 10.0
        mock_bio_emissions = torch.full((batch_size, seq_len, MODEL_CONFIG["num_bio_labels"]), -10.0, device=device)
        mock_bio_emissions[:, :, label2id_bio["O"]] = 5.0 
        word_bio_ids = test_case_mock_data["bio_token_ids_for_words"]
        current_word_bio_idx = 0
        for token_pos in range(seq_len):
            if attention_mask[0, token_pos] == 0: continue
            current_token_id_val = input_ids[0, token_pos].item()
            if current_token_id_val == tokenizer_for_mock.cls_token_id or \
               current_token_id_val == tokenizer_for_mock.sep_token_id:
                continue
            if current_word_bio_idx < len(word_bio_ids):
                bio_id_for_this_token = word_bio_ids[current_word_bio_idx]
                mock_bio_emissions[0, token_pos, bio_id_for_this_token] = 10.0
                current_word_bio_idx += 1
        mock_hidden_states = torch.randn(batch_size, seq_len, 768, device=device)
        return {"cls_logits": mock_cls_logits, "bio_emissions": mock_bio_emissions, "hidden_states": mock_hidden_states}
    return mock_forward

# --- Test Scenarios (bio_token_ids_for_words are for content tokens) ---
test_cases = [
    {
        "name": "Scenario 1: Simple Non-Causal",
        "text": "The sky is blue.",
        "mock_data": {"cls_id": 0, "bio_token_ids_for_words": [6,6,6,6,6]},
        "settings": {"use_heuristic": False, "override_cls_if_spans_found": False},
        "expected_causal": False, "expected_relations_count": 0
    },
    {
        "name": "Scenario 2: Simple Causal (C -> E) - Heuristic",
        "text": "Heavy rain caused the flood.",
        "mock_data": {"cls_id": 1, "bio_token_ids_for_words": [0,1,6,6,2,6]},
        "settings": {"use_heuristic": True, "override_cls_if_spans_found": False},
        "expected_causal": True, "expected_relations_count": 1,
        "expected_relations_texts": [("Heavy rain", "flood")]
    },
    {
        "name": "Scenario 3: Single CE Span - Heuristic (CE Self-Pair)",
        "text": "The drought was the problem.",
        "mock_data": {"cls_id": 1, "bio_token_ids_for_words": [4,5,6,6,6,6]},
        "settings": {"use_heuristic": True, "override_cls_if_spans_found": False},
        "expected_causal": True, "expected_relations_count": 1,
        "expected_relations_texts": [("The drought", "The drought")]
    },
    {
        "name": "Scenario 4: Single CE Span - Standard (No CE Self-Pair)",
        "text": "The drought was the problem.",
        "mock_data": {"cls_id": 1, "bio_token_ids_for_words": [4,5,6,6,6,6]},
        "settings": {"use_heuristic": False, "override_cls_if_spans_found": False},
        "expected_causal": False, "expected_relations_count": 0
    },
    {
        "name": "Scenario 5: CLS Override - Non-Causal to Causal",
        "text": "Stress leads to burnout.",
        "mock_data": {"cls_id": 0, "bio_token_ids_for_words": [0,6,6,2,6]},
        "settings": {"use_heuristic": False, "override_cls_if_spans_found": True},
        "expected_causal": True, "expected_relations_count": 1,
        "expected_relations_texts": [("Stress", "burnout")]
    },
    {
        "name": "Scenario 6: CLS Override - Still Non-Causal (No Spans)",
        "text": "A quiet day.",
        "mock_data": {"cls_id": 0, "bio_token_ids_for_words": [6,6,6,6]},
        "settings": {"use_heuristic": False, "override_cls_if_spans_found": True},
        "expected_causal": False, "expected_relations_count": 0
    },
    {
        "name": "Scenario 7: Multiple Causes, One Effect - Heuristic",
        "text": "Heat and lack of water caused crops to fail.",
        "mock_data": {"cls_id": 1, 
                      "bio_token_ids_for_words": [
                          label2id_bio["B-C"], label2id_bio["O"], label2id_bio["B-C"], 
                          label2id_bio["I-C"], label2id_bio["I-C"], label2id_bio["O"], 
                          label2id_bio["B-E"], label2id_bio["I-E"], label2id_bio["I-E"], 
                          label2id_bio["O"]  
                      ]},
        "settings": {"use_heuristic": True, "override_cls_if_spans_found": False},
        "expected_causal": True, "expected_relations_count": 2,
        "expected_relations_texts": [("Heat", "crops to fail"), ("lack of water", "crops to fail")]
    },
     {
        "name": "Scenario 8: 'Bad food' as Cause",
        "text": "Bad food made sick.", 
        "mock_data": {"cls_id": 1, 
                      "bio_token_ids_for_words": [
                          label2id_bio["B-C"], label2id_bio["I-C"], 
                          label2id_bio["O"],   
                          label2id_bio["B-E"], 
                          label2id_bio["O"]    
                      ]},
        "settings": {"use_heuristic": True, "override_cls_if_spans_found": False},
        "expected_causal": True, "expected_relations_count": 1,
        "expected_relations_texts": [("Bad food", "sick")]
    },
    {
        "name": "Scenario 9: Social Science - Abstract C->E (Heuristic)",
        "text": "Increased social media usage often leads to decreased real-world interaction.",
        "mock_data": {"cls_id": 1,
                      "bio_token_ids_for_words": [
                          label2id_bio["B-C"], label2id_bio["I-C"], label2id_bio["I-C"], label2id_bio["I-C"], 
                          label2id_bio["O"], label2id_bio["O"], label2id_bio["O"], 
                          label2id_bio["B-E"], label2id_bio["I-E"], label2id_bio["I-E"], 
                          label2id_bio["O"] 
                      ]},
        "settings": {"use_heuristic": True, "override_cls_if_spans_found": False},
        "expected_causal": True, "expected_relations_count": 1,
        "expected_relations_texts": [("Increased social media usage", "decreased real-world interaction")]
    },
    {
        "name": "Scenario 10: Social Science - Non-Causal",
        "text": "The study explored various demographic factors.",
        "mock_data": {"cls_id": 0, "bio_token_ids_for_words": [6,6,6,6,6,6,6]},
        "settings": {"use_heuristic": False, "override_cls_if_spans_found": False},
        "expected_causal": False, "expected_relations_count": 0
    },
    {
        "name": "Scenario 11: Social Science - Complex (Standard Rel Head Mock)",
        "text": "Policy changes and economic downturn resulted in widespread unemployment and increased poverty levels.",
        "mock_data": {"cls_id": 1,
                      "bio_token_ids_for_words": [
                          label2id_bio["B-C"], label2id_bio["I-C"], 
                          label2id_bio["O"],                       
                          label2id_bio["B-C"], label2id_bio["I-C"], 
                          label2id_bio["O"], label2id_bio["O"],    
                          label2id_bio["B-E"], label2id_bio["I-E"], 
                          label2id_bio["O"],                       
                          label2id_bio["B-E"], label2id_bio["I-E"], label2id_bio["I-E"], 
                          label2id_bio["O"]                        
                      ]},
        "settings": {"use_heuristic": False, "override_cls_if_spans_found": False},
        "expected_causal": True, 
        "expected_relations_count": 1, 
        "expected_relations_texts": [("Policy changes", "widespread unemployment")]
    },
    {
        "name": "Scenario 12: Social Science - Intervening Variable (Heuristic, 1 C, 2 E)",
        "text": "in the unequal endowment treatment, we again find a positive effect of being at the same time poorer than the group average and being a high-endowment player (model 3).",
        "mock_data": {"cls_id": 1,
                      "bio_token_ids_for_words": [
                          label2id_bio["B-C"], label2id_bio["I-C"], label2id_bio["I-C"], label2id_bio["I-C"], 
                          label2id_bio["O"], 
                          label2id_bio["B-E"], label2id_bio["I-E"], 
                          label2id_bio["O"], label2id_bio["O"], label2id_bio["O"], label2id_bio["O"], 
                          label2id_bio["B-E"], label2id_bio["I-E"], label2id_bio["I-E"], 
                          label2id_bio["O"] 
                      ]},
        "settings": {"use_heuristic": True, "override_cls_if_spans_found": False},
        "expected_causal": True, "expected_relations_count": 2,
        "expected_relations_texts": [
            ("Early childhood education programs", "cognitive skills"),
            ("Early childhood education programs", "long-term academic achievement")
        ]
    },
    {
        "name": "Scenario 13: Social Science - CE span with multiple relations (Standard)",
        "text": "The implementation of the new curriculum (CE) led to increased student engagement (E1) and also improved teacher satisfaction (E2).",
        "mock_data": {"cls_id": 1,
                      "bio_token_ids_for_words": [
                          label2id_bio["B-CE"], label2id_bio["I-CE"], label2id_bio["I-CE"], label2id_bio["I-CE"], label2id_bio["I-CE"], label2id_bio["I-CE"], 
                          label2id_bio["O"], label2id_bio["O"], 
                          label2id_bio["B-E"], label2id_bio["I-E"], label2id_bio["I-E"], 
                          label2id_bio["O"], label2id_bio["O"], label2id_bio["O"], 
                          label2id_bio["B-E"], label2id_bio["I-E"], 
                          label2id_bio["O"] 
                      ]},
        "settings": {"use_heuristic": False, "override_cls_if_spans_found": False},
        "expected_causal": True, 
        "expected_relations_count": 1, 
        "expected_relations_texts": [("The implementation of the new curriculum", "increased student engagement")]
    },
    {
        "name": "Scenario 14: Social Science - No clear C/E spans",
        "text": "This paper reviews the literature on organizational change.",
        "mock_data": {"cls_id": 0, "bio_token_ids_for_words": [6]*8},
        "settings": {"use_heuristic": False, "override_cls_if_spans_found": False},
        "expected_causal": False, "expected_relations_count": 0
    },
    {
        "name": "Scenario 15: User's Sentence (Targeted Mock for correct extraction)",
        "text": "the relatively high frequency of reports at 50 may be driven by risk aversion, since the use of the quadratic scoring rule makes it risky to report extreme values.",
        # Content tokens (approx 31, depends on exact tokenization of "50", "quadratic", etc.):
        # E: "the relatively high frequency of reports at 50" (8 tokens)
        # C: "risk aversion" (2 tokens)
        "mock_data": {"cls_id": 1,
                      "bio_token_ids_for_words": [ # Total 31 content tokens
                          label2id_bio["B-E"], label2id_bio["I-E"], label2id_bio["I-E"], label2id_bio["I-E"], # the relatively high frequency
                          label2id_bio["I-E"], label2id_bio["I-E"], label2id_bio["I-E"], label2id_bio["I-E"], # of reports at 50
                          label2id_bio["O"], label2id_bio["O"], label2id_bio["O"], label2id_bio["O"], # may be driven by
                          label2id_bio["B-C"], label2id_bio["I-C"], # risk aversion
                          label2id_bio["O"], # ,
                          label2id_bio["O"], label2id_bio["O"], label2id_bio["O"], label2id_bio["O"], # since the use of
                          label2id_bio["O"], label2id_bio["O"], label2id_bio["O"], label2id_bio["O"], # the quadratic scoring rule
                          label2id_bio["O"], label2id_bio["O"], label2id_bio["O"], label2id_bio["O"], # makes it risky to
                          label2id_bio["O"], label2id_bio["O"], label2id_bio["O"], label2id_bio["O"]  # report extreme values .
                      ]},
        "settings": {"use_heuristic": False, "override_cls_if_spans_found": False},
        "expected_causal": True,
        "expected_relations_count": 1,
        "expected_relations_texts": [("risk aversion", "the relatively high frequency of reports at 50")]
    },
    {
        "name": "Scenario 16: Social Science - Complex sentence with CE and other entities",
        "text": "it is also considered that the process could easily be adapted for virtual delivery, thus increasing its accessibility.",
        # CE: The economic stimulus
        # E1: consumer spending
        # C1: external market volatility
        # E2: marginal growth
        # Expected mock relations: (CE as C -> E1), (C1 -> E2), (CE as C -> E2) -- mock rel_head will pick one
        "mock_data": {"cls_id": 1,
                      "bio_token_ids_for_words": [
                          label2id_bio["B-CE"], label2id_bio["I-CE"], label2id_bio["I-CE"], # The economic stimulus
                          label2id_bio["O"], label2id_bio["O"], label2id_bio["O"], # (CE) aimed to boost
                          label2id_bio["B-E"], label2id_bio["I-E"], # consumer spending
                          label2id_bio["O"], # ,
                          label2id_bio["O"], # but
                          label2id_bio["B-C"], label2id_bio["I-C"], label2id_bio["I-C"],# external market volatility
                          label2id_bio["O"], # suppressed
                          label2id_bio["B-E"],# these # This is tricky, could be I-E if part of "these effects"
                                              # For simplicity, let's make "effects" separate or O
                          label2id_bio["I-E"],# effects
                          label2id_bio["O"], # ,
                          label2id_bio["O"], # leading
                          label2id_bio["O"], # to
                          label2id_bio["O"], # only
                          label2id_bio["B-E"], label2id_bio["I-E"], # marginal growth
                          label2id_bio["O"]  # .
                      ]},
        "settings": {"use_heuristic": False, "override_cls_if_spans_found": False},
        "expected_causal": True,
        "expected_relations_count": 1, # Mock rel_head limitation
        "expected_relations_texts": [("The economic stimulus", "consumer spending")] # Example of first C-E pair
    }
]

# --- Main Test Execution ---
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["encoder_name"])
        # model_path = "/home/rnorouzini/JointLearning/src/jointlearning/expert_bert_softmax/expert_bert_softmax_model.pt"
        model_path = "/home/rnorouzini/JointLearning/src/jointlearning/expert_bert_softmax/expert_bert_softmax_model.pt"
        model = JointCausalModel(**MODEL_CONFIG)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        # model.load_state_dict(torch.load(model_path, map_location=device)) # No need to load real weights for mock testing
        model.eval()
        model.to(device)
    except Exception as e:
        print(f"Error initializing model/tokenizer: {e}")
        exit()

    all_tests_passed = True
    for tc_idx, tc in enumerate(test_cases):
        print(f"\n--- Running Test: {tc['name']} ---")
        texts_batch = [tc["text"]]
        
        tokenized_inputs = tokenizer(
            texts_batch, return_tensors="pt", padding="max_length",
            max_length=72, # Increased max_length further for very long sentences
            truncation=True, return_offsets_mapping=True
        )
        
        # --- Verification of mock data alignment (optional but recommended) ---
        input_ids_list_for_debug = tokenized_inputs['input_ids'][0].tolist()
        tokens_for_debug = tokenizer.convert_ids_to_tokens(input_ids_list_for_debug)
        content_tokens_for_debug = []
        for k_debug in range(tokenized_inputs['input_ids'].shape[1]):
            if tokenized_inputs['attention_mask'][0, k_debug] == 0: continue
            token_id_val = tokenized_inputs['input_ids'][0, k_debug].item()
            if token_id_val == tokenizer.cls_token_id or token_id_val == tokenizer.sep_token_id:
                continue
            content_tokens_for_debug.append(tokens_for_debug[k_debug])
        
        if len(tc['mock_data']['bio_token_ids_for_words']) != len(content_tokens_for_debug):
            print(f"  DEBUG WARNING for {tc['name']}: Mismatch in mock bio_ids length ({len(tc['mock_data']['bio_token_ids_for_words'])}) and actual content tokens ({len(content_tokens_for_debug)}).")
            print(f"  Actual content tokens ({len(content_tokens_for_debug)}): {content_tokens_for_debug}")
            print(f"  Mock BIO IDs ({len(tc['mock_data']['bio_token_ids_for_words'])}): {tc['mock_data']['bio_token_ids_for_words']}")
        # --- End Verification ---

        original_forward = model.forward
        unbound_mock_fn = get_mock_forward_fn(tokenized_inputs, tc["mock_data"], tokenizer, device=device)
        model.forward = types.MethodType(unbound_mock_fn, model)

        try:
            predictions = model.predict_batch(
                texts_batch, tokenized_inputs, device=device, **tc["settings"]
            )
            result = predictions[0]
            print(f"  Predicted Output: {result}")
            assert result["causal"] == tc["expected_causal"], f"Causal flag. Expected {tc['expected_causal']}, Got {result['causal']}"
            assert len(result["relations"]) == tc["expected_relations_count"], f"Relations count. Expected {tc['expected_relations_count']}, Got {len(result['relations'])}"
            if "expected_relations_texts" in tc:
                extracted_rel_texts = sorted([(r["cause"], r["effect"]) for r in result["relations"]])
                expected_rel_texts = sorted(tc["expected_relations_texts"])
                assert extracted_rel_texts == expected_rel_texts, f"Relation texts. Expected {expected_rel_texts}, Got {extracted_rel_texts}"
            print(f"  Test PASSED!")
        except AssertionError as e: 
            print(f"  Test FAILED: {e}")
            all_tests_passed = False
        except Exception as e: 
            print(f"  Test ERRORED: {e}"); import traceback; traceback.print_exc()
            all_tests_passed = False
        finally: 
            model.forward = original_forward

    if all_tests_passed:
        print("\n\nALL TESTS PASSED SUCCESSFULLY!")
    else:
        print("\n\nSOME TESTS FAILED OR ERRORED.")


# %%
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import json
import pandas as pd
import random
import re # For checking punctuation in random spans
from functools import partial

# --- Configuration & Label Mappings --- (Assuming these are defined globally or passed appropriately)
id2label_span = {0: "B-C", 1: "I-C", 2: "B-E", 3: "I-E", 4: "B-CE", 5: "I-CE", 6: "O"}
label2id_span = {v: k for k, v in id2label_span.items()}
entity_label_to_bio_prefix = {"cause": "C", "effect": "E", "internal_CE": "CE", "non-causal": "O"}
NO_RELATION_LABEL_STR = "Rel_None"
POSITIVE_RELATION_TYPE_TO_ID = {"Rel_CE": 1, "Rel_Zero": 2}
id2label_rel = {0: NO_RELATION_LABEL_STR, 1: "Rel_CE", 2: "Rel_Zero"}
label2id_rel = {v: k for k, v in id2label_rel.items()}
NEGATIVE_SAMPLE_REL_ID = label2id_rel[NO_RELATION_LABEL_STR]

# Helper function to check span overlap
def check_span_overlap_util(span1, span2):
    """Checks if two token spans (start_idx, end_idx) overlap."""
    return max(span1[0], span2[0]) <= min(span1[1], span2[1])

class CausalDataset(Dataset):
    """
    PyTorch Dataset class for processing causal text data.
    __getitem__ returns UNPADDED sequences. Padding is handled by the collate function.
    """
    def __init__(self, dataframe, tokenizer_name, max_length=512, negative_relation_rate=1.0, 
                 max_random_span_len=5): # Added max_random_span_len parameter
        self.dataframe = dataframe.copy()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length # Max length for TRUNCATION
        self.negative_relation_rate = negative_relation_rate
        self.max_random_span_len = max_random_span_len # Store the new parameter

        def safe_json_loads(data_str):
            try:
                if isinstance(data_str, str): return json.loads(data_str.replace("'", "\""))
                return []
            except: return []
        self.dataframe.loc[:, 'entities_parsed'] = self.dataframe['entities'].apply(safe_json_loads)
        self.dataframe.loc[:, 'relations_parsed'] = self.dataframe['relations'].apply(safe_json_loads)

    def __len__(self):
        return len(self.dataframe)

    def _get_word_indices_for_entity(self, entity_char_start, entity_char_end, token_offsets, word_ids_list):
        covered_word_indices = set()
        if entity_char_start is None or entity_char_end is None: return frozenset()
        for token_idx, (tok_char_start, tok_char_end) in enumerate(token_offsets):
            if tok_char_start == tok_char_end and tok_char_start == 0 and (token_idx >= len(word_ids_list) or word_ids_list[token_idx] is None): continue
            if tok_char_start < entity_char_end and entity_char_start < tok_char_end:
                if token_idx < len(word_ids_list):
                    word_id = word_ids_list[token_idx]
                    if word_id is not None: covered_word_indices.add(word_id)
        return frozenset(covered_word_indices)

    def _is_span_valid_for_random_neg_sampling(self, span_coords, tokens_list, word_ids_list_for_span_check, unique_candidate_arg_spans):
        s_start, s_end = span_coords
        if not (0 <= s_start <= s_end < len(tokens_list)): return False
        if any(check_span_overlap_util(span_coords, gold_span) for gold_span in unique_candidate_arg_spans):
            return False
        current_span_word_ids = [word_ids_list_for_span_check[i] for i in range(s_start, s_end + 1) if i < len(word_ids_list_for_span_check)]
        if not current_span_word_ids or any(wid is None for wid in current_span_word_ids):
            return False
        span_text = self.tokenizer.convert_tokens_to_string(tokens_list[s_start : s_end + 1]).strip()
        if not span_text: return False
        if re.fullmatch(r'[\W_]+', span_text): return False
        return True

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = str(row.get('text', '')).replace(";;", " ")
        raw_entities_data = row.get('entities_parsed', [])
        relations_data = row.get('relations_parsed', [])

        tokenized_output = self.tokenizer(
            text, max_length=self.max_length, padding=False, truncation=True,
            return_offsets_mapping=True, return_attention_mask=True
        )
        input_ids_list = tokenized_output['input_ids']
        attention_mask_list = tokenized_output['attention_mask']
        offset_mapping = tokenized_output['offset_mapping']
        word_ids_list = tokenized_output.word_ids()

        entities_with_word_spans = []
        for entity in raw_entities_data:
            if not (isinstance(entity, dict) and all(k in entity for k in ['start_offset', 'end_offset', 'id', 'label'])): continue
            word_indices_set = self._get_word_indices_for_entity(entity['start_offset'], entity['end_offset'], offset_mapping, word_ids_list)
            if word_indices_set: entities_with_word_spans.append({**entity, 'word_indices': word_indices_set})

        word_span_to_entities = {}
        for entity in entities_with_word_spans:
            key = entity['word_indices']
            if key not in word_span_to_entities: word_span_to_entities[key] = []
            word_span_to_entities[key].append(entity)

        temp_bio_entities, processed_original_ids_for_bio = [], set()
        for word_indices_key, entities_in_word_span in word_span_to_entities.items():
            if not word_indices_key: continue
            cause_entity, effect_entity = None, None
            for e in entities_in_word_span:
                if e['label'] == 'cause': cause_entity = e
                elif e['label'] == 'effect': effect_entity = e
            if cause_entity and effect_entity:
                ce_char_start = min(cause_entity['start_offset'], effect_entity['start_offset'])
                ce_char_end = max(cause_entity['end_offset'], effect_entity['end_offset'])
                temp_bio_entities.append({
                    'label': 'internal_CE', 'start_offset': ce_char_start, 'end_offset': ce_char_end,
                    'original_ids': [cause_entity['id'], effect_entity['id']],
                    'display_id': f"CE_{cause_entity['id']}_{effect_entity['id']}", 'word_indices': word_indices_key
                })
                processed_original_ids_for_bio.add(cause_entity['id']); processed_original_ids_for_bio.add(effect_entity['id'])
            else:
                for e in entities_in_word_span:
                    if e['id'] not in processed_original_ids_for_bio:
                        temp_bio_entities.append(e); processed_original_ids_for_bio.add(e['id'])

        is_causal_sentence = 0; final_bio_entities = []
        if any(e.get('label') in ['cause', 'effect', 'internal_CE'] for e in temp_bio_entities):
            is_causal_sentence = 1
            final_bio_entities = [e for e in temp_bio_entities if e.get('label') != 'non-causal']
        elif any(e.get('label') == 'non-causal' for e in temp_bio_entities):
             is_causal_sentence = 0
             final_bio_entities = [e for e in temp_bio_entities if e.get('label') == 'non-causal']

        current_sequence_length = len(input_ids_list)
        bio_labels_list = [label2id_span["O"]] * current_sequence_length
        entity_spans_for_relations = {}

        if final_bio_entities:
            for entity_to_tag in final_bio_entities:
                entity_label_str = entity_to_tag.get('label')
                start_char, end_char = entity_to_tag.get('start_offset'), entity_to_tag.get('end_offset')
                if entity_label_str is None or start_char is None or end_char is None: continue
                token_start, token_end = -1, -1
                for i, (offset_s, offset_e) in enumerate(offset_mapping):
                    if offset_s == offset_e and offset_s == 0 and (i >= len(word_ids_list) or word_ids_list[i] is None): continue
                    if offset_s < end_char and start_char < offset_e:
                        if token_start == -1: token_start = i
                        token_end = i
                if token_start != -1 :
                    current_token_end_refined = token_start
                    for i_ref in range(token_start, len(offset_mapping)):
                        offset_s_ref, offset_e_ref = offset_mapping[i_ref]
                        if offset_s_ref == offset_e_ref and offset_s_ref == 0 and (i_ref >= len(word_ids_list) or word_ids_list[i_ref] is None): continue
                        if offset_s_ref < end_char and offset_e_ref > start_char :
                            current_token_end_refined = i_ref
                        elif offset_s_ref >= end_char :
                            break
                    token_end = current_token_end_refined
                if token_start != -1 and token_end != -1 and token_start <= token_end:
                    bio_prefix = entity_label_to_bio_prefix.get(entity_label_str)
                    if bio_prefix:
                        if bio_prefix != "O" and token_start < current_sequence_length:
                            bio_labels_list[token_start] = label2id_span[f"B-{bio_prefix}"]
                            for i_bio in range(token_start + 1, min(token_end + 1, current_sequence_length)):
                                if i_bio < current_sequence_length: bio_labels_list[i_bio] = label2id_span[f"I-{bio_prefix}"]
                    current_span_for_relation = (token_start, token_end)
                    if 'original_ids' in entity_to_tag:
                        for orig_id in entity_to_tag['original_ids']: entity_spans_for_relations[orig_id] = current_span_for_relation
                    elif 'id' in entity_to_tag: entity_spans_for_relations[entity_to_tag['id']] = current_span_for_relation

        for i in range(current_sequence_length):
            if i < len(word_ids_list) and word_ids_list[i] is None:
                bio_labels_list[i] = -100

        relation_tuples = []
        if is_causal_sentence == 1:
            if relations_data:
                for rel in relations_data:
                    if not isinstance(rel, dict): continue
                    from_id, to_id, rel_type_str = rel.get('from_id'), rel.get('to_id'), rel.get('type')
                    rel_label = POSITIVE_RELATION_TYPE_TO_ID.get(rel_type_str)
                    if rel_label is not None and from_id in entity_spans_for_relations and to_id in entity_spans_for_relations:
                        c_span, e_span = entity_spans_for_relations[from_id], entity_spans_for_relations[to_id]
                        is_self_loop_from_ce = False
                        if c_span == e_span:
                            for fbe in final_bio_entities:
                                if fbe.get('label')=='internal_CE' and from_id in fbe.get('original_ids',[]) and to_id in fbe.get('original_ids',[]):
                                    is_self_loop_from_ce = True; break
                        if not is_self_loop_from_ce: relation_tuples.append((c_span, e_span, rel_label))

            unique_candidate_arg_spans = []
            temp_arg_spans = []
            for tagged_entity in final_bio_entities:
                if tagged_entity.get('label') in ['cause', 'effect', 'internal_CE']:
                    span_id_lookup = tagged_entity.get('original_ids', [tagged_entity.get('id')])[0]
                    if span_id_lookup and span_id_lookup in entity_spans_for_relations:
                        temp_arg_spans.append(entity_spans_for_relations[span_id_lookup])
            for s in temp_arg_spans:
                if s not in unique_candidate_arg_spans: unique_candidate_arg_spans.append(s)

            num_positive_relations = sum(1 for _,_,lab in relation_tuples if lab!=NEGATIVE_SAMPLE_REL_ID)
            num_negative_to_generate = int(num_positive_relations * self.negative_relation_rate)
            if num_positive_relations == 0 and unique_candidate_arg_spans and self.negative_relation_rate > 0:
                 num_negative_to_generate = max(1, int(len(unique_candidate_arg_spans) * self.negative_relation_rate * 0.5))

            generated_neg_count = 0
            if len(unique_candidate_arg_spans) >= 2 and generated_neg_count < num_negative_to_generate:
                potential_neg_pairs_s1 = []
                for i_idx in range(len(unique_candidate_arg_spans)):
                    for j_idx in range(len(unique_candidate_arg_spans)):
                        if i_idx == j_idx: continue
                        s1, s2 = unique_candidate_arg_spans[i_idx], unique_candidate_arg_spans[j_idx]
                        is_gold_fwd = any(c==s1 and e==s2 for c,e,l in relation_tuples if l!=NEGATIVE_SAMPLE_REL_ID)
                        is_gold_bwd = any(c==s2 and e==s1 for c,e,l in relation_tuples if l!=NEGATIVE_SAMPLE_REL_ID)
                        if not is_gold_fwd and not is_gold_bwd:
                            if not any(c==s1 and e==s2 and l==NEGATIVE_SAMPLE_REL_ID for c,e,l in relation_tuples):
                                potential_neg_pairs_s1.append((s1, s2, NEGATIVE_SAMPLE_REL_ID))
                random.shuffle(potential_neg_pairs_s1)
                for neg_pair in potential_neg_pairs_s1:
                    if generated_neg_count >= num_negative_to_generate: break
                    relation_tuples.append(neg_pair); generated_neg_count += 1

            if generated_neg_count < num_negative_to_generate:
                current_tokens_str_list = self.tokenizer.convert_ids_to_tokens(input_ids_list)
                
                attempts_s2, max_attempts_s2 = 0, (num_negative_to_generate - generated_neg_count) * 50 

                if unique_candidate_arg_spans:
                    for _ in range(max_attempts_s2 // 2):
                        if generated_neg_count >= num_negative_to_generate or current_sequence_length <=1 : break
                        existing_arg = random.choice(unique_candidate_arg_spans)
                        if current_sequence_length <= 1: continue
                        r_s = random.randint(0, current_sequence_length - 1)
                        # Use the configurable max_random_span_len
                        r_l = random.randint(1, min(self.max_random_span_len, current_sequence_length - r_s))
                        r_e = r_s + r_l - 1
                        if r_s > r_e: continue
                        new_rand_span = (r_s, r_e)
                        if not self._is_span_valid_for_random_neg_sampling(new_rand_span, current_tokens_str_list, word_ids_list, unique_candidate_arg_spans): continue
                        if check_span_overlap_util(existing_arg, new_rand_span): continue
                        pairs_to_try = [(existing_arg, new_rand_span, NEGATIVE_SAMPLE_REL_ID), (new_rand_span, existing_arg, NEGATIVE_SAMPLE_REL_ID)]
                        random.shuffle(pairs_to_try)
                        added_in_this_iteration_s2_1 = False
                        for s1_r, s2_r, rel_r in pairs_to_try:
                            if generated_neg_count >= num_negative_to_generate: break
                            is_g = any((c==s1_r and e==s2_r)or(c==s2_r and e==s1_r) for c,e,l in relation_tuples if l!=NEGATIVE_SAMPLE_REL_ID)
                            is_a = any(c==s1_r and e==s2_r and l==NEGATIVE_SAMPLE_REL_ID for c,e,l in relation_tuples)
                            if not is_g and not is_a:
                                relation_tuples.append((s1_r,s2_r,rel_r)); generated_neg_count+=1; added_in_this_iteration_s2_1=True; break
                        if added_in_this_iteration_s2_1 and generated_neg_count >= num_negative_to_generate: break


                while generated_neg_count < num_negative_to_generate and attempts_s2 < max_attempts_s2 and current_sequence_length > 1 :
                    attempts_s2 +=1
                    if current_sequence_length <=1 : break
                    s1_s = random.randint(0, current_sequence_length - 1)
                    # Use the configurable max_random_span_len
                    s1_l = random.randint(1, min(self.max_random_span_len, current_sequence_length - s1_s))
                    s1_e = s1_s + s1_l - 1;
                    if s1_s > s1_e: continue
                    sp1_r = (s1_s, s1_e)
                    if not self._is_span_valid_for_random_neg_sampling(sp1_r, current_tokens_str_list, word_ids_list, unique_candidate_arg_spans): continue
                    
                    s2_s = random.randint(0, current_sequence_length - 1)
                    # Use the configurable max_random_span_len
                    s2_l = random.randint(1, min(self.max_random_span_len, current_sequence_length - s2_s))
                    s2_e = s2_s + s2_l - 1
                    if s2_s > s2_e: continue
                    sp2_r = (s2_s, s2_e)
                    if not self._is_span_valid_for_random_neg_sampling(sp2_r, current_tokens_str_list, word_ids_list, unique_candidate_arg_spans): continue
                    
                    if sp1_r == sp2_r or check_span_overlap_util(sp1_r, sp2_r): continue
                    is_g = any((c==sp1_r and e==sp2_r)or(c==sp2_r and e==sp1_r) for c,e,l in relation_tuples if l!=NEGATIVE_SAMPLE_REL_ID)
                    if is_g: continue
                    if any(c==sp1_r and e==sp2_r and l==NEGATIVE_SAMPLE_REL_ID for c,e,l in relation_tuples): continue
                    relation_tuples.append((sp1_r, sp2_r, NEGATIVE_SAMPLE_REL_ID)); generated_neg_count +=1
        return {
            "input_ids": input_ids_list, "attention_mask": attention_mask_list,
            "bio_labels": bio_labels_list, "cls_label": is_causal_sentence,
            "relation_tuples": relation_tuples, "text": text,
            "tokenized_tokens": self.tokenizer.convert_ids_to_tokens(input_ids_list),
            "original_entities_data": raw_entities_data,
            "final_bio_entities": final_bio_entities,
            "original_relations_data": relations_data
        }

class CausalDatasetCollator:
    """
    Collator class for CausalDataset. Handles dynamic padding and tensor conversion.
    """
    def __init__(self, tokenizer: AutoTokenizer, num_rel_labels_model_expects: int):
        self.tokenizer = tokenizer
        self.num_rel_labels_model_expects = num_rel_labels_model_expects
    
    def __call__(self, batch: list) -> dict:
        features_for_padding, bio_labels_unpadded_batch, cls_labels_list, relation_tuples_batch = [], [], [], []
        for item in batch:
            features_for_padding.append({"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]})
            bio_labels_unpadded_batch.append(item["bio_labels"])
            cls_labels_list.append(item["cls_label"])
            relation_tuples_batch.append(item["relation_tuples"])

        padded_batch_features = self.tokenizer.pad(features_for_padding, padding=True, return_tensors="pt")
        input_ids_padded = padded_batch_features["input_ids"]
        attention_mask_padded = padded_batch_features["attention_mask"]

        max_len_in_batch = input_ids_padded.shape[1]
        bio_labels_padded_list = [bio_list + [-100]*(max_len_in_batch - len(bio_list)) for bio_list in bio_labels_unpadded_batch]
        bio_labels_padded = torch.tensor(bio_labels_padded_list, dtype=torch.long)
        cls_labels_batched = torch.tensor(cls_labels_list, dtype=torch.long)

        pair_idx, c_starts, c_ends, e_starts, e_ends, rel_labs = [], [], [], [], [], []
        for i, rel_tuples_sample in enumerate(relation_tuples_batch):
            if rel_tuples_sample:
                for (c_span, e_span, rel_id) in rel_tuples_sample:
                    if rel_id >= self.num_rel_labels_model_expects:
                        rel_id = NEGATIVE_SAMPLE_REL_ID
                        if rel_id >= self.num_rel_labels_model_expects: continue
                    pair_idx.append(i)
                    c_starts.append(c_span[0]); c_ends.append(c_span[1])
                    e_starts.append(e_span[0]); e_ends.append(e_span[1])
                    rel_labs.append(rel_id)
        
        output = {
            "input_ids": input_ids_padded, "attention_mask": attention_mask_padded,
            "cls_labels": cls_labels_batched, "bio_labels": bio_labels_padded,
            "pair_batch": None, "cause_starts": None, "cause_ends": None,
            "effect_starts": None, "effect_ends": None, "rel_labels": None
        }
        if pair_idx:
            output.update({
                "pair_batch": torch.tensor(pair_idx, dtype=torch.long),
                "cause_starts": torch.tensor(c_starts, dtype=torch.long),
                "cause_ends": torch.tensor(c_ends, dtype=torch.long),
                "effect_starts": torch.tensor(e_starts, dtype=torch.long),
                "effect_ends": torch.tensor(e_ends, dtype=torch.long),
                "rel_labels": torch.tensor(rel_labs, dtype=torch.long)
            })
        return output
# %%
import torch
from torch.utils.data import Dataset, DataLoader # Added DataLoader for testing collate_fn
from transformers import AutoTokenizer
import json
import pandas as pd
import random
import re # For checking punctuation in random spans
from functools import partial # For using collate_fn with DataLoader

# --- Configuration & Label Mappings ---
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
    def __init__(self, dataframe, tokenizer_name, max_length=512, negative_relation_rate=1.0):
        self.dataframe = dataframe.copy()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length # Max length for TRUNCATION
        self.negative_relation_rate = negative_relation_rate

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
        if s_start > s_end : return False
        if any(check_span_overlap_util(span_coords, gold_span) for gold_span in unique_candidate_arg_spans):
            return False
        current_span_tokens = tokens_list[s_start : s_end + 1]
        current_span_word_ids = [word_ids_list_for_span_check[i] for i in range(s_start, s_end + 1) if i < len(word_ids_list_for_span_check)]
        if not any(wid is not None for wid in current_span_word_ids): return False
        span_text = self.tokenizer.convert_tokens_to_string(current_span_tokens).strip()
        if not span_text: return False
        if re.fullmatch(r'[\W_]+', span_text): return False
        return True

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = str(row.get('text', '')).replace(";;", " ")
        raw_entities_data = row.get('entities_parsed', [])
        relations_data = row.get('relations_parsed', [])

        # Tokenize: NO PADDING in __getitem__, only truncation.
        tokenized_output = self.tokenizer(
            text, max_length=self.max_length, padding=False, truncation=True,
            return_offsets_mapping=True, return_attention_mask=True
        )
        input_ids_list = tokenized_output['input_ids']
        attention_mask_list = tokenized_output['attention_mask'] # Unpadded
        offset_mapping = tokenized_output['offset_mapping']
        word_ids_list = tokenized_output.word_ids()

        # --- Entity Preprocessing ---
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

        # --- BIO Tagging (for unpadded sequence) ---
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
                if token_start != -1:
                    current_token_end = token_start
                    for i in range(token_start, len(offset_mapping)):
                        offset_s, offset_e = offset_mapping[i]
                        if offset_s == offset_e and offset_s == 0 and (i >= len(word_ids_list) or word_ids_list[i] is None): continue
                        if offset_s < end_char and offset_e > start_char : current_token_end = i
                        else:
                            if offset_s >= end_char : break 
                    token_end = current_token_end
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
            if i < len(word_ids_list) and word_ids_list[i] is None: bio_labels_list[i] = -100
        
        # --- Relation Processing ---
        relation_tuples = []
        if is_causal_sentence == 1:
            # Positive relations
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
            
            # Negative Sampling
            unique_candidate_arg_spans = []
            temp_arg_spans = [] # Renamed for clarity
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
            # Stage 1
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
            
            # Stage 2
            if generated_neg_count < num_negative_to_generate:
                # Use unpadded tokens and word_ids for validity checks in Stage 2
                current_tokens_str_list = self.tokenizer.convert_ids_to_tokens(input_ids_list)
                # word_ids_list is already for the unpadded sequence here.

                attempts_s2, max_attempts_s2 = 0, (num_negative_to_generate - generated_neg_count) * 20
                # Sub-Stage 2.1
                if unique_candidate_arg_spans:
                    for _ in range(max_attempts_s2 // 2): 
                        if generated_neg_count >= num_negative_to_generate or current_sequence_length <=1 : break
                        existing_arg = random.choice(unique_candidate_arg_spans)
                        if current_sequence_length <= 1: continue
                        r_s = random.randint(0, current_sequence_length - 1)
                        r_l = random.randint(1, min(3, current_sequence_length - r_s))
                        r_e = r_s + r_l - 1
                        if r_s > r_e: continue
                        new_rand_span = (r_s, r_e)
                        if not self._is_span_valid_for_random_neg_sampling(new_rand_span, current_tokens_str_list, word_ids_list, unique_candidate_arg_spans): continue
                        if check_span_overlap_util(existing_arg, new_rand_span): continue
                        
                        pairs_to_try = [(existing_arg, new_rand_span, NEGATIVE_SAMPLE_REL_ID), (new_rand_span, existing_arg, NEGATIVE_SAMPLE_REL_ID)]
                        random.shuffle(pairs_to_try)
                        for s1_r, s2_r, rel_r in pairs_to_try:
                            if generated_neg_count >= num_negative_to_generate: break
                            is_g = any((c==s1_r and e==s2_r)or(c==s2_r and e==s1_r) for c,e,l in relation_tuples if l!=NEGATIVE_SAMPLE_REL_ID)
                            is_a = any(c==s1_r and e==s2_r and l==NEGATIVE_SAMPLE_REL_ID for c,e,l in relation_tuples)
                            if not is_g and not is_a: relation_tuples.append((s1_r,s2_r,rel_r)); generated_neg_count+=1; break
                # Sub-Stage 2.2
                while generated_neg_count < num_negative_to_generate and attempts_s2 < max_attempts_s2 and current_sequence_length > 1 :
                    attempts_s2 +=1
                    if current_sequence_length <=1 : break
                    s1_s = random.randint(0, current_sequence_length - 1)
                    s1_l = random.randint(1, min(3, current_sequence_length - s1_s))
                    s1_e = s1_s + s1_l - 1;
                    if s1_s > s1_e: continue
                    sp1_r = (s1_s, s1_e)
                    if not self._is_span_valid_for_random_neg_sampling(sp1_r, current_tokens_str_list, word_ids_list, unique_candidate_arg_spans): continue
                    s2_s = random.randint(0, current_sequence_length - 1)
                    s2_l = random.randint(1, min(3, current_sequence_length - s2_s))
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
            "original_entities_data": raw_entities_data, "final_bio_entities": final_bio_entities,
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
                for (cs, ce, rel_id) in rel_tuples_sample: # Corrected tuple unpacking
                    # Assuming rel_tuples_sample contains ((c_start,c_end), (e_start,e_end), rel_id)
                    c_span, e_span = cs, ce # If cs is already (start,end)
                    # If cs is not a span but part of a larger tuple, adjust unpacking:
                    # e.g. if cs = ((c_start, c_end), (e_start, e_end)) and ce = rel_id
                    # Then: c_span, e_span = cs[0], cs[1]
                    #     rel_label_id = ce
                    # Based on your CausalDataset, it's: ((c_s,c_e), (e_s,e_e), rel_id)
                    # So the loop should be: for (c_span_tuple, e_span_tuple, rel_label_id) in rel_tuples_sample:
                    # And then: c_s, c_e = c_span_tuple; e_s, e_e = e_span_tuple
                    # The current code: for (c_span, e_span, rel_label_id) in rel_tuples_for_sample:
                    # implies c_span is (start,end), e_span is (start,end)
                    
                    # Assuming c_span and e_span are already (start, end) tuples:
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

# --- generate_full_report function ---
def generate_full_report(csv_path="train.csv", tokenizer_name="google-bert/bert-base-uncased", 
                         max_length_truncate=256, num_samples_to_report=15, negative_rel_rate_for_report=2.0): # Increased rate
    random.seed(8643) 
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded {csv_path}. Total rows: {len(df)}\n")
    except FileNotFoundError:
        print(f"Error: {csv_path} not found."); return
    if len(df) == 0: print("Dataset is empty."); return
    
    # This dataset instance will use padding=False in __getitem__
    full_dataset_instance = CausalDataset(df, tokenizer_name, max_length_truncate, negative_rel_rate_for_report)
    
    actual_num_to_report = min(num_samples_to_report, len(df))
    k_sample = min(actual_num_to_report, len(df))
    if k_sample <=0 : print("Not enough samples to report."); return
        
    random_indices = random.sample(range(len(df)), k_sample)
    indices_to_report = set(random_indices)
    
    ce_sample_found_in_random = any(
        any(e.get('label') == 'internal_CE' for e in full_dataset_instance[r_idx]['final_bio_entities'])
        for r_idx in random_indices
    )
    if not ce_sample_found_in_random:
        print("CE sample not in initial random selection. Searching for one to add...")
        ce_specific_idx = -1
        remaining_indices = list(set(range(len(df))) - indices_to_report)
        random.shuffle(remaining_indices)
        for search_idx in remaining_indices:
            if any(e.get('label') == 'internal_CE' for e in full_dataset_instance[search_idx]['final_bio_entities']):
                ce_specific_idx = search_idx; print(f"Found CE sample at iloc {ce_specific_idx}."); break
        if ce_specific_idx != -1:
            if len(indices_to_report) >= actual_num_to_report and random_indices: indices_to_report.remove(random_indices[0]) 
            indices_to_report.add(ce_specific_idx)
        else: print("Could not find an additional CE-producing sample.")

    final_indices_to_report = sorted(list(indices_to_report))
    
    print(f"--- Generating Report for {len(final_indices_to_report)} Sample(s) (iloc indices: {final_indices_to_report}) ---")
    print(f"Using negative_relation_rate for report generation: {negative_rel_rate_for_report}")
    print(f"NOTE: CausalDataset.__getitem__ returns UNPADDED sequences. Padding occurs in collate_fn.\n")

    for report_count, i in enumerate(final_indices_to_report):
        actual_df_index = df.index[i]
        print(f"\n\n========================= REPORT FOR SAMPLE #{report_count+1} (CSV Original Index: {actual_df_index}, iloc: {i}) =========================")
        original_row = df.iloc[i]
        processed_sample = full_dataset_instance[i]
        print("\n1. ORIGINAL DATA FROM CSV:")
        print(f"  Raw Text: {original_row['text']}")
        print(f"  Raw Entities: {original_row['entities']}")
        print(f"  Raw Relations: {original_row['relations']}")
        print("\n2. PROCESSED OUTPUT FROM CausalDataset.__getitem__ (UNPADDED):")
        print(f"  Cleaned Text: {processed_sample['text']}")
        print(f"  Sentence Classification (0=Non-Causal, 1=Causal): {processed_sample['cls_label']}")
        print("\n  Entities Considered for BIO Tagging (after word-level C/E merge):")
        if processed_sample['final_bio_entities']:
            for bio_entity in processed_sample['final_bio_entities']:
                print(f"    - Label: {bio_entity.get('label', 'N/A')}, Offsets: ({bio_entity.get('start_offset', 'N/A')}, {bio_entity.get('end_offset', 'N/A')}), Display ID: {bio_entity.get('display_id', 'N/A')}, Word Indices: {sorted(list(bio_entity.get('word_indices', set())))}")
                if 'original_ids' in bio_entity: print(f"      Original IDs merged: {bio_entity['original_ids']}")
        else: print("    No entities were considered for BIO tagging.")
        print("\n  Tokenization and BIO Labels (UNPADDED sequence from __getitem__):")
        tokens = processed_sample['tokenized_tokens']
        bio_labels_list = processed_sample['bio_labels']
        
        report_tokenized_output_unpadded = full_dataset_instance.tokenizer(processed_sample['text'], max_length=full_dataset_instance.max_length, padding=False, truncation=True)
        report_word_ids_unpadded = report_tokenized_output_unpadded.word_ids()

        print(f"  Sequence Length (unpadded): {len(tokens)}")
        print(f"  {'Token':<20} | {'BIO Label':<10} (ID) | Word ID")
        print(f"  {'-'*20}-+-{'-'*10}------+-{'-'*7}")
        for token_idx, (token_str, bio_id_val) in enumerate(zip(tokens, bio_labels_list)):
            bio_tag_str = id2label_span.get(bio_id_val, f"RAW({bio_id_val})")
            word_id_for_token = report_word_ids_unpadded[token_idx] if token_idx < len(report_word_ids_unpadded) else "N/A"
            if bio_id_val == -100: print(f"  {token_str:<20} | IGNORED    ({bio_id_val:<3}) | {str(word_id_for_token):<7}")
            else: print(f"  {token_str:<20} | {bio_tag_str:<10} ({bio_id_val:<3}) | {str(word_id_for_token):<7}")

        print("\n  Relation Tuples (Token indices are for the UNPADDED sequence):")
        if processed_sample['relation_tuples']:
            for c_span, e_span, label_id_val in processed_sample['relation_tuples']:
                c_tokens_list = tokens[c_span[0] : min(c_span[1]+1, len(tokens))]
                e_tokens_list = tokens[e_span[0] : min(e_span[1]+1, len(tokens))]
                rel_label_str = id2label_rel.get(label_id_val, f"RAW_ID({label_id_val})")
                print(f"    - Cause: {c_tokens_list} (Indices: {c_span}) | Effect: {e_tokens_list} (Indices: {e_span}) | Relation: {rel_label_str} (ID: {label_id_val})")
                if label_id_val == NEGATIVE_SAMPLE_REL_ID: print("      (This is a negatively sampled relation: Rel_None)")
        else: print("    No relations extracted or generated for this sample (Zero Connections).")
        print("\n  Original Parsed Data (as loaded by CausalDataset):")
        print(f"    Parsed Entities: {processed_sample['original_entities_data']}")
        print(f"    Parsed Relations: {processed_sample['original_relations_data']}")
        
    print("\n========================= END OF INDIVIDUAL SAMPLE REPORTS =========================")

    print("\n\n--- TESTING CausalDatasetCollator (Dynamic Padding) ---")
    if len(final_indices_to_report) >= 2:
        collate_test_batch_indices = final_indices_to_report[:min(4, len(final_indices_to_report))]
        collate_test_batch = [full_dataset_instance[i] for i in collate_test_batch_indices]
        print(f"Testing collator with {len(collate_test_batch)} samples (iloc indices: {collate_test_batch_indices}).")
        
        collator_instance = CausalDatasetCollator(
            tokenizer=full_dataset_instance.tokenizer,
            num_rel_labels_model_expects=len(id2label_rel) # Use actual number of defined relation labels
        )
        try:
            batched_data = collator_instance(collate_test_batch)
            print("Collator executed successfully.")
            print("Keys in collated batch:", batched_data.keys())
            for key, value in batched_data.items():
                if isinstance(value, torch.Tensor): print(f"  {key}: Tensor shape {value.shape}, dtype {value.dtype}")
                elif value is None: print(f"  {key}: None")
                else: print(f"  {key}: Type {type(value)}")
            if batched_data.get("pair_batch") is not None: print(f"  Number of relation pairs in collated batch: {len(batched_data['pair_batch'])}")
            if "input_ids" in batched_data:
                print(f"  Batch dynamically padded to sequence length: {batched_data['input_ids'].shape[1]}")
                assert batched_data["input_ids"].shape == batched_data["attention_mask"].shape
                assert batched_data["input_ids"].shape == batched_data["bio_labels"].shape
                print("  Verified: input_ids, attention_mask, and bio_labels are padded to the same batch max length.")
        except Exception as e: print(f"Error during collator test: {e}"); import traceback; traceback.print_exc()
    else: print("Not enough samples for collator test.")
    print("\n========================= END OF COLLATE FUNCTION TEST =========================")

if __name__ == '__main__':
    NUMBER_OF_SAMPLES_FOR_DETAILED_REPORT = 15
    NEGATIVE_SAMPLING_RATE_IN_REPORT = 2.0 # Increased to see more negatives
    MAX_LEN_FOR_DATASET_TRUNCATION = 256 

    # Ensure this path is correct for your environment
    CSV_FILE_PATH = "datasets/train.csv" # Assuming train.csv is in the same directory as the script
    # CSV_FILE_PATH = "datasets/train.csv" # Or if it's in a subdirectory

    generate_full_report(
        csv_path=CSV_FILE_PATH, 
        tokenizer_name="google-bert/bert-base-uncased",
        max_length_truncate=MAX_LEN_FOR_DATASET_TRUNCATION,
        num_samples_to_report=NUMBER_OF_SAMPLES_FOR_DETAILED_REPORT,
        negative_rel_rate_for_report=NEGATIVE_SAMPLING_RATE_IN_REPORT
    )
# %%

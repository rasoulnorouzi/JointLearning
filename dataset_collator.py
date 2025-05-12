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
# Defines the mapping from integer IDs to BIO (Begin, Inside, Outside) span labels.
id2label_span = {0: "B-C", 1: "I-C", 2: "B-E", 3: "I-E", 4: "B-CE", 5: "I-CE", 6: "O"}
label2id_span = {v: k for k, v in id2label_span.items()} # Reverse mapping

# Maps entity labels to BIO prefixes. "internal_CE" is for consolidated C/E spans.
entity_label_to_bio_prefix = {"cause": "C", "effect": "E", "internal_CE": "CE", "non-causal": "O"}

# Defines relation labels.
NO_RELATION_LABEL_STR = "Rel_None" # For negatively sampled pairs
POSITIVE_RELATION_TYPE_TO_ID = {"Rel_CE": 1, "Rel_Zero": 2} # Maps type strings from data to IDs
id2label_rel = {0: NO_RELATION_LABEL_STR, 1: "Rel_CE", 2: "Rel_Zero"}
label2id_rel = {v: k for k, v in id2label_rel.items()} # Reverse mapping
NEGATIVE_SAMPLE_REL_ID = label2id_rel[NO_RELATION_LABEL_STR] # Integer ID for "Rel_None"

# Helper function to check span overlap (can be defined globally or as a static method)
def check_span_overlap_util(span1, span2):
    """Checks if two token spans (start_idx, end_idx) overlap."""
    return max(span1[0], span2[0]) <= min(span1[1], span2[1])

class CausalDataset(Dataset):
    """
    PyTorch Dataset class for processing causal text data.
    MODIFIED to NOT pad in __getitem__. Padding is handled by the collate function.
    Handles sentence classification, BIO span tagging (with word-level CE consolidation),
    and relation extraction (positive and multi-stage negative sampling).
    """
    def __init__(self, dataframe, tokenizer_name, max_length=512, negative_relation_rate=1.0):
        """
        Initializes the dataset.
        Args:
            dataframe (pd.DataFrame): DataFrame containing the raw data.
            tokenizer_name (str): Name of the Hugging Face tokenizer.
            max_length (int): Maximum sequence length for TRUNCATION by the tokenizer.
            negative_relation_rate (float): Rate for generating negative relation samples.
        """
        self.dataframe = dataframe.copy()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length # Max length for truncation by tokenizer
        self.negative_relation_rate = negative_relation_rate

        # Safely parses JSON-like strings in 'entities' and 'relations' columns.
        def safe_json_loads(data_str):
            try:
                if isinstance(data_str, str): return json.loads(data_str.replace("'", "\""))
                return [] 
            except: return [] # Catch all exceptions during parsing
        self.dataframe.loc[:, 'entities_parsed'] = self.dataframe['entities'].apply(safe_json_loads)
        self.dataframe.loc[:, 'relations_parsed'] = self.dataframe['relations'].apply(safe_json_loads)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.dataframe)

    def _get_word_indices_for_entity(self, entity_char_start, entity_char_end, token_offsets, word_ids_list):
        """Maps an entity's character span to a frozenset of word indices it covers."""
        covered_word_indices = set()
        if entity_char_start is None or entity_char_end is None: return frozenset()
        for token_idx, (tok_char_start, tok_char_end) in enumerate(token_offsets):
            # Skip special tokens like [CLS]/[SEP] if they have (0,0) offset by chance,
            # but allow actual content at the start of the sequence.
            if tok_char_start == tok_char_end and tok_char_start == 0 and word_ids_list[token_idx] is None: continue
            if tok_char_start < entity_char_end and entity_char_start < tok_char_end: # Overlap condition
                if token_idx < len(word_ids_list): # Ensure index is valid for word_ids_list
                    word_id = word_ids_list[token_idx]
                    if word_id is not None: # Valid word_id (not None for special tokens)
                        covered_word_indices.add(word_id)
        return frozenset(covered_word_indices)

    def _is_span_valid_for_random_neg_sampling(self, span_coords, tokens_list, word_ids_list_for_span_check, unique_candidate_arg_spans):
        """
        Checks if a randomly generated token span is "valid" for Stage 2 negative sampling.
        A valid span should be non-empty, contain actual word material (not just punctuation/special tokens),
        and not overlap with existing gold argument spans.
        """
        s_start, s_end = span_coords
        if not (0 <= s_start <= s_end < len(tokens_list)): return False # Basic boundary check
        if s_start > s_end : return False # Invalid span definition

        if any(check_span_overlap_util(span_coords, gold_span) for gold_span in unique_candidate_arg_spans):
            return False
        
        current_span_tokens = tokens_list[s_start : s_end + 1]
        # Ensure word_ids_list_for_span_check is correctly sliced or accessed
        current_span_word_ids = [word_ids_list_for_span_check[i] for i in range(s_start, s_end + 1) if i < len(word_ids_list_for_span_check)]

        if not any(wid is not None for wid in current_span_word_ids): return False # Must map to some actual word
        span_text = self.tokenizer.convert_tokens_to_string(current_span_tokens).strip()
        if not span_text: return False # Empty string after joining tokens
        if re.fullmatch(r'[\W_]+', span_text): return False # Consists only of punctuation/whitespace/underscores
        return True

    def __getitem__(self, idx):
        """
        Processes a single sample from the dataset.
        Returns unpadded sequences; padding is handled by the collate function.
        """
        row = self.dataframe.iloc[idx]
        text = str(row.get('text', '')).replace(";;", " ")
        raw_entities_data = row.get('entities_parsed', [])
        relations_data = row.get('relations_parsed', [])

        # Tokenize the input text: NO PADDING HERE, only truncation.
        # return_tensors="pt" is removed here; lists are returned.
        tokenized_output = self.tokenizer(
            text,
            max_length=self.max_length, # Truncates if longer
            padding=False,              # IMPORTANT: No padding in __getitem__
            truncation=True,
            return_offsets_mapping=True,
            return_attention_mask=True  # Still useful for knowing valid tokens before padding
        )

        input_ids_list = tokenized_output['input_ids']         # Python list of token IDs
        attention_mask_list = tokenized_output['attention_mask'] # Python list of 0s and 1s (unpadded)
        offset_mapping = tokenized_output['offset_mapping']    # List of (char_start, char_end) tuples
        word_ids_list = tokenized_output.word_ids()            # List of word indices or None

        # --- Entity Preprocessing (Word-Level CE Span Consolidation) ---
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
            if cause_entity and effect_entity: # Same words are C and E
                ce_char_start = min(cause_entity['start_offset'], effect_entity['start_offset'])
                ce_char_end = max(cause_entity['end_offset'], effect_entity['end_offset'])
                temp_bio_entities.append({
                    'label': 'internal_CE', 'start_offset': ce_char_start, 'end_offset': ce_char_end,
                    'original_ids': [cause_entity['id'], effect_entity['id']],
                    'display_id': f"CE_{cause_entity['id']}_{effect_entity['id']}", 'word_indices': word_indices_key
                })
                processed_original_ids_for_bio.add(cause_entity['id']); processed_original_ids_for_bio.add(effect_entity['id'])
            else: # Add non-consolidated entities
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

        # --- BIO Tagging for the unpadded sequence ---
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
                    if offset_s == offset_e and offset_s == 0 and word_ids_list[i] is None: continue
                    if offset_s < end_char and start_char < offset_e:
                        if token_start == -1: token_start = i
                        token_end = i
                if token_start != -1:
                    current_token_end = token_start
                    for i in range(token_start, len(offset_mapping)):
                        offset_s, offset_e = offset_mapping[i]
                        if offset_s == offset_e and offset_s == 0 and word_ids_list[i] is None: continue
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
        
        # Assign -100 to special tokens (identified by word_ids_list[i] is None)
        for i in range(current_sequence_length):
            if i < len(word_ids_list) and word_ids_list[i] is None:
                bio_labels_list[i] = -100
        
        # --- Relation Processing (Positive and Negative Sampling) ---
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
                                if fbe.get('label') == 'internal_CE' and from_id in fbe.get('original_ids',[]) and to_id in fbe.get('original_ids',[]):
                                    is_self_loop_from_ce = True; break
                        if not is_self_loop_from_ce: relation_tuples.append((c_span, e_span, rel_label))
            
            # Negative Sampling (Refined Two-Stage Logic)
            unique_candidate_arg_spans = []
            temp_arg_spans_for_neg_sampling = []
            for tagged_entity in final_bio_entities:
                if tagged_entity.get('label') in ['cause', 'effect', 'internal_CE']:
                    span_id_to_lookup = None
                    if 'original_ids' in tagged_entity: span_id_to_lookup = tagged_entity['original_ids'][0]
                    elif 'id' in tagged_entity: span_id_to_lookup = tagged_entity['id']
                    if span_id_to_lookup and span_id_to_lookup in entity_spans_for_relations:
                        temp_arg_spans_for_neg_sampling.append(entity_spans_for_relations[span_id_to_lookup])
            for s in temp_arg_spans_for_neg_sampling:
                if s not in unique_candidate_arg_spans: unique_candidate_arg_spans.append(s)

            num_positive_relations = sum(1 for _,_,lab in relation_tuples if lab!=NEGATIVE_SAMPLE_REL_ID)
            num_negative_to_generate = int(num_positive_relations * self.negative_relation_rate)
            if num_positive_relations == 0 and unique_candidate_arg_spans and self.negative_relation_rate > 0:
                 num_negative_to_generate = max(1, int(len(unique_candidate_arg_spans) * self.negative_relation_rate * 0.5))
            
            generated_neg_count = 0
            # Stage 1 (Negative Sampling)
            if len(unique_candidate_arg_spans) >= 2 and generated_neg_count < num_negative_to_generate:
                potential_neg_pairs_s1 = []
                for i_idx in range(len(unique_candidate_arg_spans)):
                    for j_idx in range(len(unique_candidate_arg_spans)):
                        if i_idx == j_idx: continue 
                        s1_coords, s2_coords = unique_candidate_arg_spans[i_idx], unique_candidate_arg_spans[j_idx]
                        is_gold_fwd = any(c==s1_coords and e==s2_coords for c,e,lab in relation_tuples if lab!=NEGATIVE_SAMPLE_REL_ID)
                        is_gold_bwd = any(c==s2_coords and e==s1_coords for c,e,lab in relation_tuples if lab!=NEGATIVE_SAMPLE_REL_ID)
                        if not is_gold_fwd and not is_gold_bwd:
                            if not any(c==s1_coords and e==s2_coords and lab==NEGATIVE_SAMPLE_REL_ID for c,e,lab in relation_tuples):
                                potential_neg_pairs_s1.append((s1_coords, s2_coords, NEGATIVE_SAMPLE_REL_ID))
                random.shuffle(potential_neg_pairs_s1)
                for neg_pair in potential_neg_pairs_s1:
                    if generated_neg_count >= num_negative_to_generate: break
                    relation_tuples.append(neg_pair); generated_neg_count += 1
            
            # Stage 2 (Negative Sampling)
            if generated_neg_count < num_negative_to_generate:
                current_tokens_str_list = self.tokenizer.convert_ids_to_tokens(input_ids_list) # For _is_span_valid
                attempts_s2, max_attempts_s2 = 0, (num_negative_to_generate - generated_neg_count) * 20

                # Sub-Stage 2.1: Pair existing argument with a new valid random span
                if unique_candidate_arg_spans:
                    for _ in range(max_attempts_s2 // 2): 
                        if generated_neg_count >= num_negative_to_generate or current_sequence_length <=3 : break
                        existing_arg_span = random.choice(unique_candidate_arg_spans)
                        
                        r_start = random.randint(0, current_sequence_length - 1) # Allow CLS/SEP if valid word
                        r_len = random.randint(1, min(3, current_sequence_length - r_start))
                        r_end = r_start + r_len - 1
                        if r_end >= current_sequence_length: r_end = current_sequence_length -1 # Ensure valid end
                        if r_start > r_end: continue # Skip if length became non-positive

                        new_random_span = (r_start, r_end)

                        if not self._is_span_valid_for_random_neg_sampling(new_random_span, current_tokens_str_list, word_ids_list, unique_candidate_arg_spans):
                            continue
                        if check_span_overlap_util(existing_arg_span, new_random_span): continue

                        pairs_to_try = [(existing_arg_span, new_random_span, NEGATIVE_SAMPLE_REL_ID),
                                        (new_random_span, existing_arg_span, NEGATIVE_SAMPLE_REL_ID)]
                        random.shuffle(pairs_to_try)
                        for s1_rand, s2_rand, rel_lab_rand in pairs_to_try:
                            if generated_neg_count >= num_negative_to_generate: break
                            is_gold = any((c==s1_rand and e==s2_rand)or(c==s2_rand and e==s1_rand) for c,e,l in relation_tuples if l!=NEGATIVE_SAMPLE_REL_ID)
                            is_added = any(c==s1_rand and e==s2_rand and l==NEGATIVE_SAMPLE_REL_ID for c,e,l in relation_tuples)
                            if not is_gold and not is_added:
                                relation_tuples.append((s1_rand, s2_rand, rel_lab_rand)); generated_neg_count += 1; break
                
                # Sub-Stage 2.2: Fallback to pairing two new valid random spans
                while generated_neg_count < num_negative_to_generate and attempts_s2 < max_attempts_s2 and current_sequence_length > 3 :
                    attempts_s2 +=1
                    if current_sequence_length -1 <=1 : break # Need at least 2 distinct tokens
                    
                    s1_start = random.randint(0, current_sequence_length - 1)
                    s1_len = random.randint(1, min(3, current_sequence_length - s1_start))
                    s1_end = s1_start + s1_len - 1
                    if s1_end >= current_sequence_length: s1_end = current_sequence_length -1
                    if s1_start > s1_end: continue
                    span1_rand = (s1_start, s1_end)
                    if not self._is_span_valid_for_random_neg_sampling(span1_rand, current_tokens_str_list, word_ids_list, unique_candidate_arg_spans): continue

                    s2_start = random.randint(0, current_sequence_length - 1)
                    s2_len = random.randint(1, min(3, current_sequence_length - s2_start))
                    s2_end = s2_start + s2_len - 1
                    if s2_end >= current_sequence_length: s2_end = current_sequence_length -1
                    if s2_start > s2_end: continue
                    span2_rand = (s2_start, s2_end)
                    if not self._is_span_valid_for_random_neg_sampling(span2_rand, current_tokens_str_list, word_ids_list, unique_candidate_arg_spans): continue
                    
                    if span1_rand == span2_rand or check_span_overlap_util(span1_rand, span2_rand): continue
                    
                    is_gold = any((c==span1_rand and e==span2_rand)or(c==span2_rand and e==span1_rand) for c,e,l in relation_tuples if l!=NEGATIVE_SAMPLE_REL_ID)
                    if is_gold: continue
                    if any(c==span1_rand and e==span2_rand and l==NEGATIVE_SAMPLE_REL_ID for c,e,l in relation_tuples): continue
                    relation_tuples.append((span1_rand, span2_rand, NEGATIVE_SAMPLE_REL_ID)); generated_neg_count +=1

        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "bio_labels": bio_labels_list,
            "cls_label": is_causal_sentence, # Return as int
            "relation_tuples": relation_tuples,
            # For report/debugging:
            "text": text,
            "tokenized_tokens": self.tokenizer.convert_ids_to_tokens(input_ids_list),
            "original_entities_data": raw_entities_data,
            "final_bio_entities": final_bio_entities,
            "original_relations_data": relations_data
        }

# --- Collate Function with Dynamic Padding ---
def collate_fn_with_dynamic_padding(batch, tokenizer, num_rel_labels_model_expects):
    """
    Collates a list of samples from CausalDataset (which returns unpadded sequences)
    into a batch with dynamic padding. Handles padding for input_ids, attention_mask,
    and bio_labels. Aggregates relation data into tensors.
    """
    features_for_padding, bio_labels_unpadded_batch, cls_labels_list, relation_tuples_batch = [], [], [], []
    for item in batch:
        features_for_padding.append({"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]})
        bio_labels_unpadded_batch.append(item["bio_labels"])
        cls_labels_list.append(item["cls_label"])
        relation_tuples_batch.append(item["relation_tuples"])

    padded_batch_features = tokenizer.pad(features_for_padding, padding=True, return_tensors="pt")
    input_ids_padded = padded_batch_features["input_ids"]
    attention_mask_padded = padded_batch_features["attention_mask"]

    max_len_in_batch = input_ids_padded.shape[1]
    bio_labels_padded_list = [bio_list + [-100]*(max_len_in_batch - len(bio_list)) for bio_list in bio_labels_unpadded_batch]
    bio_labels_padded = torch.tensor(bio_labels_padded_list, dtype=torch.long)
    cls_labels_batched = torch.tensor(cls_labels_list, dtype=torch.long)

    pair_batch_indices, cause_starts_list, cause_ends_list, effect_starts_list, effect_ends_list, rel_labels_list = [], [], [], [], [], []
    for i, rel_tuples_for_sample in enumerate(relation_tuples_batch):
        if rel_tuples_for_sample:
            for (c_span, e_span, rel_label_id) in rel_tuples_for_sample:
                if rel_label_id >= num_rel_labels_model_expects:
                    rel_label_id = NEGATIVE_SAMPLE_REL_ID
                    if rel_label_id >= num_rel_labels_model_expects: continue
                pair_batch_indices.append(i)
                cause_starts_list.append(c_span[0]); cause_ends_list.append(c_span[1])
                effect_starts_list.append(e_span[0]); effect_ends_list.append(e_span[1])
                rel_labels_list.append(rel_label_id)
    
    output_dict = {
        "input_ids": input_ids_padded, "attention_mask": attention_mask_padded,
        "cls_labels": cls_labels_batched, "bio_labels": bio_labels_padded,
        "pair_batch": None, "cause_starts": None, "cause_ends": None,
        "effect_starts": None, "effect_ends": None, "rel_labels": None
    }
    if pair_batch_indices:
        output_dict.update({
            "pair_batch": torch.tensor(pair_batch_indices, dtype=torch.long),
            "cause_starts": torch.tensor(cause_starts_list, dtype=torch.long),
            "cause_ends": torch.tensor(cause_ends_list, dtype=torch.long),
            "effect_starts": torch.tensor(effect_starts_list, dtype=torch.long),
            "effect_ends": torch.tensor(effect_ends_list, dtype=torch.long),
            "rel_labels": torch.tensor(rel_labels_list, dtype=torch.long)
        })
    return output_dict

# --- generate_full_report function (modified to use the new CausalDataset and show unpadded info) ---
def generate_full_report(csv_path="train.csv", tokenizer_name="google-bert/bert-base-uncased", 
                         max_length_truncate=256, num_samples_to_report=15, negative_rel_rate_for_report=1.0):
    """
    Generates a detailed report for random samples, using the CausalDataset
    that returns unpadded sequences. Also tests the dynamic padding collate function.
    """
    random.seed(8642) # For reproducibility of random sample selection
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded {csv_path}. Total rows: {len(df)}\n")
    except FileNotFoundError:
        print(f"Error: {csv_path} not found."); return
    if len(df) == 0: print("Dataset is empty."); return
    
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
                ce_specific_idx = search_idx
                print(f"Found a specific CE-producing sample at iloc index {ce_specific_idx}.")
                break
        if ce_specific_idx != -1:
            if len(indices_to_report) >= actual_num_to_report and random_indices:
                indices_to_report.remove(random_indices[0]) 
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
                print(f"    - Label: {bio_entity.get('label', 'N/A')}, "
                      f"Offsets: ({bio_entity.get('start_offset', 'N/A')}, {bio_entity.get('end_offset', 'N/A')}), "
                      f"Display ID: {bio_entity.get('display_id', 'N/A')}, "
                      f"Word Indices: {sorted(list(bio_entity.get('word_indices', set())))}")
                if 'original_ids' in bio_entity: print(f"      Original IDs merged: {bio_entity['original_ids']}")
        else: print("    No entities were considered for BIO tagging.")
        print("\n  Tokenization and BIO Labels (UNPADDED sequence from __getitem__):")
        tokens = processed_sample['tokenized_tokens']
        bio_labels_list = processed_sample['bio_labels']
        
        # For word_ids in report, get them from the unpadded tokenization for this sample
        # This ensures alignment with the unpadded tokens and bio_labels shown.
        report_tokenized_output_unpadded = full_dataset_instance.tokenizer(
            processed_sample['text'], 
            max_length=full_dataset_instance.max_length, 
            padding=False, # Match __getitem__
            truncation=True
        )
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
                print(f"    - Cause: {c_tokens_list} (Token Indices: {c_span})")
                print(f"      Effect: {e_tokens_list} (Token Indices: {e_span})")
                print(f"      Relation: {rel_label_str} (ID: {label_id_val})")
                if label_id_val == NEGATIVE_SAMPLE_REL_ID: print("        (This is a negatively sampled relation: Rel_None)")
                print("      ----")
        else: print("    No relations extracted or generated for this sample (Zero Connections).")
        print("\n  Original Parsed Data (as loaded by CausalDataset):")
        print(f"    Parsed Entities: {processed_sample['original_entities_data']}")
        print(f"    Parsed Relations: {processed_sample['original_relations_data']}")
        
    print("\n========================= END OF INDIVIDUAL SAMPLE REPORTS =========================")

    # --- Test the collate function with a few samples from the dataset ---
    print("\n\n--- TESTING collate_fn_with_dynamic_padding ---")
    if len(final_indices_to_report) >= 2:
        collate_test_batch_indices = final_indices_to_report[:min(4, len(final_indices_to_report))]
        collate_test_batch = [full_dataset_instance[i] for i in collate_test_batch_indices]
        print(f"Testing collate_fn with {len(collate_test_batch)} samples (iloc indices: {collate_test_batch_indices}).")
        
        collate_tokenizer = full_dataset_instance.tokenizer
        model_expected_rel_labels = 3 # Example: Rel_None (0), Rel_CE (1), Rel_Zero (2)

        try:
            batched_data_from_collate = collate_fn_with_dynamic_padding(
                collate_test_batch, 
                tokenizer=collate_tokenizer,
                num_rel_labels_model_expects=model_expected_rel_labels
            )
            print("Collate function executed successfully.")
            print("Keys in collated batch:", batched_data_from_collate.keys())
            for key, value in batched_data_from_collate.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: Tensor shape {value.shape}, dtype {value.dtype}")
                elif value is None: print(f"  {key}: None")
                else: print(f"  {key}: Type {type(value)}, Value: {value}")
            if batched_data_from_collate.get("pair_batch") is not None:
                 print(f"  Number of relation pairs in collated batch: {len(batched_data_from_collate['pair_batch'])}")
            if "input_ids" in batched_data_from_collate:
                batch_seq_len = batched_data_from_collate["input_ids"].shape[1]
                print(f"  Batch dynamically padded to sequence length: {batch_seq_len}")
                assert batched_data_from_collate["input_ids"].shape == batched_data_from_collate["attention_mask"].shape
                assert batched_data_from_collate["input_ids"].shape == batched_data_from_collate["bio_labels"].shape
                print("  Verified: input_ids, attention_mask, and bio_labels are padded to the same batch max length.")
        except Exception as e:
            print(f"Error during collate_fn test: {e}"); import traceback; traceback.print_exc()
    else: print("Not enough samples selected for report to test collate function with a batch size of >= 2.")
    print("\n========================= END OF COLLATE FUNCTION TEST =========================")


if __name__ == '__main__':
    NUMBER_OF_SAMPLES_FOR_DETAILED_REPORT = 15
    NEGATIVE_SAMPLING_RATE_IN_REPORT = 1.0    

    # Note: The user provided `csv_path="datasets/train.csv"`. 
    # Ensure this path is correct for your environment.
    # If "datasets" is a subdirectory in your current working directory, it should work.
    # Otherwise, adjust the path or place train.csv in the same directory as the script.
    generate_full_report(
        csv_path="datasets/train.csv", # Adjusted to look in current dir, or use "datasets/train.csv"
        num_samples_to_report=NUMBER_OF_SAMPLES_FOR_DETAILED_REPORT,
        negative_rel_rate_for_report=NEGATIVE_SAMPLING_RATE_IN_REPORT,
        max_length_truncate=256 # Max length for truncation in CausalDataset
    )
# %%

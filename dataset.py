# %%
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import json
import pandas as pd
import random
import re # For checking punctuation in random spans

# --- Configuration & Label Mappings ---
# (Comments from previous version apply here)
id2label_span = {0: "B-C", 1: "I-C", 2: "B-E", 3: "I-E", 4: "B-CE", 5: "I-CE", 6: "O"}
label2id_span = {v: k for k, v in id2label_span.items()}
entity_label_to_bio_prefix = {"cause": "C", "effect": "E", "internal_CE": "CE", "non-causal": "O"}
NO_RELATION_LABEL_STR = "Rel_None"
POSITIVE_RELATION_TYPE_TO_ID = {"Rel_CE": 1, "Rel_Zero": 2}
id2label_rel = {0: NO_RELATION_LABEL_STR, 1: "Rel_CE", 2: "Rel_Zero"}
label2id_rel = {v: k for k, v in id2label_rel.items()}
NEGATIVE_SAMPLE_REL_ID = label2id_rel[NO_RELATION_LABEL_STR]

class CausalDataset(Dataset):
    """
    PyTorch Dataset class for processing causal text data.
    Handles sentence classification, BIO span tagging (with word-level CE consolidation),
    and relation extraction (positive and multi-stage negative sampling).
    """
    def __init__(self, dataframe, tokenizer_name, max_length=128, negative_relation_rate=1.0):
        self.dataframe = dataframe.copy()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
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
            if tok_char_start == tok_char_end: continue
            if tok_char_start < entity_char_end and entity_char_start < tok_char_end:
                if token_idx < len(word_ids_list):
                    word_id = word_ids_list[token_idx]
                    if word_id is not None: covered_word_indices.add(word_id)
        return frozenset(covered_word_indices)

    def _is_span_valid_for_random_neg_sampling(self, span_coords, tokens, word_ids_list, unique_candidate_arg_spans):
        """
        Checks if a randomly generated token span is "valid" for negative sampling.
        A valid span:
        1. Is not empty and has a positive length.
        2. Consists of actual word material (not just special tokens or pure punctuation).
        3. Does not overlap with existing gold argument spans.
        """
        s_start, s_end = span_coords
        if not (0 <= s_start <= s_end < len(tokens)): return False # Boundary check
        if s_start > s_end : return False # Invalid span

        # Check for overlap with existing gold argument spans
        def check_overlap(span1, span2): return max(span1[0], span2[0]) <= min(span1[1], span2[1])
        if any(check_overlap(span_coords, gold_span) for gold_span in unique_candidate_arg_spans):
            return False

        # Check if span contains meaningful content (not just punctuation or special tokens)
        span_tokens = tokens[s_start : s_end + 1]
        span_word_ids = [word_ids_list[i] for i in range(s_start, s_end + 1) if i < len(word_ids_list)]
        
        if not any(wid is not None for wid in span_word_ids): return False # Must map to some word

        span_text = self.tokenizer.convert_tokens_to_string(span_tokens).strip()
        if not span_text: return False # Empty string after joining
        if re.fullmatch(r'[\W_]+', span_text): return False # Consists only of punctuation/whitespace/underscores

        return True


    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        text = str(row.get('text', '')).replace(";;", " ")
        raw_entities_data = row.get('entities_parsed', [])
        relations_data = row.get('relations_parsed', [])

        tokenized_output = self.tokenizer(
            text, max_length=self.max_length, padding="max_length", truncation=True,
            return_offsets_mapping=True, return_attention_mask=True, return_tensors="pt"
        )
        input_ids = tokenized_output.input_ids.squeeze(0)
        attention_mask = tokenized_output.attention_mask.squeeze(0)
        offset_mapping = tokenized_output.offset_mapping.squeeze(0).tolist()
        word_ids_list = tokenized_output.word_ids()

        # --- Stage 1: Entity Preprocessing (Word-Level CE Span Consolidation) ---
        entities_with_word_spans = []
        # Map raw entities to the set of word indices they cover
        for entity in raw_entities_data:
            if not (isinstance(entity, dict) and all(k in entity for k in ['start_offset', 'end_offset', 'id', 'label'])): continue
            word_indices_set = self._get_word_indices_for_entity(entity['start_offset'], entity['end_offset'], offset_mapping, word_ids_list)
            if word_indices_set: entities_with_word_spans.append({**entity, 'word_indices': word_indices_set})

        # Group entities by the exact set of word indices they cover
        word_span_to_entities = {}
        for entity in entities_with_word_spans:
            key = entity['word_indices']
            if key not in word_span_to_entities: word_span_to_entities[key] = []
            word_span_to_entities[key].append(entity)

        temp_bio_entities, processed_original_ids_for_bio = [], set()
        # Consolidate C/E entities that map to the same set of words
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
        
        # --- Stage 2: Sentence Classification & Finalizing Entities for BIO ---
        is_causal_sentence = 0; final_bio_entities = []
        if any(e.get('label') in ['cause', 'effect', 'internal_CE'] for e in temp_bio_entities):
            is_causal_sentence = 1
            final_bio_entities = [e for e in temp_bio_entities if e.get('label') != 'non-causal']
        elif any(e.get('label') == 'non-causal' for e in temp_bio_entities):
             final_bio_entities = [e for e in temp_bio_entities if e.get('label') == 'non-causal'] # Will be tagged 'O'

        # --- Stage 3: BIO Tagging ---
        bio_labels = [label2id_span["O"]] * self.max_length; entity_spans_for_relations = {}
        if final_bio_entities:
            for entity_to_tag in final_bio_entities:
                # ... (BIO tagging logic as in previous version, using entity_to_tag['start_offset'], ['end_offset']
                # and entity_to_tag['label'] -> bio_prefix -> B-/I- tags) ...
                # This part remains largely the same, mapping char offsets to token spans.
                entity_label_str = entity_to_tag.get('label')
                start_char, end_char = entity_to_tag.get('start_offset'), entity_to_tag.get('end_offset')
                if entity_label_str is None or start_char is None or end_char is None: continue

                token_start, token_end = -1, -1
                for i, (offset_s, offset_e) in enumerate(offset_mapping):
                    if offset_s == offset_e == 0: continue
                    if offset_s < end_char and start_char < offset_e:
                        if token_start == -1: token_start = i
                        token_end = i
                if token_start != -1:
                    current_token_end = token_start
                    for i in range(token_start, len(offset_mapping)):
                        offset_s, offset_e = offset_mapping[i]
                        if offset_s == offset_e == 0: continue
                        if offset_s < end_char and offset_e > start_char : current_token_end = i
                        else:
                            if offset_s >= end_char : break 
                    token_end = current_token_end

                if token_start != -1 and token_end != -1 and token_start <= token_end:
                    bio_prefix = entity_label_to_bio_prefix.get(entity_label_str)
                    if bio_prefix:
                        if bio_prefix != "O" and token_start < self.max_length:
                            bio_labels[token_start] = label2id_span[f"B-{bio_prefix}"]
                            for i_bio in range(token_start + 1, min(token_end + 1, self.max_length)):
                                if i_bio < self.max_length: bio_labels[i_bio] = label2id_span[f"I-{bio_prefix}"]
                    
                    current_span_for_relation = (token_start, token_end)
                    if 'original_ids' in entity_to_tag:
                        for orig_id in entity_to_tag['original_ids']: entity_spans_for_relations[orig_id] = current_span_for_relation
                    elif 'id' in entity_to_tag: entity_spans_for_relations[entity_to_tag['id']] = current_span_for_relation

        # Assign -100 to special tokens
        for i, token_id_val in enumerate(input_ids):
            if token_id_val in [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id]:
                if i < len(bio_labels): bio_labels[i] = -100
        
        # --- Stage 4: Relation Processing (Positive and Negative Sampling) ---
        relation_tuples = []
        if is_causal_sentence == 1:
            # Positive relations
            if relations_data:
                for rel in relations_data:
                    # ... (Positive relation extraction as in previous version) ...
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
            unique_candidate_arg_spans = [] # List of actual C/E/CE token spans
            # ... (populate unique_candidate_arg_spans from final_bio_entities and entity_spans_for_relations as before) ...
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

            # Stage 1 (Negative Sampling): Pair existing argument spans, avoiding flipped gold relations.
            if len(unique_candidate_arg_spans) >= 2 and generated_neg_count < num_negative_to_generate:
                potential_neg_pairs_s1 = []
                for i_idx in range(len(unique_candidate_arg_spans)):
                    for j_idx in range(len(unique_candidate_arg_spans)):
                        if i_idx == j_idx: continue # Avoid self-pairing (already handles CE with itself)
                        s1_coords, s2_coords = unique_candidate_arg_spans[i_idx], unique_candidate_arg_spans[j_idx]
                        
                        is_gold_fwd = any(c==s1_coords and e==s2_coords for c,e,lab in relation_tuples if lab!=NEGATIVE_SAMPLE_REL_ID)
                        is_gold_bwd = any(c==s2_coords and e==s1_coords for c,e,lab in relation_tuples if lab!=NEGATIVE_SAMPLE_REL_ID)
                        
                        if not is_gold_fwd and not is_gold_bwd: # Critical: neither direction is gold
                            if not any(c==s1_coords and e==s2_coords and lab==NEGATIVE_SAMPLE_REL_ID for c,e,lab in relation_tuples):
                                potential_neg_pairs_s1.append((s1_coords, s2_coords, NEGATIVE_SAMPLE_REL_ID))
                random.shuffle(potential_neg_pairs_s1)
                for neg_pair in potential_neg_pairs_s1:
                    if generated_neg_count >= num_negative_to_generate: break
                    relation_tuples.append(neg_pair); generated_neg_count += 1
            
            # Stage 2 (Negative Sampling): Pair existing argument with a new valid random span, or two new valid random spans.
            if generated_neg_count < num_negative_to_generate:
                actual_seq_len = torch.sum(attention_mask).item()
                attempts_s2, max_attempts_s2 = 0, (num_negative_to_generate - generated_neg_count) * 20 # Increased attempts

                # Try pairing existing arg with new random span first
                if unique_candidate_arg_spans:
                    for _ in range(max_attempts_s2 // 2): # Allocate some attempts to this sub-strategy
                        if generated_neg_count >= num_negative_to_generate: break
                        if actual_seq_len <=3 : break
                        
                        existing_arg_span = random.choice(unique_candidate_arg_spans)
                        
                        # Generate one new valid random span
                        r_start = random.randint(1, actual_seq_len - 2)
                        r_len = random.randint(1, min(3, actual_seq_len - 1 - r_start))
                        r_end = r_start + r_len - 1
                        new_random_span = (r_start, r_end)

                        if not self._is_span_valid_for_random_neg_sampling(new_random_span, self.tokenizer.convert_ids_to_tokens(input_ids), word_ids_list, unique_candidate_arg_spans):
                            continue
                        if check_span_overlap_util(existing_arg_span, new_random_span): # Defined in thought process
                            continue

                        # Try (existing, new_random) and (new_random, existing)
                        pairs_to_try = [
                            (existing_arg_span, new_random_span, NEGATIVE_SAMPLE_REL_ID),
                            (new_random_span, existing_arg_span, NEGATIVE_SAMPLE_REL_ID)
                        ]
                        random.shuffle(pairs_to_try) # Randomize order of trying these two directions

                        for s1_rand, s2_rand, rel_lab_rand in pairs_to_try:
                            if generated_neg_count >= num_negative_to_generate: break
                            is_gold_rand = any((c==s1_rand and e==s2_rand) or (c==s2_rand and e==s1_rand) for c,e,lab in relation_tuples if lab != NEGATIVE_SAMPLE_REL_ID)
                            is_added_neg_rand = any(c==s1_rand and e==s2_rand and lab==NEGATIVE_SAMPLE_REL_ID for c,e,lab in relation_tuples)
                            if not is_gold_rand and not is_added_neg_rand:
                                relation_tuples.append((s1_rand, s2_rand, rel_lab_rand))
                                generated_neg_count += 1
                                break # Added one, move to next attempt or break if quota met

                # Fallback: Pair two new valid random spans if still not enough
                while generated_neg_count < num_negative_to_generate and attempts_s2 < max_attempts_s2 and actual_seq_len > 3 :
                    attempts_s2 +=1
                    # ... (logic for pairing two new _is_span_valid_for_random_neg_sampling spans as refined before)
                    if actual_seq_len -2 <=1 : break
                    s1_start, s1_end = random.randint(1, actual_seq_len - 2), 0
                    s1_end = s1_start + random.randint(0, min(2, actual_seq_len - 2 - s1_start))
                    span1_rand = (s1_start, s1_end)
                    if not self._is_span_valid_for_random_neg_sampling(span1_rand, self.tokenizer.convert_ids_to_tokens(input_ids), word_ids_list, unique_candidate_arg_spans): continue

                    s2_start, s2_end = random.randint(1, actual_seq_len - 2), 0
                    s2_end = s2_start + random.randint(0, min(2, actual_seq_len - 2 - s2_start))
                    span2_rand = (s2_start, s2_end)
                    if not self._is_span_valid_for_random_neg_sampling(span2_rand, self.tokenizer.convert_ids_to_tokens(input_ids), word_ids_list, unique_candidate_arg_spans): continue
                    
                    if span1_rand == span2_rand or check_span_overlap_util(span1_rand, span2_rand): continue
                    
                    is_gold_rand = any((c==span1_rand and e==span2_rand)or(c==span2_rand and e==span1_rand) for c,e,lab in relation_tuples if lab!=NEGATIVE_SAMPLE_REL_ID)
                    if is_gold_rand: continue
                    if any(c==span1_rand and e==span2_rand and lab==NEGATIVE_SAMPLE_REL_ID for c,e,lab in relation_tuples): continue
                    relation_tuples.append((span1_rand, span2_rand, NEGATIVE_SAMPLE_REL_ID)); generated_neg_count +=1


        return {
            "text": text, "input_ids": input_ids, "attention_mask": attention_mask,
            "cls_label": torch.tensor(is_causal_sentence, dtype=torch.long),
            "bio_labels": torch.tensor(bio_labels, dtype=torch.long),
            "tokenized_tokens": self.tokenizer.convert_ids_to_tokens(input_ids),
            "relation_tuples": relation_tuples,
            "original_entities_data": raw_entities_data, "final_bio_entities": final_bio_entities,
            "original_relations_data": relations_data
        }

# (Helper function defined globally for use in find_and_process_CE_like_sample if used standalone)
def _get_word_indices_for_entity_global(entity_char_start, entity_char_end, token_offsets, word_ids_list):
    covered_word_indices = set()
    if entity_char_start is None or entity_char_end is None: return frozenset()
    for token_idx, (tok_char_start, tok_char_end) in enumerate(token_offsets):
        if tok_char_start == tok_char_end: continue
        if tok_char_start < entity_char_end and entity_char_start < tok_char_end:
            if token_idx < len(word_ids_list): word_id = word_ids_list[token_idx]
            if word_id is not None: covered_word_indices.add(word_id)
    return frozenset(covered_word_indices)

def generate_full_report(csv_path="train.csv", tokenizer_name="google-bert/bert-base-uncased", 
                         max_length=256, num_samples_to_report=15, negative_rel_rate_for_report=1.0): # Report 15 samples
    """
    Generates a detailed report for a specified number of *random* samples from the dataset,
    ensuring at least one sample demonstrating CE tagging is included.
    """
    random.seed(8642) # Set the random seed
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded {csv_path}. Total rows: {len(df)}\n")
    except FileNotFoundError:
        print(f"Error: {csv_path} not found."); return

    if len(df) == 0: print("Dataset is empty."); return
    
    # Instantiate dataset once for processing
    # Note: CausalDataset internally copies the df, so the original df here is safe.
    full_dataset_instance = CausalDataset(df, tokenizer_name, max_length, negative_relation_rate=negative_rel_rate_for_report)


    # --- Sample Selection: Random + Ensure CE ---
    actual_num_to_report = min(num_samples_to_report, len(df))
    random_indices = random.sample(range(len(df)), min(actual_num_to_report, len(df)))
    
    indices_to_report = set(random_indices)
    
    # Check if any of these random samples will produce CE tags.
    # This requires partially processing them or checking their `final_bio_entities`.
    ce_sample_found_in_random = False
    for r_idx in random_indices:
        # Temporarily get the item to check for CE tags in its processed entities
        # This is a bit heavy as it calls __getitem__
        # A lighter check might be to directly inspect raw_entities for the CE pattern
        # but using the dataset's logic is more accurate to what it *would* tag.
        temp_processed_sample = full_dataset_instance[r_idx] # Calls __getitem__
        if any(e.get('label') == 'internal_CE' for e in temp_processed_sample['final_bio_entities']):
            ce_sample_found_in_random = True
            break
            
    if not ce_sample_found_in_random:
        print("CE sample not in initial random selection. Searching for one...")
        # Search for a CE sample (using logic similar to CausalDataset's internal CE detection)
        # This search logic needs to be efficient or targeted.
        # For now, iterate and call __getitem__ until one is found or limit is reached.
        ce_specific_idx = -1
        for search_idx in range(len(df)):
            if search_idx in indices_to_report : continue # Already selected
            
            # Check if this sample would result in CE tags
            # This is the most reliable way: use the dataset's own processing logic
            potential_ce_sample_processed = full_dataset_instance[search_idx]
            if any(e.get('label') == 'internal_CE' for e in potential_ce_sample_processed['final_bio_entities']):
                ce_specific_idx = search_idx
                print(f"Found a specific CE-producing sample at iloc index {ce_specific_idx}.")
                break
        
        if ce_specific_idx != -1:
            if len(indices_to_report) >= actual_num_to_report and random_indices:
                indices_to_report.remove(random_indices[0]) # Remove one random to make space
            indices_to_report.add(ce_specific_idx)
        else:
            print("Could not find a CE-producing sample to add to the report.")

    final_indices_to_report = sorted(list(indices_to_report))
    
    print(f"--- Generating Report for {len(final_indices_to_report)} Sample(s) (iloc indices: {final_indices_to_report}) ---")
    print(f"Using negative_relation_rate for report generation: {negative_rel_rate_for_report}")

    # --- Report Generation Loop ---
    for report_count, i in enumerate(final_indices_to_report):
        actual_df_index = df.index[i] 
        print(f"\n\n========================= REPORT FOR SAMPLE #{report_count+1} (CSV Original Index: {actual_df_index}, iloc: {i}) =========================")
        
        original_row = df.iloc[i]
        print("\n1. ORIGINAL DATA FROM CSV:")
        # ... (printing original data as before) ...
        print(f"  Raw Text: {original_row['text']}")
        print(f"  Raw Entities: {original_row['entities']}")
        print(f"  Raw Relations: {original_row['relations']}")

        processed_sample = full_dataset_instance[i] # Get the already instantiated dataset item
        print("\n2. PROCESSED OUTPUT FROM CausalDataset:")
        # ... (detailed printing of processed_sample parts as in previous version,
        # including tokenized_tokens, bio_labels with word_ids, relation_tuples etc.) ...
        print(f"  Cleaned Text (for model): {processed_sample['text']}")
        print(f"  Sentence Classification (0=Non-Causal, 1=Causal): {processed_sample['cls_label'].item()}")

        print("\n  Entities Considered for BIO Tagging (after word-level C/E merge):")
        if processed_sample['final_bio_entities']:
            for bio_entity in processed_sample['final_bio_entities']:
                print(f"    - Label: {bio_entity.get('label', 'N/A')}, "
                      f"Offsets: ({bio_entity.get('start_offset', 'N/A')}, {bio_entity.get('end_offset', 'N/A')}), "
                      f"Display ID: {bio_entity.get('display_id', 'N/A')}, "
                      f"Word Indices: {sorted(list(bio_entity.get('word_indices', set())))}")
                if 'original_ids' in bio_entity: print(f"      Original IDs merged: {bio_entity['original_ids']}")
        else: print("    No entities were considered for BIO tagging.")

        print("\n  Tokenization and BIO Labels:")
        tokens = processed_sample['tokenized_tokens']
        bio_labels_list = processed_sample['bio_labels'].tolist()
        print(f"  {'Token':<20} | {'BIO Label':<10} (ID) | Word ID (from tokenizer)")
        print(f"  {'-'*20}-+-{'-'*10}------+-{'-'*20}")
        
        # Get word_ids aligned with the final tokenized output for accurate reporting
        report_tokenized_output = full_dataset_instance.tokenizer(processed_sample['text'], 
                                                    max_length=full_dataset_instance.max_length, 
                                                    padding="max_length", 
                                                    truncation=True)
        report_word_ids = report_tokenized_output.word_ids()

        for token_idx, (token_str, bio_id_val) in enumerate(zip(tokens, bio_labels_list)):
            bio_tag_str = id2label_span.get(bio_id_val, f"RAW({bio_id_val})")
            word_id_for_token = report_word_ids[token_idx] if token_idx < len(report_word_ids) else "N/A"
            if bio_id_val == -100: print(f"  {token_str:<20} | IGNORED    ({bio_id_val:<3}) | {str(word_id_for_token):<20}")
            else: print(f"  {token_str:<20} | {bio_tag_str:<10} ({bio_id_val:<3}) | {str(word_id_for_token):<20}")

        print("\n  Relation Tuples (Cause, Effect, Label):")
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
            
        print("\n  Original Parsed Data (loaded by CausalDataset for its internal processing):")
        print(f"    Parsed Entities (raw_entities_data from CSV): {processed_sample['original_entities_data']}")
        print(f"    Parsed Relations (relations_data from CSV): {processed_sample['original_relations_data']}")
        
    print("\n========================= END OF REPORT =========================")

if __name__ == '__main__':
    NUMBER_OF_SAMPLES_FOR_DETAILED_REPORT = 15 # User requested 15 random samples
    NEGATIVE_SAMPLING_RATE_IN_REPORT = 1.0    

    # Helper to check span overlap, ensure it's defined if CausalDataset's internal one isn't accessible.
    def check_span_overlap_util(span1, span2): return max(span1[0], span2[0]) <= min(span1[1], span2[1])

    generate_full_report(
        csv_path="datasets/train.csv",
        num_samples_to_report=NUMBER_OF_SAMPLES_FOR_DETAILED_REPORT,
        negative_rel_rate_for_report=NEGATIVE_SAMPLING_RATE_IN_REPORT
    )
# %%

# %%
import random
from src.jointlearning.dataset_collator import CausalDatasetCollator, CausalDataset
from src.jointlearning.config import id2label_bio, id2label_rel, NEGATIVE_SAMPLE_REL_ID
import pandas as pd
import torch


# --- generate_full_report function ---
def generate_full_report(csv_path="train.csv", tokenizer_name="google-bert/bert-base-uncased", 
                         max_length_truncate=256, num_samples_to_report=15, negative_rel_rate_for_report=3.0): # Increased rate
    random.seed(8642) 
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
            bio_tag_str = id2label_bio.get(bio_id_val, f"RAW({bio_id_val})")
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
    NUMBER_OF_SAMPLES_FOR_DETAILED_REPORT = 40
    NEGATIVE_SAMPLING_RATE_IN_REPORT = 2.0 # Increased to see more negatives
    MAX_LEN_FOR_DATASET_TRUNCATION = 256 

    # Ensure this path is correct for your environment
    CSV_FILE_PATH = "datasets/expert_multi_task_data/train.csv" # Assuming train.csv is in the same directory as the script
    # CSV_FILE_PATH = "datasets/train.csv" # Or if it's in a subdirectory

    generate_full_report(
        csv_path=CSV_FILE_PATH, 
        tokenizer_name="google-bert/bert-base-uncased",
        max_length_truncate=MAX_LEN_FOR_DATASET_TRUNCATION,
        num_samples_to_report=NUMBER_OF_SAMPLES_FOR_DETAILED_REPORT,
        negative_rel_rate_for_report=NEGATIVE_SAMPLING_RATE_IN_REPORT
    )
# %%

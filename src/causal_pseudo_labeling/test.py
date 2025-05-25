import json

def format_and_print_samples(file_path, num_samples=40):
    """
    Reads samples from a Doccano-formatted JSONL file,
    reformats them, and prints a specified number of samples.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                
                try:
                    doccano_sample = json.loads(line.strip())
                except json.JSONDecodeError as e:
                    print(f"Skipping line {i+1} due to JSON decode error: {e}")
                    continue

                text = doccano_sample.get("text", "")
                entities = doccano_sample.get("entities", [])
                relations_info = doccano_sample.get("relations", [])

                print(f"Input: {text}\n")
                
                output_relations = []
                entity_id_to_text = {}

                for entity in entities:
                    entity_id = entity.get("id")
                    start = entity.get("start_offset")
                    end = entity.get("end_offset")
                    if entity_id is not None and start is not None and end is not None:
                        # Ensure offsets are within text bounds
                        if 0 <= start < end <= len(text):
                            entity_id_to_text[entity_id] = text[start:end]
                        else:
                            print(f"Warning: Invalid offsets for entity ID {entity_id} in sample {i+1}. Text length: {len(text)}, start: {start}, end: {end}")
                            entity_id_to_text[entity_id] = "INVALID_OFFSET_ENTITY"
                
                for rel_info in relations_info:
                    from_id = rel_info.get("from_id")
                    to_id = rel_info.get("to_id")
                    rel_type = rel_info.get("type") # Get the type as is from doccano
                    if rel_type is None:
                        rel_type = "Rel_CE" # Default if type is missing, as per original example


                    cause_text = entity_id_to_text.get(from_id, "UNKNOWN_CAUSE")
                    effect_text = entity_id_to_text.get(to_id, "UNKNOWN_EFFECT")
                    
                    output_relations.append({
                        "cause": cause_text,
                        "effect": effect_text,
                        "type": rel_type
                    })

                # Determine 'causal' based on whether relations were found
                is_causal = bool(output_relations) # If there are relations, it's causal, otherwise it's not.

                output_data = {
                    "text": text,
                    "causal": is_causal, 
                    "relations": output_relations
                }
                
                print("Output:")
                print(json.dumps(output_data, indent=2, ensure_ascii=False))
                print("\n---\n")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    file_to_test = r"C:/Users/norouzin/Desktop/JointLearning/src/causal_pseudo_labeling/annotation_datasets/doccano_llama38b.jsonl"
    format_and_print_samples(file_to_test, num_samples=40)
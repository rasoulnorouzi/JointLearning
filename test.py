import pandas as pd
import json
import ast

def csv_to_jsonl(csv_file_path, jsonl_file_path):
    """
    Convert CSV file to JSONL format.
    
    Args:
        csv_file_path (str): Path to the input CSV file
        jsonl_file_path (str): Path to the output JSONL file
    """
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Open the output JSONL file
    with open(jsonl_file_path, 'w', encoding='utf-8') as jsonl_file:
        for _, row in df.iterrows():
            # Create a dictionary for each row
            record = {
                'id': row['id'],
                'text': row['text'],
                'entities': ast.literal_eval(row['entities']) if pd.notna(row['entities']) and row['entities'] != '[]' else [],
                'relations': ast.literal_eval(row['relations']) if pd.notna(row['relations']) and row['relations'] != '[]' else [],
                'Comments': ast.literal_eval(row['Comments']) if pd.notna(row['Comments']) and row['Comments'] != '[]' else []
            }
            
            # Write the record as a JSON line
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    # Define file paths
    csv_file = r"C:\Users\norouzin\Desktop\JointLearning\Notebooks\predictions\bert-softmax__span_only__thr0.75.csv"
    jsonl_file = r"C:\Users\norouzin\Desktop\JointLearning\datasets\expert_multi_task_data\bert-softmax__span_only__thr0.75.jsonl"

    # Convert CSV to JSONL
    csv_to_jsonl(csv_file, jsonl_file)
    print(f"Successfully converted {csv_file} to {jsonl_file}")
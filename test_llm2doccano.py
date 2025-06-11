# Test script for the updated llm2doccano module
import sys
import os
import pandas as pd

# Add the src directory to the Python path
sys.path.append("C:\\Users\\norouzin\\Desktop\\JointLearning\\src")

# Import the convert_llm_output_to_doccano function
from causal_pseudo_labeling.llm2doccano import convert_llm_output_to_doccano

def main():
    # Example 1: Convert a JSONL file to DataFrame
    input_file = "datasets/pseudo_annotate_data/llama3_8b_raw.jsonl"
    
    print("Converting JSONL file to DataFrame...")
    df = convert_llm_output_to_doccano(input_file)
    
    print(f"\nDataFrame shape: {df.shape}")
    print(f"DataFrame columns: {df.columns.tolist()}")
    print("\nFirst few rows of the DataFrame:")
    print(df.head(2))
    
    # Example 2: Convert and save to file
    output_file = "datasets/pseudo_annotate_data/llama3_8b_processed_new.jsonl"
    print(f"\nConverting JSONL file and saving to {output_file}...")
    df = convert_llm_output_to_doccano(input_file, output_file)
    
    # Example 3: Save DataFrame to CSV
    csv_output = "datasets/pseudo_annotate_data/llama3_8b_processed_new.csv"
    print(f"\nSaving DataFrame to CSV: {csv_output}...")
    df.to_csv(csv_output, index=False)
    
    print("\nConversion completed successfully!")

if __name__ == "__main__":
    main()

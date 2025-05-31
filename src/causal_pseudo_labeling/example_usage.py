#!/usr/bin/env python3
"""
Example Usage: LLM to Doccano Converter

This script demonstrates how to use the LLM to Doccano converter.
It includes progress tracking and comprehensive result verification.

Key Features Demonstrated:
1. Converting LLM output to Doccano format
2. Maintaining exact sample count
3. Handling dual-role entities (spans that are both cause and effect)
4. Robust error handling for malformed samples
5. Progress tracking with detailed statistics

Usage:
    python converter_example.py
"""

from llm2doccano import convert_llama3_to_doccano
from typing import Dict
import json

def main():
    """Main conversion example with verification."""
    
    print("🔄 LLM to Doccano Format Converter")
    print("=" * 50)
    # Configuration
    input_file = "C:\\Users\\norouzin\\Desktop\\JointLearning\\datasets\\pseudo_annotated_data\\llama3_8b_raw.jsonl"
    output_file = "C:\\Users\\norouzin\\Desktop\\JointLearning\\src\\causal_pseudo_labeling\\annotation_datasets\\doccano_final_converted.jsonl"

    print(f"📥 Input file: {input_file}")
    print(f"📤 Output file: {output_file}")
    print()
    
    # Perform conversion
    try:
        stats = convert_llama3_to_doccano(input_file, output_file)
        
        # Verify conversion integrity
        print("\n🔍 VERIFICATION")
        print("-" * 30)
        
        # Check sample count preservation
        with open(input_file, 'r', encoding='utf-8') as f:
            original_count = len(f.readlines())
        
        with open(output_file, 'r', encoding='utf-8') as f:
            converted_count = len(f.readlines())
        
        count_match = original_count == converted_count
        print(f"Original samples: {original_count}")
        print(f"Converted samples: {converted_count}")
        print(f"Count preservation: {'✅ PASS' if count_match else '❌ FAIL'}")
        
        if not count_match:
            print("❌ ERROR: Sample count mismatch!")
            return False
        
        # Sample verification
        print(f"\n📊 CONVERSION STATISTICS")
        print("-" * 30)
        print(f"Total samples: {stats['total_samples']:,}")
        print(f"Causal samples: {stats['causal_samples']:,} ({stats['causal_samples']/stats['total_samples']*100:.1f}%)")
        print(f"Non-causal samples: {stats['non_causal_samples']:,} ({stats['non_causal_samples']/stats['total_samples']*100:.1f}%)")
        print(f"Error samples: {stats['error_samples']:,} ({stats['error_samples']/stats['total_samples']*100:.1f}%)")
        print(f"Entities created: {stats['entity_count']:,}")
        print(f"Relations created: {stats['relation_count']:,}")
        
        # Check for dual-role entities
        print(f"\n🔗 DUAL-ROLE ENTITY VERIFICATION")
        print("-" * 30)
        dual_role_count = check_dual_role_entities(output_file)
        print(f"Samples with dual-role entities: {dual_role_count}")
        
        # Show sample examples
        print(f"\n📋 SAMPLE EXAMPLES")
        print("-" * 30)
        show_sample_examples(output_file)
        
        print(f"\n✅ Conversion completed successfully!")
        print(f"✅ Output ready for Doccano: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

def check_dual_role_entities(output_file: str) -> int:
    """Check for samples with dual-role entities."""
    dual_role_count = 0
    
    with open(output_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            try:
                sample = json.loads(line.strip())
                entities = sample.get('entities', [])
                
                # Group entities by position to find overlapping ones
                position_groups = {}
                for entity in entities:
                    pos_key = f"{entity['start_offset']}-{entity['end_offset']}"
                    if pos_key not in position_groups:
                        position_groups[pos_key] = []
                    position_groups[pos_key].append(entity['label'])
                
                # Check if any position has multiple roles
                has_dual_roles = any(len(labels) > 1 for labels in position_groups.values())
                if has_dual_roles:
                    dual_role_count += 1
                    
                    # Show first few examples
                    if dual_role_count <= 3:
                        print(f"  Example {dual_role_count} (sample {line_num}):")
                        for pos, labels in position_groups.items():
                            if len(labels) > 1:
                                start, end = pos.split('-')
                                text_span = sample['text'][int(start):int(end)]
                                print(f"    Span: '{text_span[:50]}{'...' if len(text_span) > 50 else ''}'")
                                print(f"    Roles: {sorted(set(labels))}")
                
            except json.JSONDecodeError:
                continue
    
    return dual_role_count

def show_sample_examples(output_file: str) -> None:
    """Show examples of different sample types."""
    examples_found = {'causal': 0, 'non-causal': 0}
    
    with open(output_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if examples_found['causal'] >= 1 and examples_found['non-causal'] >= 1:
                break
                
            try:
                sample = json.loads(line.strip())
                entities = sample.get('entities', [])
                relations = sample.get('relations', [])
                
                # Determine sample type
                if relations and any(e['label'] in ['cause', 'effect'] for e in entities):
                    sample_type = 'causal'
                else:
                    sample_type = 'non-causal'
                
                if examples_found[sample_type] == 0:
                    examples_found[sample_type] = 1
                    print(f"\n  {sample_type.title()} example (sample {line_num}):")
                    print(f"    Text: '{sample['text'][:100]}{'...' if len(sample['text']) > 100 else ''}'")
                    print(f"    Entities: {len(entities)}")
                    print(f"    Relations: {len(relations)}")
                    
                    if sample_type == 'causal' and entities:
                        print(f"    Entity types: {sorted(set(e['label'] for e in entities))}")
                
            except json.JSONDecodeError:
                continue

def convert_custom_files(input_path: str, output_path: str) -> Dict:
    """
    Convenience function for converting custom files.
    
    Args:
        input_path: Path to your LLaMA 3 output JSONL file
        output_path: Path where you want the Doccano format file saved
    
    Returns:
        Conversion statistics dictionary
    
    Example:
        stats = convert_custom_files("my_llama_output.jsonl", "my_doccano_input.jsonl")
        print(f"Converted {stats['total_samples']} samples")
    """
    return convert_llama3_to_doccano(input_path, output_path)

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 All done! Your file is ready for import into Doccano.")
    else:
        print("\n💥 Conversion failed. Please check the error messages above.")

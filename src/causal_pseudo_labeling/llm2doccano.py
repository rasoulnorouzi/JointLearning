#!/usr/bin/env python3
"""
LLM to Doccano Format Converter

This converter transforms LLM raw output format into Doccano training format
with the following key features:

1. **Exact Sample Count Preservation**: Maintains the exact same number of samples
   as the input file. No samples are skipped.

2. **Dual-Role Entity Handling**: When a span appears as both cause and effect
   in different relations, separate entities are created for each role.

3. **Robust Error Handling**: Malformed samples are converted to non-causal
   entries rather than being skipped.

4. **Hallucination Tolerance**: Spans that don't exist in the text are ignored,
   but if all spans in a sample are hallucinated, it becomes non-causal.

Usage:
    from llm2doccano import convert_llama3_to_doccano
    
    stats = convert_llama3_to_doccano("input.jsonl", "output.jsonl")
    print(f"Converted {stats['total_samples']} samples")
"""

import json
import re
from typing import List, Dict, Tuple, Optional

def convert_llama3_to_doccano(input_file: str, output_file: str) -> Dict[str, int]:
    """
    Convert LLM raw output format to Doccano training format.
    
    Args:
        input_file: Path to the LLM raw JSONL file
        output_file: Path to save the converted Doccano format JSONL file
    
    Returns:
        Dictionary with conversion statistics:
        - total_samples: Total number of samples processed
        - causal_samples: Number of samples with causal relations
        - non_causal_samples: Number of non-causal samples
        - error_samples: Number of samples with parsing errors
        - entity_count: Total entities created
        - relation_count: Total relations created
    """
    stats = {
        'total_samples': 0,
        'causal_samples': 0,
        'non_causal_samples': 0,
        'error_samples': 0,
        'entity_count': 0,
        'relation_count': 0
    }
    
    converted_samples = []
    
    # Read all lines from input file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Converting {len(lines)} samples from LLaMA 3 format to Doccano format...")
    
    # Process each line
    for i, line in enumerate(lines):
        if i % 1000 == 0:
            print(f"Processing sample {i}/{len(lines)}")
            
        stats['total_samples'] += 1
        
        # Create default non-causal sample - EVERY sample gets an entry
        default_sample = create_default_sample(i, stats)
        
        try:
            # Extract data from the line
            text, relations_data, is_causal = extract_data_from_line(line)
            
            if not text:
                # No valid text found - use default non-causal
                stats['non_causal_samples'] += 1
                converted_samples.append(default_sample)
                continue
              # Create sample with extracted text
            sample = {
                'id': i,
                'text': text + ';;',  # Add delimiter as in doccano format
                'entities': [],
                'relations': [],
                'Comments': []
            }
            
            if not is_causal or not relations_data:
                # Non-causal case
                sample = create_non_causal_sample(i, text, stats)
                stats['non_causal_samples'] += 1
            else:
                # Causal case - process relations with dual-role handling
                entities_and_relations = process_causal_relations(text, relations_data, stats)
                
                sample['entities'] = entities_and_relations['entities']
                sample['relations'] = entities_and_relations['relations']
                
                # If no valid entities found, treat as non-causal
                if not sample['entities']:
                    sample = create_non_causal_sample(i, text, stats)
                    stats['non_causal_samples'] += 1
                else:
                    stats['causal_samples'] += 1
            
            converted_samples.append(sample)
            
        except Exception as e:
            # Even with errors, we must maintain exact sample count
            stats['error_samples'] += 1
            stats['non_causal_samples'] += 1
            converted_samples.append(default_sample)
    
    # Write all converted samples to output file
    print(f"Writing {len(converted_samples)} samples to output file...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in converted_samples:
            f.write(json.dumps(sample) + '\n')
    
    # Print conversion statistics
    print(f"\n{'='*60}")
    print("CONVERSION COMPLETED SUCCESSFULLY")
    print(f"{'='*60}")
    print(f"Total samples processed: {stats['total_samples']}")
    print(f"Causal samples: {stats['causal_samples']} ({stats['causal_samples']/stats['total_samples']*100:.1f}%)")
    print(f"Non-causal samples: {stats['non_causal_samples']} ({stats['non_causal_samples']/stats['total_samples']*100:.1f}%)")
    print(f"Error samples: {stats['error_samples']} ({stats['error_samples']/stats['total_samples']*100:.1f}%)")
    print(f"Total entities created: {stats['entity_count']}")
    print(f"Total relations created: {stats['relation_count']}")
    print(f"Output saved to: {output_file}")
    print(f"{'='*60}")
    
    return stats

def create_default_sample(sample_id: int, stats: Dict) -> Dict:
    """Create a default non-causal sample for error cases."""
    sample = {
        "id": sample_id,
        "text": ";;",  # Empty text with delimiter
        "entities": [{
            'id': stats['entity_count'],
            'label': 'non-causal',
            'start_offset': 0,
            'end_offset': 0
        }],
        "relations": [],
        "Comments": []
    }
    stats['entity_count'] += 1
    return sample

def create_non_causal_sample(sample_id: int, text: str, stats: Dict) -> Dict:
    """Create a non-causal sample with the given text."""
    sample = {
        'id': sample_id,
        'text': text + ';;',
        'entities': [{
            'id': stats['entity_count'],
            'label': 'non-causal',
            'start_offset': 0,
            'end_offset': len(text)
        }],
        'relations': [],
        'Comments': []
    }
    stats['entity_count'] += 1
    return sample

def extract_data_from_line(line: str) -> Tuple[str, List[Dict], bool]:
    """
    Extract text, relations, and causal flag from a LLaMA output line.
    Uses multiple strategies for robust extraction.
    """
    try:
        line = line.strip()
        
        # Remove outer quotes if present (common in LLaMA output)
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]
            line = line.replace('\\"', '"').replace('\\n', '\n')
        
        # Find JSON block in the line
        json_match = re.search(r'\{.*\}', line, re.DOTALL)
        if not json_match:
            # Try to extract text directly from malformed output
            text_match = re.search(r'"text":\s*"([^"]*)"', line)
            if text_match:
                return text_match.group(1), [], False
            return "", [], False
        
        # Parse JSON with error recovery
        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            json_str = json_match.group()
            json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
            json_str = re.sub(r',\s*]', ']', json_str)
            
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                # Last resort: extract text manually
                text_match = re.search(r'"text":\s*"([^"]*)"', json_str)
                if text_match:
                    return text_match.group(1), [], False
                return "", [], False
        
        # Extract fields with defaults
        text = data.get('text', '').strip()
        is_causal = data.get('causal', False)
        relations_data = data.get('relations', [])
        
        # Ensure relations is a list
        if not isinstance(relations_data, list):
            relations_data = []
        
        return text, relations_data, is_causal
        
    except Exception:
        return "", [], False

def process_causal_relations(text: str, relations_data: List[Dict], stats: Dict) -> Dict:
    """
    Process causal relations with proper dual-role entity handling.
    
    Key feature: When a span appears as both cause and effect in different relations,
    separate entities are created for each role.
    """
    entities_map = {}  # span_text -> list of entities for this span
    entities_list = []
    relations_list = []
    
    # Step 1: Collect all spans and identify their roles
    span_roles = {}  # span_text -> set of roles ('cause', 'effect')
    
    for relation in relations_data:
        cause_span = relation.get('cause', '').strip()
        effect_span = relation.get('effect', '').strip()
        
        # Track cause spans
        if cause_span:
            if cause_span not in span_roles:
                span_roles[cause_span] = set()
            span_roles[cause_span].add('cause')
        
        # Track effect spans
        if effect_span:
            if effect_span not in span_roles:
                span_roles[effect_span] = set()
            span_roles[effect_span].add('effect')
    
    # Step 2: Create entities for each span-role combination
    for span_text, roles in span_roles.items():
        # Find the span in the text
        start_pos = find_span_in_text(text, span_text)
        if start_pos != -1:  # Only if span actually exists in text
            end_pos = start_pos + len(span_text)
            
            # Create separate entity for each role this span plays
            span_entities = []
            for role in sorted(roles):  # Sort for consistent ordering
                entity = {
                    'id': stats['entity_count'],
                    'label': role,
                    'start_offset': start_pos,
                    'end_offset': end_pos,
                    'span_text': span_text  # Temporary field for relation creation
                }
                span_entities.append(entity)
                entities_list.append(entity)
                stats['entity_count'] += 1
            
            entities_map[span_text] = span_entities
    
    # Step 3: Create relations between appropriate entities
    for relation in relations_data:
        cause_span = relation.get('cause', '').strip()
        effect_span = relation.get('effect', '').strip()
        
        if cause_span in entities_map and effect_span in entities_map:
            # Find the cause entity (labeled 'cause') for the cause span
            cause_entity = None
            for entity in entities_map[cause_span]:
                if entity['label'] == 'cause':
                    cause_entity = entity
                    break
            
            # Find the effect entity (labeled 'effect') for the effect span
            effect_entity = None
            for entity in entities_map[effect_span]:
                if entity['label'] == 'effect':
                    effect_entity = entity
                    break
            
            # Create relation if both entities exist
            if cause_entity and effect_entity:
                relations_list.append({
                    'id': stats['relation_count'],
                    'from_id': cause_entity['id'],
                    'to_id': effect_entity['id'],
                    'type': 'Rel_CE'
                })
                stats['relation_count'] += 1
    
    # Step 4: Clean up entities (remove temporary span_text field)
    final_entities = []
    for entity in entities_list:
        final_entities.append({
            'id': entity['id'],
            'label': entity['label'],
            'start_offset': entity['start_offset'],
            'end_offset': entity['end_offset']
        })
    
    return {
        'entities': final_entities,
        'relations': relations_list
    }

def find_span_in_text(text: str, span: str) -> int:
    """
    Find the starting position of a span in text with flexible matching.
    Handles LLM hallucinations by trying multiple matching strategies.
    """
    if not span or not text:
        return -1
    
    text_lower = text.lower()
    span_lower = span.lower()
    
    # Strategy 1: Direct exact match
    pos = text_lower.find(span_lower)
    if pos != -1:
        return pos
    
    # Strategy 2: Normalized whitespace matching
    span_normalized = ' '.join(span_lower.split())
    text_normalized = ' '.join(text_lower.split())
    pos = text_normalized.find(span_normalized)
    if pos != -1:
        # Convert position back to original text
        words_before = text_normalized[:pos].split()
        original_words = text.split()
        if len(words_before) <= len(original_words):
            return len(' '.join(original_words[:len(words_before)]))
    
    # Strategy 3: Partial matching (if significant portion exists)
    words = span_lower.split()
    if len(words) > 1:
        for i in range(len(words)):
            for j in range(i + 1, len(words) + 1):
                sub_span = ' '.join(words[i:j])
                # Only consider if it's a significant portion
                if len(sub_span) >= len(span_lower) * 0.75:
                    pos = text_lower.find(sub_span)
                    if pos != -1:
                        return pos
    
    # Strategy 4: Single word matching (last resort)
    if len(words) == 1 and len(span_lower) > 3:
        pos = text_lower.find(span_lower)
        if pos != -1:
            return pos
    
    # Span not found - likely an LLM hallucination
    return -1

if __name__ == "__main__":
    input_file = "C:\\Users\\norouzin\\Desktop\\JointLearning\\datasets\\pseudo_annotated_data\\llama3_8b_raw.jsonl"
    output_file = "C:\\Users\\norouzin\\Desktop\\JointLearning\\datasets\\pseudo_annotated_data\\llama3_8b_doccano.jsonl"
    stats = convert_llama3_to_doccano(input_file, output_file)
    print(f"Conversion completed with stats: {stats}")

    # convert to csv
    import pandas as pd
    df = pd.read_json(output_file, lines=True)
    df.to_csv(output_file.replace('.jsonl', '.csv'), index=False)
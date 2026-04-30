
"""Robust LLM output to Doccano format converter.

Handles raw LLM output (JSON string or dict), extracts causal relations
with proper dual-role entity handling for causal chains, and outputs
Doccano-format DataFrames.
"""

import json
import re
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union


def _strip_thinking(raw: str) -> str:
    """Remove <think>...</think> reasoning blocks from model output."""
    import re
    # Remove complete think blocks
    cleaned = re.sub(r'<think>.*?</think>\s*', '', raw, flags=re.DOTALL)
    # If only opening tag (unclosed), remove everything after it
    if '<think>' in cleaned and '</think>' not in cleaned:
        cleaned = cleaned.split('<think>')[0]
    return cleaned.strip()


def parse_llm_output(raw: str) -> Dict:
    """Extract JSON dict from raw LLM output string.

    Handles: bare JSON, JSON-in-text, double-escaped JSON, code-fenced JSON,
    and outputs with <think> reasoning blocks.
    """
    if not isinstance(raw, str):
        return {"text": "", "causal": False, "relations": []}

    # Strip any <think> reasoning blocks before parsing
    raw = _strip_thinking(raw)

    # Try direct parse first
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
        # If it parsed to a string, try parsing again (double-escaped)
        if isinstance(data, str):
            try:
                inner = json.loads(data)
                if isinstance(inner, dict):
                    return inner
            except (json.JSONDecodeError, TypeError):
                pass
    except (json.JSONDecodeError, TypeError):
        pass

    # Try to find JSON object in the text
    # First, check for code-fenced JSON
    for fence_pat in [r'```json\s*(\{.*?\})\s*```', r'```\s*(\{.*?\})\s*```']:
        match = re.search(fence_pat, raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, TypeError):
                pass

    # Bracket-counting JSON extraction (handles nested braces)
    start = raw.find('{')
    if start != -1:
        depth = 0
        end = -1
        for i in range(start, len(raw)):
            if raw[i] == '{':
                depth += 1
            elif raw[i] == '}':
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end != -1:
            try:
                data = json.loads(raw[start:end + 1])
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, TypeError):
                pass

    # Try up to the last closing brace (handles garbage after valid JSON)
    if start != -1:
        last_brace = raw.rfind('}')
        if last_brace > start:
            try:
                data = json.loads(raw[start:last_brace + 1])
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, TypeError):
                pass

    # Try completing unclosed JSON by appending closing brackets
    if start != -1 and end == -1:
        for closer in ['}]', ']}', '}}]', '}]}', '"}]}']:
            try:
                data = json.loads(raw[start:] + closer)
                if isinstance(data, dict):
                    return data
            except (json.JSONDecodeError, TypeError):
                pass

    # Last resort: try to extract text and treat as non-causal
    text_match = re.search(r'"text"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    if text_match:
        text = text_match.group(1).replace('\\"', '"')
        return {"text": text, "causal": False, "relations": []}

    return {"text": "", "causal": False, "relations": []}


def find_span_offset(text: str, span: str) -> Optional[int]:
    """Find span start offset in text. Returns None if not found."""
    if not span or not text:
        return None
    text_lower = text.lower()
    span_lower = span.lower()

    # Direct match
    pos = text_lower.find(span_lower)
    if pos != -1:
        return pos

    # Normalized whitespace
    span_norm = ' '.join(span_lower.split())
    text_norm = ' '.join(text_lower.split())
    pos = text_norm.find(span_norm)
    if pos != -1:
        words_before = text_norm[:pos].split()
        if pos == 0:
            return 0
        original = text.split()
        return len(' '.join(original[:len(words_before)])) + (1 if len(words_before) > 0 else 0)

    # Try partial match (at least 75% of span)
    words = span_lower.split()
    if len(words) > 1:
        for i in range(len(words)):
            for j in range(i + 1, len(words) + 1):
                sub = ' '.join(words[i:j])
                if len(sub) >= len(span_lower) * 0.75:
                    pos = text_lower.find(sub)
                    if pos != -1:
                        return pos

    return None


def convert_raw_to_doccano(raw_data: Union[str, List[Dict], List[str]]) -> pd.DataFrame:
    """Convert LLM raw outputs to Doccano-format DataFrame.

    Args:
        raw_data: Either a file path (JSONL), list of dicts (already parsed),
                 or list of strings (raw LLM outputs).

    Returns:
        DataFrame with columns: id, text, entities, relations, Comments
    """
    # Load data
    if isinstance(raw_data, str):
        with open(raw_data, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"Loaded {len(lines)} lines from {raw_data}")
    elif isinstance(raw_data, list):
        lines = raw_data
    else:
        raise ValueError("raw_data must be file path or list")

    stats = {'total': 0, 'causal': 0, 'non_causal': 0, 'errors': 0, 'entities': 0, 'relations': 0}
    samples = []

    for i, line in enumerate(lines):
        if i % 500 == 0:
            print(f"Processing {i}/{len(lines)}")

        stats['total'] += 1

        # Parse line
        if isinstance(line, dict):
            data = line
        elif isinstance(line, str):
            data = parse_llm_output(line.strip())
        else:
            data = {"text": "", "causal": False, "relations": []}

        text = data.get('text', '').strip()
        is_causal = data.get('causal', False)
        relations_raw = data.get('relations', [])

        if not is_causal or not relations_raw or not isinstance(relations_raw, list):
            # Non-causal sample
            processed_text = text if text.endswith(";;") else (text + ";;" if text else ";;")
            entity = {
                'id': stats['entities'],
                'label': 'non-causal',
                'start_offset': 0,
                'end_offset': len(processed_text) - 2,
            }
            stats['entities'] += 1
            stats['non_causal'] += 1
            samples.append({
                'id': i,
                'text': processed_text,
                'entities': str([entity]),
                'relations': '[]',
                'Comments': '[]',
            })
            continue

        # Causal sample - build entities and relations
        try:
            # Collect unique spans and their roles
            span_roles: Dict[str, set] = {}
            for rel in relations_raw:
                cause = rel.get('cause', '').strip()
                effect = rel.get('effect', '').strip()
                if cause:
                    span_roles.setdefault(cause, set()).add('cause')
                if effect:
                    span_roles.setdefault(effect, set()).add('effect')

            # Create entities: one per span-role pair (dual-role spans get two entities)
            entities = []
            span_entities: Dict[str, List[Dict]] = {}
            for span_text, roles in span_roles.items():
                offset = find_span_offset(text, span_text)
                if offset is None:
                    continue  # Hallucinated span, skip
                end_offset = offset + len(span_text)
                for role in sorted(roles):
                    ent = {
                        'id': stats['entities'],
                        'label': role,
                        'start_offset': offset,
                        'end_offset': end_offset,
                    }
                    stats['entities'] += 1
                    entities.append(ent)
                    span_entities.setdefault(span_text, []).append(ent)

            # Create relations
            relations = []
            for rel in relations_raw:
                cause_span = rel.get('cause', '').strip()
                effect_span = rel.get('effect', '').strip()
                if cause_span not in span_entities or effect_span not in span_entities:
                    continue
                # Find the cause entity (labeled 'cause') and effect entity (labeled 'effect')
                cause_ent = next((e for e in span_entities[cause_span] if e['label'] == 'cause'), None)
                effect_ent = next((e for e in span_entities[effect_span] if e['label'] == 'effect'), None)
                if cause_ent and effect_ent:
                    relations.append({
                        'id': stats['relations'],
                        'from_id': cause_ent['id'],
                        'to_id': effect_ent['id'],
                        'type': 'Rel_CE',
                    })
                    stats['relations'] += 1

            if not entities:
                # All spans were hallucinated, treat as non-causal
                processed_text = text if text.endswith(";;") else (text + ";;" if text else ";;")
                entity = {
                    'id': stats['entities'],
                    'label': 'non-causal',
                    'start_offset': 0,
                    'end_offset': len(processed_text) - 2,
                }
                stats['entities'] += 1
                stats['non_causal'] += 1
                samples.append({
                    'id': i,
                    'text': processed_text,
                    'entities': str([entity]),
                    'relations': '[]',
                    'Comments': '[]',
                })
            else:
                stats['causal'] += 1
                processed_text = text if text.endswith(";;") else (text + ";;" if text else ";;")
                samples.append({
                    'id': i,
                    'text': processed_text,
                    'entities': str(entities),
                    'relations': str(relations),
                    'Comments': '[]',
                })
        except Exception as e:
            stats['errors'] += 1
            stats['non_causal'] += 1
            processed_text = text if text.endswith(";;") else (text + ";;" if text else ";;")
            entity = {
                'id': stats['entities'],
                'label': 'non-causal',
                'start_offset': 0,
                'end_offset': len(processed_text) - 2,
            }
            stats['entities'] += 1
            samples.append({
                'id': i,
                'text': processed_text,
                'entities': str([entity]),
                'relations': '[]',
                'Comments': '[]',
            })

    print(f"\n{'='*60}")
    print("CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"Total: {stats['total']}")
    print(f"Causal: {stats['causal']} ({100*stats['causal']/max(1,stats['total']):.1f}%)")
    print(f"Non-causal: {stats['non_causal']} ({100*stats['non_causal']/max(1,stats['total']):.1f}%)")
    print(f"Errors: {stats['errors']}")
    print(f"Entities: {stats['entities']}, Relations: {stats['relations']}")
    print(f"{'='*60}")

    return pd.DataFrame(samples)

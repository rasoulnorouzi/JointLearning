import difflib
import json
import re
from tqdm import tqdm

"""
llsm outputs formatted as:
{
  "text": "[exact input sentence]",
  "causal": true,
  "relations": [
     {
       "cause": "[exact cause text 1]",
       "effect": "[exact effect text 1]",
       "polarity": "[Polarity1]"
     },
     {
       "cause": "[exact cause text 2]",
       "effect": "[exact effect text 2]",
       "polarity": "[Polarity2]"
     }
     // ... potentially more relations
  ]
}
"""

def locate_best_matching_span(text, phrase, threshold=0.8):
    """
    Attempts to find the best fuzzy match for a given phrase within a larger text.
    This is useful when phrases provided by LLMs don't exactly match the sentence structure
    due to lemmatization, paraphrasing, or slight rewording.

    Parameters:
        text (str): The full sentence in which the phrase should appear.
        phrase (str): The causal or effect phrase to locate.
        threshold (float): Minimum similarity score for accepting a fuzzy match.

    Returns:
        tuple[int, int] | None: Start and end character indices of the matched span, or None if not found.
    """
    phrase_len = len(phrase)
    best_match = None
    highest_ratio = 0

    for i in range(len(text) - phrase_len + 1):
        window = text[i:i + phrase_len + 20]
        ratio = difflib.SequenceMatcher(None, phrase.lower(), window.lower()).ratio()
        if ratio > highest_ratio and ratio >= threshold:
            highest_ratio = ratio
            best_match = (i, i + len(window))

    return best_match

def clean_json_code_block(s):
    """
    Removes Markdown code block markers (e.g., ```json ... ```) and trims whitespace.
    """
    if not isinstance(s, str):
        return s
    s = s.strip()
    # Remove triple backticks and optional language specifier at the start
    if s.startswith("```"):
        lines = s.splitlines()
        # Remove first line if it starts with ```
        if lines and lines[0].startswith("```"):
            lines = lines[1:]  # Fixed: Was incorrectly nested as [lines[1:]]
        # Remove last line if it is ```
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    # Remove leading/trailing single backticks (rare)
    s = re.sub(r"^`+|`+$", "", s).strip()
    return s

def safe_json_loads(s):
    """
    Attempts to fix common JSON formatting issues from LLM outputs before parsing.
    Always returns a dict with at least 'text', 'relations', and 'causal' keys.
    """
    if not isinstance(s, str):
        print(f"safe_json_loads: Input is not a string: {type(s)}")
        return {"text": "", "relations": [], "causal": None}

    original_string_for_debugging = s

    # --- Step 1: Clean Markdown code blocks and outer whitespace ---
    s = clean_json_code_block(s)
    if not s:
        print("safe_json_loads: String is empty after cleaning code blocks.")
        return {"text": "", "relations": [], "causal": None}

    # --- Step 2: Remove problematic control characters (except \t, \n, \r) ---
    s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', s)
    
    # --- Special Case: Check if input is completely malformed and try to repair structure ---
    if not s.strip().startswith('{') and not s.strip().startswith('['):
        if re.search(r'"text"\s*:', s):
            s = '{' + s
        else:
            if ':' in s and ('"' in s or "'" in s):
                s = '{' + s + '}'
    if not s.strip().endswith('}') and not s.strip().endswith(']'):
        if s.strip().startswith('{'):
            s = s + '}'
        elif s.strip().startswith('['):
            s = s + ']'

    max_attempts = 10
    last_error = None

    for attempt in range(max_attempts):
        try:
            obj = json.loads(s)
            # Ensure template keys exist
            if "text" not in obj:
                obj["text"] = ""
            if "relations" not in obj:
                obj["relations"] = []
            if "causal" not in obj:
                obj["causal"] = None
            return obj
        except json.JSONDecodeError as e:
            last_error = e
            applied_fix = False
            error_msg = str(e)
            # ...existing fix steps...
            # Fix for "Expecting property name enclosed in double quotes"
            if "Expecting property name enclosed in double quotes" in error_msg:
                s_new = re.sub(r'([{,]\s*)([a-zA-Z0-9_\-]+)(\s*:)', r'\1"\2"\3', s)
                if s_new != s:
                    s = s_new
                    applied_fix = True
                    continue
            if "Expecting ',' delimiter" in error_msg or "Expecting ':' delimiter" in error_msg:
                open_braces = s.count('{')
                close_braces = s.count('}')
                open_brackets = s.count('[')
                close_brackets = s.count(']')
                if open_braces > close_braces:
                    s = s + ('}' * (open_braces - close_braces))
                    applied_fix = True
                if open_brackets > close_braces:
                    s = s + (']' * (open_brackets - close_braces))
                    applied_fix = True
                if applied_fix:
                    continue
            s_new = s.replace('tban', 'than').replace('tbe', 'the').replace('cboosing', 'choosing')
            s_new = s_new.replace(''', "'").replace(''', "'").replace('"', '"').replace('"', '"')
            if s_new != s:
                s = s_new
                applied_fix = True
                continue
            s_new = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)
            if s_new != s:
                s = s_new
                applied_fix = True
                continue
            s_new = re.sub(r'\\u([^0-9a-fA-F]|$|[0-9a-fA-F]{0,3}$)', r'\\\\u\1', s)
            if s_new != s:
                s = s_new
                applied_fix = True
                continue
            s_new = re.sub(r'(?<!\\)\n', r'\\n', s)
            if s_new != s:
                s = s_new
                applied_fix = True
                continue
            # ...existing code for quote fixes, trailing commas, etc...
            # Fix 3n: Remove whitespace and newlines right after '{' if it's followed by '"'
            s_new = re.sub(r'(\{)\s+(")', r'\1\2', s)
            if s_new != s:
                s = s_new
                applied_fix = True
                continue
            if not applied_fix:
                break

    # --- Last Resort: Extract fields manually to match template ---
    text_val = ""
    relations_val = []
    causal_val = None

    # Try to extract text
    text_match = re.search(r'"text"\s*:\s*"([^"]*)"', original_string_for_debugging)
    if text_match:
        text_val = text_match.group(1)

    # Try to extract causal
    causal_match = re.search(r'"causal"\s*:\s*(true|false|null)', original_string_for_debugging, re.IGNORECASE)
    if causal_match:
        val = causal_match.group(1).lower()
        if val == "true":
            causal_val = True
        elif val == "false":
            causal_val = False
        else:
            causal_val = None

    # Try to extract relations as a list of dicts
    relations_match = re.search(r'"relations"\s*:\s*(\[[^\]]*\])', original_string_for_debugging, re.DOTALL)
    if relations_match:
        try:
            rels = json.loads(relations_match.group(1))
            # Ensure each relation has the required keys
            relations_val = [
                {
                    "cause": rel.get("cause", ""),
                    "effect": rel.get("effect", ""),
                    "polarity": rel.get("polarity", "")
                }
                for rel in rels if isinstance(rel, dict)
            ]
        except Exception:
            relations_val = []
    # If not found, try to extract at least one cause/effect/polarity
    if not relations_val:
        rels = []
        rel_pattern = re.compile(
            r'\{\s*"cause"\s*:\s*"([^"]*)"\s*,\s*"effect"\s*:\s*"([^"]*)"\s*,\s*"polarity"\s*:\s*"([^"]*)"\s*\}',
            re.DOTALL)
        for m in rel_pattern.finditer(original_string_for_debugging):
            rels.append({"cause": m.group(1), "effect": m.group(2), "polarity": m.group(3)})
        relations_val = rels

    # Always return the template, even if all fields are empty
    return {"text": text_val, "relations": relations_val, "causal": causal_val}

def convert_llm_output_to_doccano_format(ollama_data):
    """
    Transform LLM outputs into Doccano JSON-Lines.

    Rules enforced here
    -------------------
    • Causal sentence  = at least one relation WHERE
        – 'cause'  is a non-empty string AND
        – 'effect' is a non-empty string
      Everything else is tagged as non-causal.

    • Partial relations (only cause OR only effect) are
      recorded in `Comments` but NOT exported to Doccano.
    """
    doccano_formatted   = []
    global_entity_id    = 1_000
    global_relation_id  =   500

    for i, item in enumerate(tqdm(ollama_data, desc="Converting")):
        output_raw = clean_json_code_block(item.get("output", ""))

        # Always land on a dict with the three keys
        sd = safe_json_loads(output_raw)

        text       = sd["text"]
        rel_raw    = sd.get("relations", []) or []
        # keep only **complete** relations
        rel_valid  = [r for r in rel_raw if r.get("cause") and r.get("effect")]

        # final causal flag (rule-based)
        is_causal  = bool(rel_valid) and sd.get("causal", True) is True

        base = {
            "id"        : i + 1,
            "text"      : text,
            "entities"  : [],
            "relations" : [],
            "Comments"  : []
        }

        # record dropped partial relations for audit
        dropped = [r for r in rel_raw if r not in rel_valid]
        if dropped:
            base["Comments"].append(
                f"{len(dropped)} partial relation(s) ignored (cause xor effect missing)"
            )

        if not is_causal:
            # ONE span covering the whole sentence
            base["entities"].append({
                "id"           : global_entity_id,
                "label"        : "non-causal",
                "start_offset" : 0,
                "end_offset"   : len(text)
            })
            global_entity_id += 1
            doccano_formatted.append(base)
            continue

        # ---- causal path --------------------------------------------------
        ent_index = {}                   # phrase ➜ entity_id

        for rel in rel_valid:
            for role in ("cause", "effect"):
                phrase = rel[role]
                if phrase in ent_index:
                    continue

                # try exact, then fuzzy match
                exact = re.search(re.escape(phrase), text)
                span  = exact.span() if exact else locate_best_matching_span(text, phrase)
                if not span:
                    base["Comments"].append(
                        f"Could not locate {role} phrase: '{phrase}'"
                    )
                    continue

                ent_id = global_entity_id
                global_entity_id += 1
                ent_index[phrase] = ent_id

                base["entities"].append({
                    "id"           : ent_id,
                    "label"        : role,
                    "start_offset" : span[0],
                    "end_offset"   : span[1]
                })

            # only add relation if BOTH ends were found
            c_id = ent_index.get(rel["cause"])
            e_id = ent_index.get(rel["effect"])
            if c_id and e_id:
                base["relations"].append({
                    "id"      : global_relation_id,
                    "from_id" : c_id,
                    "to_id"   : e_id,
                    "type"    : "Rel_CE"
                })
                global_relation_id += 1

        doccano_formatted.append(base)

    return doccano_formatted

"""
How to use: 

# llm33b_raw.jsonl
with open("llm33b_raw.jsonl", "r", encoding="utf-8") as f:
    llm33b_data = [json.loads(line) for line in f.readlines()]
doccano_llm33b = convert_llm_output_to_doccano_format(llm33b_data)

# Save the Doccano formatted data to a JSON file
with open("annotation_datasets/doccano_llm33b.jsonl", "w", encoding="utf-8") as f:
    for sample in doccano_llm33b:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

# llama38b_raw.jsonl
with open("llama38b_raw.jsonl", "r", encoding="utf-8") as f:
    llama38b_data = [json.loads(line) for line in f.readlines()]    

doccano_llama38b = convert_llm_output_to_doccano_format(llama38b_data)

# Save the Doccano formatted data to a JSON file
with open("annotation_datasets/doccano_llama38b.jsonl", "w", encoding="utf-8") as f:
    for sample in doccano_llama38b:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

# gemma34b_raw.jsonl
with open("gemma34b_raw.jsonl", "r", encoding="utf-8") as f:
    gemma34b_data = [json.loads(line) for line in f.readlines()]

doccano_gemma34b = convert_llm_output_to_doccano_format(gemma34b_data)

# Save the Doccano formatted data to a JSON file
with open("annotation_datasets/doccano_gemma34b.jsonl", "w", encoding="utf-8") as f:
    for sample in doccano_gemma34b:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

# gemma312b_raw.jsonl
with open("gemma312b_raw.jsonl", "r", encoding="utf-8") as f:
    gemma312b_data = [json.loads(line) for line in f.readlines()]

doccano_gemma312b = convert_llm_output_to_doccano_format(gemma312b_data)

# Save the Doccano formatted data to a JSON file
with open("annotation_datasets/doccano_gemma312b.jsonl", "w", encoding="utf-8") as f:
    for sample in doccano_gemma312b:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

"""
llama38b_path  = r"C:\Users\norouzin\Desktop\JointLearning\datasets\pseudo_annotate_data\llama3_8b_raw.jsonl"
with open(llama38b_path, "r", encoding="utf-8") as f:
    parsed_lines = [json.loads(line) for line in f.readlines()]
    llama38b_data = [{"output": text_content} for text_content in parsed_lines]
doccano_llama38b = convert_llm_output_to_doccano_format(llama38b_data)
# Save the Doccano formatted data to a JSON file
with open("C:\\Users\\norouzin\\Desktop\\JointLearning\\src\\causal_pseudo_labeling\\annotation_datasets\\doccano_llama38b.jsonl", "w", encoding="utf-8") as f:
    for sample in doccano_llama38b:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")
#!/usr/bin/env python3
"""
Sequential (pipeline) evaluation: chain CLS → BIO → REL models at test time.

Loads the three separately-trained models and processes each sentence through
a strict pipeline where the CLS decision is FINAL — downstream failures are
NOT allowed to retroactively flip the causal flag.

    1. CLS model  → is_causal  (threshold 0.5) — FINAL, never changed
    2. If causal: BIO model → spans  (may be empty — BIO's error)
    3. If spans:  REL model → relations  (may be empty — REL's error)

Each task's errors are attributed to that task only.

Usage:
    python evaluate_sequential.py
"""
import sys
import os
import json
import re

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
for d in [SRC_DIR, os.path.join(SRC_DIR, "jointlearning")]:
    if d not in sys.path:
        sys.path.insert(0, d)

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from jointlearning.model import JointCausalModel
from analysis.causal_eval import evaluate, display_results

from cmp_config import (
    MODEL_PATHS, TEST_DATA_PATH, PREDICTIONS_DIR, DEVICE, SEED,
    MODEL_CONFIG, id2label_bio, id2label_cls, id2label_rel,
)

REL_THRESHOLD = 0.5
BATCH_SIZE = 32


def load_model(ckpt_path: str, device: torch.device) -> JointCausalModel:
    model = JointCausalModel(**MODEL_CONFIG)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def sequential_predict(
    cls_model: JointCausalModel,
    bio_model: JointCausalModel,
    rel_model: JointCausalModel,
    sentences: list,
    tokenizer,
    device: torch.device,
) -> list:
    """Chain CLS → BIO → REL. CLS decision is FINAL — never flipped downstream.

    Returns list of dicts with keys: text, causal, spans, relations.
    Spans include character offsets for direct Doccano entity creation.
    """
    results = []

    # Strip ;; suffix to match CausalDataset tokenization (model was trained this way)
    clean_sents = [s.rstrip(';') for s in sentences]
    enc = tokenizer(
        clean_sents, return_tensors="pt", truncation=True, max_length=512, padding=True,
        return_offsets_mapping=True,
    )
    offset_mapping = enc.pop("offset_mapping")
    enc = {k: v.to(device) for k, v in enc.items()}
    offset_mapping = offset_mapping.to(device)

    with torch.no_grad():
        cls_out = cls_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        cls_logits = cls_out["cls_logits"]

        bio_out = bio_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        bio_emissions = bio_out["bio_emissions"]

        for i in range(len(sentences)):
            # --- Stage 1: CLS (FINAL — never changed) ---
            prob_causal = torch.softmax(cls_logits[i], dim=-1)[1].item()
            is_causal = prob_causal >= 0.5

            if not is_causal:
                results.append({
                    "text": sentences[i], "causal": False,
                    "spans": [], "relations": [],
                })
                continue

            # --- Stage 2: BIO ---
            seq_len = enc["attention_mask"][i].sum().item()
            tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][i][:seq_len])
            bio_ids = bio_emissions[i][:seq_len].argmax(-1).tolist()
            bio_labels = [id2label_bio[j] for j in bio_ids]

            fixed_labels = bio_model._apply_bio_rules(tokens, bio_labels)
            spans = bio_model._merge_spans(tokens, fixed_labels, tokenizer)

            # Convert token positions → character offsets using offset_mapping
            offsets = offset_mapping[i][:seq_len].cpu().tolist()
            span_dicts = []
            for sp in spans:
                if sp.start_tok < len(offsets) and sp.end_tok < len(offsets):
                    start_char = offsets[sp.start_tok][0]
                    end_char = offsets[sp.end_tok][1]
                    role_label = "cause" if sp.role in ("C", "CE") else "effect" if sp.role == "E" else sp.role
                    span_dicts.append({
                        "role": sp.role,
                        "label": role_label,
                        "text": sp.text,
                        "start_char": start_char,
                        "end_char": end_char,
                        "start_tok": sp.start_tok,
                        "end_tok": sp.end_tok,
                    })

            cause_spans = [s for s in spans if s.role in ("C", "CE")]
            effect_spans = [s for s in spans if s.role in ("E", "CE")]

            # Build a lookup: token (start, end) → char offsets
            tok2char = {}
            for sd in span_dicts:
                tok2char[(sd["start_tok"], sd["end_tok"])] = (sd["start_char"], sd["end_char"])

            # --- Stage 3: REL (may produce empty relations — REL's error) ---
            rels = []
            pair_meta = []
            for c in cause_spans:
                for e in effect_spans:
                    if (c.start_tok == e.start_tok and c.end_tok == e.end_tok
                            and c.role != "CE" and e.role != "CE"):
                        continue
                    pair_meta.append((c, e))

            if pair_meta:
                pair_batch = torch.zeros(len(pair_meta), dtype=torch.long, device=device)
                c_starts = torch.tensor([c.start_tok for c, _ in pair_meta], device=device)
                c_ends = torch.tensor([c.end_tok for c, _ in pair_meta], device=device)
                e_starts = torch.tensor([e.start_tok for _, e in pair_meta], device=device)
                e_ends = torch.tensor([e.end_tok for _, e in pair_meta], device=device)

                rel_out = rel_model(
                    input_ids=enc["input_ids"][i:i+1],
                    attention_mask=enc["attention_mask"][i:i+1],
                    pair_batch=pair_batch,
                    cause_starts=c_starts,
                    cause_ends=c_ends,
                    effect_starts=e_starts,
                    effect_ends=e_ends,
                )
                rel_logits = rel_out["rel_logits"]
                probs = torch.softmax(rel_logits, dim=-1)[:, 1].tolist()

                for (c, e), p in zip(pair_meta, probs):
                    if p >= REL_THRESHOLD and c.text.lower() != e.text.lower():
                        c_offsets = tok2char.get((c.start_tok, c.end_tok), (None, None))
                        e_offsets = tok2char.get((e.start_tok, e.end_tok), (None, None))
                        rels.append({
                            "cause": c.text, "effect": e.text, "type": "Rel_CE",
                            "cause_start": c_offsets[0], "cause_end": c_offsets[1],
                            "effect_start": e_offsets[0], "effect_end": e_offsets[1],
                        })

            # CLS stays True even if spans or relations are empty
            results.append({
                "text": sentences[i],
                "causal": True,
                "spans": span_dicts,
                "relations": rels,
            })

    return results


# ---------------------------------------------------------------------------
# Doccano converter — creates entities from span character offsets,
# NOT from relations. This preserves CLS decision fidelity.
# ---------------------------------------------------------------------------

def _find_span_offset(text: str, span_text: str) -> int:
    """Find start character offset of span_text in text (case-insensitive)."""
    if not span_text or not text:
        return -1
    tl = text.lower()
    sl = span_text.lower()
    pos = tl.find(sl)
    if pos != -1:
        return pos
    # Normalised whitespace fallback
    sn = " ".join(sl.split())
    tn = " ".join(tl.split())
    pos = tn.find(sn)
    if pos != -1:
        words_before = tn[:pos].split()
        original_words = text.split()
        if len(words_before) <= len(original_words):
            return len(" ".join(original_words[:len(words_before)]))
    return -1


def sequential_to_doccano(seq_results: list) -> pd.DataFrame:
    """Convert sequential_predict output to Doccano-format DataFrame.

    Entities are built directly from BIO span character offsets.
    Relations are built from REL model predictions.
    The CLS decision is preserved: causal=True always gets entities.
    """
    rows = []
    stats = {"total": len(seq_results), "causal": 0, "non_causal": 0,
             "entities": 0, "relations": 0, "errors": 0}

    for idx, item in enumerate(seq_results):
        text = item.get("text", "")
        is_causal = item.get("causal", False)
        spans = item.get("spans", [])
        relations = item.get("relations", [])

        if not is_causal:
            sample = {
                "id": idx,
                "text": text,
                "entities": str([{"id": 0, "label": "non-causal",
                                  "start_offset": 0, "end_offset": len(text.rstrip(';'))}]),
                "relations": "[]",
                "Comments": "[]",
            }
            stats["non_causal"] += 1
            stats["entities"] += 1
            rows.append(sample)
            continue

        stats["causal"] += 1

        # --- Build entities from BIO spans ---
        # Group by (start_char, end_char) → set of labels for dual-role handling
        span_groups = {}  # (start, end) -> {"labels": set(), "text": str}
        for sp in spans:
            key = (sp["start_char"], sp["end_char"])
            if key not in span_groups:
                span_groups[key] = {"labels": set(), "text": sp["text"]}
            # Map CE → both cause and effect
            role = sp["role"]
            if role in ("C", "CE"):
                span_groups[key]["labels"].add("cause")
            if role in ("E", "CE"):
                span_groups[key]["labels"].add("effect")

        entities = []
        entity_id = 0
        # Map (start, end, label) → entity_id for relation linking
        entity_map = {}

        for (start, end), info in sorted(span_groups.items()):
            for label in sorted(info["labels"]):
                eid = entity_id
                entities.append({
                    "id": eid, "label": label,
                    "start_offset": start, "end_offset": end,
                })
                entity_map[(start, end, label)] = eid
                entity_id += 1

        # If no valid entities, create a non-causal fallback to preserve sample count
        if not entities:
            entities = [{"id": 0, "label": "non-causal",
                         "start_offset": 0, "end_offset": len(text.rstrip(';'))}]
            stats["entities"] += 1
            sample = {
                "id": idx, "text": text,
                "entities": str(entities), "relations": "[]", "Comments": "[]",
            }
            rows.append(sample)
            continue

        stats["entities"] += len(entities)

        # --- Build relations from REL predictions ---
        # Use character offsets carried through from the span positions
        relation_list = []
        rel_id = 0
        seen_rels = set()
        for rel in relations:
            cause_start = rel.get("cause_start")
            cause_end = rel.get("cause_end")
            effect_start = rel.get("effect_start")
            effect_end = rel.get("effect_end")

            # Skip if offsets are missing or invalid
            if None in (cause_start, cause_end, effect_start, effect_end):
                continue

            cause_key = (cause_start, cause_end, "cause")
            effect_key = (effect_start, effect_end, "effect")

            if cause_key in entity_map and effect_key in entity_map:
                key = (entity_map[cause_key], entity_map[effect_key])
                if key not in seen_rels:
                    seen_rels.add(key)
                    relation_list.append({
                        "id": rel_id,
                        "from_id": entity_map[cause_key],
                        "to_id": entity_map[effect_key],
                        "type": "Rel_CE",
                    })
                    rel_id += 1

        stats["relations"] += len(relation_list)

        sample = {
            "id": idx,
            "text": text,
            "entities": str(entities),
            "relations": str(relation_list),
            "Comments": "[]",
        }
        rows.append(sample)

    df = pd.DataFrame(rows)

    print(f"\n{'='*60}")
    print("SEQUENTIAL → DOCCANO CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"Total samples: {stats['total']}")
    print(f"Causal: {stats['causal']}  Non-causal: {stats['non_causal']}")
    print(f"Total entities: {stats['entities']}")
    print(f"Total relations: {stats['relations']}")
    if stats['errors']:
        print(f"Errors: {stats['errors']}")
    print(f"{'='*60}")

    return df


def main():
    print(f"Device: {DEVICE}")
    print(f"Loading models...")

    cls_model = load_model(MODEL_PATHS["cls"], DEVICE)
    bio_model = load_model(MODEL_PATHS["bio"], DEVICE)
    rel_model = load_model(MODEL_PATHS["rel"], DEVICE)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["encoder_name"])
    test_df = pd.read_csv(TEST_DATA_PATH)
    texts = test_df["text"].tolist()
    print(f"Test sentences: {len(texts)}")

    # --- Predict ---
    all_results = []
    num_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE
    for bi in tqdm(range(num_batches), desc="Sequential predict"):
        batch = texts[bi * BATCH_SIZE : (bi + 1) * BATCH_SIZE]
        all_results.extend(sequential_predict(cls_model, bio_model, rel_model, batch, tokenizer, DEVICE))

    # Quick stats
    n_causal = sum(1 for r in all_results if r["causal"])
    n_rels = sum(len(r.get("relations", [])) for r in all_results)
    n_spans = sum(len(r.get("spans", [])) for r in all_results)
    print(f"Predicted: {n_causal}/{len(all_results)} causal, {n_spans} spans, {n_rels} relations")

    # --- Save raw predictions ---
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    raw_path = os.path.join(PREDICTIONS_DIR, "sequential_predictions.json")

    # Clean spans for JSON (remove non-serializable fields)
    clean_results = []
    for r in all_results:
        cr = {"text": r["text"], "causal": r["causal"], "relations": r["relations"]}
        cr["spans"] = [{k: v for k, v in s.items() if k not in ("start_tok", "end_tok")}
                       for s in r.get("spans", [])]
        clean_results.append(cr)
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, ensure_ascii=False, indent=2)
    print(f"Raw predictions saved to {raw_path}")

    # --- Convert to Doccano (using custom converter) ---
    pred_df = sequential_to_doccano(all_results)
    csv_path = os.path.join(PREDICTIONS_DIR, "sequential_predictions_doccano.csv")
    pred_df.to_csv(csv_path, index=False)
    print(f"Doccano CSV saved to {csv_path}")

    # --- Evaluate across all 4 combos ---
    gold_df = pd.read_csv(TEST_DATA_PATH)

    scenarios = ["end_to_end", "filtered_causal"]
    eval_modes = ["discovery", "coverage"]

    print("\n" + "=" * 70)
    print("  SEQUENTIAL PIPELINE EVALUATION RESULTS")
    print("=" * 70)

    all_metrics = []
    for scenario in scenarios:
        for eval_mode in eval_modes:
            result = evaluate(gold_df, pred_df, scenario=scenario, eval_mode=eval_mode)
            display_results(result, title_prefix=f"Sequential | {scenario} | {eval_mode}")

            all_metrics.append({
                "scenario": scenario,
                "eval_mode": eval_mode,
                "Task1_F1": result["Task1"]["F1"],
                "Task2_Macro": result["Task2_macro"]["F1"],
                "Task3_F1": result["Task3"]["F1"],
                "Total_Macro": result["Total_Macro"]["F1"],
            })
            print(f"  Task1 F1={result['Task1']['F1']:.4f}  "
                  f"Task2 Macro={result['Task2_macro']['F1']:.4f}  "
                  f"Task3 F1={result['Task3']['F1']:.4f}  "
                  f"Total Macro={result['Total_Macro']['F1']:.4f}")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join(PREDICTIONS_DIR, "sequential_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMetrics saved to {metrics_path}")
    print(metrics_df.to_string(index=False))

    for m in [cls_model, bio_model, rel_model]:
        del m
    torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()

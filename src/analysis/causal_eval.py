from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Union
import pandas as pd
import ast

# --- Type Definitions ---
Span = Tuple[int,int]
Entity = Tuple[int,int,str,str]
# A relation is represented by its source span, target span, and label
Relation = Tuple[Span, Span, str] 

# --- Core Helper Functions ---
def intervals_overlap(a: Span, b: Span) -> bool:
    """Check if two span intervals overlap."""
    return max(a[0], b[0]) < min(a[1], b[1])

def _parse_ents(raw, doc_id):
    out = []
    for ent in raw.get("labels", []) or raw.get("entities", []) or raw.get("spans", []):
        if isinstance(ent, dict):
            s = ent.get("start_offset") or ent.get("start"); e = ent.get("end_offset") or ent.get("end")
            lbl = ent.get("label") or ent.get("type") or ""
            eid = str(ent.get("id") or ent.get("entity_id") or f"{doc_id}:{s}-{e}")
        else:
            if len(ent) == 3: s, e, lbl = ent
            elif len(ent) == 4: _, s, e, lbl = ent
            else: continue
            eid = f"{doc_id}:{s}-{e}"
        if s is None or e is None: continue
        out.append((int(s), int(e), str(lbl).lower(), eid))
    return out

def _parse_rels(raw):
    out = []
    for rel in raw.get("relations", []) or raw.get("links", []):
        if isinstance(rel, dict):
            src = str(rel.get("from_id") or rel.get("src")); tgt = str(rel.get("to_id") or rel.get("dst"))
            pol = str(rel.get("type") or rel.get("label") or rel.get("relation")).lower()
        else:
            if len(rel) < 3: continue
            src, tgt, pol = rel[:3]; pol = str(pol).lower()
        out.append((src, tgt, pol))
    return out

# --- Task 1: Document-level Classification ---
def _task1(gold, pred):
    tp = fp = fn = tn = 0
    for g, p in zip(gold, pred):
        gc = any(l in {"cause", "effect"} for *_, l, _ in g["entities"])
        pc = any(l in {"cause", "effect"} for *_, l, _ in p["entities"])
        if gc and pc: tp += 1
        elif gc and not pc: fn += 1
        elif not gc and pc: fp += 1
        else: tn += 1
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0
    acc = (tp + tn) / len(gold)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Precision": prec, "Recall": rec, "F1": f1, "Accuracy": acc, "N": len(gold)}


# --- Unified Metric Calculation Helpers ---

def calculate_span_metrics(golds: List[Span], preds: List[Span], mode: str) -> Tuple[int, int, int]:
    """Calculates TP, FP, FN for spans based on the evaluation mode."""
    if mode == 'discovery':
        used_preds = set()
        tp = 0
        for g in golds:
            for i, p in enumerate(preds):
                if i in used_preds: continue
                if intervals_overlap(g, p):
                    tp += 1
                    used_preds.add(i)
                    break
        fn = len(golds) - tp
        fp = sum(
            1 for i, p in enumerate(preds)
            if i not in used_preds and not any(intervals_overlap(p, g) for g in golds)
        )
        return tp, fp, fn

    elif mode == 'coverage':
        if not golds and not preds: return 0, 0, 0
        if not preds: return 0, 0, len(golds)
        if not golds: return 0, len(preds), 0

        tp = sum(1 for p in preds if any(intervals_overlap(p, g) for g in golds))
        fp = len(preds) - tp
        fn = sum(1 for g in golds if not any(intervals_overlap(g, p) for p in preds))
        return tp, fp, fn
    else:
        raise ValueError(f"Unknown evaluation mode: '{mode}'")

def calculate_relation_metrics(golds: List[Relation], preds: List[Relation], mode: str) -> Tuple[int, int, int]:
    """Calculates TP, FP, FN for relations based on the evaluation mode."""
    if mode == 'discovery':
        cnt = Counter()
        used_pred = set()
        for gs_c, gs_e, g_lab in golds:
            hit = None
            for i, (ps_c, ps_e, p_lab) in enumerate(preds):
                if i in used_pred: continue
                if intervals_overlap(gs_c, ps_c) and intervals_overlap(gs_e, ps_e):
                    hit = i
                    break
            if hit is None:
                cnt['FN'] += 1
            else:
                used_pred.add(hit)
                _, _, p_lab = preds[hit]
                if p_lab == g_lab: cnt['TP'] += 1
                else: cnt['FP'] += 1; cnt['FN'] += 1
        
        for i, (ps_c, ps_e, _) in enumerate(preds):
            if i in used_pred: continue
            if not any(intervals_overlap(ps_c, gs_c) and intervals_overlap(ps_e, gs_e) for gs_c, gs_e, _ in golds):
                cnt['FP'] += 1
        return cnt['TP'], cnt['FP'], cnt['FN']

    elif mode == 'coverage':
        tp = fp = fn = 0
        # Calculate TP and FP from the perspective of predictions
        for p_cause, p_effect, p_label in preds:
            is_match = False
            for g_cause, g_effect, g_label in golds:
                if intervals_overlap(p_cause, g_cause) and intervals_overlap(p_effect, g_effect):
                    if p_label == g_label:
                        is_match = True
                        break # Found a correct match for this prediction
            if is_match:
                tp += 1
            else:
                fp += 1 # Prediction is wrong if no matching gold relation is found
        
        # Calculate FN from the perspective of gold standards
        for g_cause, g_effect, g_label in golds:
            is_found = False
            for p_cause, p_effect, p_label in preds:
                if intervals_overlap(g_cause, p_cause) and intervals_overlap(g_effect, p_effect):
                    if g_label == p_label:
                        is_found = True
                        break # This gold relation was found
            if not is_found:
                fn += 1
        return tp, fp, fn
    else:
        raise ValueError(f"Unknown evaluation mode: '{mode}'")


# --- Task-specific Evaluation Functions ---

def _task2(gold: list, pred: list, scenario: str, eval_mode: str) -> dict:
    """Task 2: Cause/Effect Span Extraction."""
    counts = {lbl: {'TP': 0, 'FP': 0, 'FN': 0} for lbl in ('cause', 'effect')}
    for g_doc, p_doc in zip(gold, pred):
        if scenario.lower() == 'filtered_causal' and not (
            any(lbl in {'cause', 'effect'} for *_, lbl, _ in g_doc['entities']) and
            any(lbl in {'cause', 'effect'} for *_, lbl, _ in p_doc['entities'])
        ): continue
        
        gb = defaultdict(list); pb = defaultdict(list)
        for s, e, lbl, _ in g_doc['entities']:
            if lbl in counts: gb[lbl].append((s, e))
        for s, e, lbl, _ in p_doc['entities']:
            if lbl in counts: pb[lbl].append((s, e))
        
        for lbl in counts:
            tp, fp, fn = calculate_span_metrics(gb[lbl], pb[lbl], mode=eval_mode)
            counts[lbl]['TP'] += tp
            counts[lbl]['FP'] += fp
            counts[lbl]['FN'] += fn

    result = {}
    for lbl, c in counts.items():
        prec = c['TP'] / (c['TP'] + c['FP']) if (c['TP'] + c['FP']) else 0.0
        rec = c['TP'] / (c['TP'] + c['FN']) if (c['TP'] + c['FN']) else 0.0
        f1  = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        result[lbl] = {'Precision': prec, 'Recall': rec, 'F1': f1, 'TP': c['TP'], 'FP': c['FP'], 'FN': c['FN']}
    return result

def _task3(gold_docs: list, pred_docs: list, scenario: str, eval_mode: str, label_map: dict = None) -> dict:
    """Task 3: Cause→Effect Relation Classification."""
    total_tp, total_fp, total_fn = 0, 0, 0
    norm = (lambda lbl: label_map.get(lbl, lbl)) if label_map else (lambda lbl: lbl)

    for g_doc, p_doc in zip(gold_docs, pred_docs):
        if scenario.lower() == 'filtered_causal' and not (
            any(lbl in {'cause', 'effect'} for *_, lbl, _ in g_doc['entities']) and
            any(lbl in {'cause', 'effect'} for *_, lbl, _ in p_doc['entities'])
        ): continue

        g_ent = {eid: (s, e, lbl) for s, e, lbl, eid in g_doc['entities']}
        p_ent = {eid: (s, e, lbl) for s, e, lbl, eid in p_doc['entities']}
        
        gold_links = [
            ((g_ent[src][0], g_ent[src][1]), (g_ent[tgt][0], g_ent[tgt][1]), norm(lbl))
            for src, tgt, lbl in g_doc['relations']
            if src in g_ent and tgt in g_ent and g_ent[src][2] == 'cause' and g_ent[tgt][2] == 'effect'
        ]
        pred_links = [
            ((p_ent[src][0], p_ent[src][1]), (p_ent[tgt][0], p_ent[tgt][1]), norm(lbl))
            for src, tgt, lbl in p_doc['relations']
            if src in p_ent and tgt in p_ent and p_ent[src][2] == 'cause' and p_ent[tgt][2] == 'effect'
        ]

        tp, fp, fn = calculate_relation_metrics(gold_links, pred_links, mode=eval_mode)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    prec = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    rec  = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc  = total_tp / (total_tp + total_fp + total_fn) if total_tp + total_fp + total_fn else 0.0
    return {'TP': total_tp, 'FP': total_fp, 'FN': total_fn, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}


# --- Main Driver and Display Functions ---

def evaluate(gold_df, pred_df, scenario='all_documents', label_map=None, eval_mode='discovery'):
    """
    Evaluate causal extraction performance.
    
    Args:
        gold_df, pred_df: Pandas DataFrames with gold and prediction data.
        scenario: Evaluation scenario ('all_documents' or 'filtered_causal').
        label_map: Optional mapping for normalizing relation labels.
        eval_mode (str): Mode for Task 2/3 evaluation.
                         'discovery' (default): Ignores extra overlapping items.
                         'coverage': Counts all overlapping correct items as TPs.
    Returns:
        Dictionary with evaluation results.
    """
    gold = []
    pred = []
    
    def process_df(df):
        docs = []
        for _, row in df.iterrows():
            did = str(row.get('id', len(docs)))
            entities_str = row.get('entities', '[]')
            relations_str = row.get('relations', '[]')
            try:
                entities = _parse_ents({"labels": ast.literal_eval(entities_str)}, did)
            except (ValueError, SyntaxError): entities = []
            try:
                relations = _parse_rels({"relations": ast.literal_eval(relations_str)})
            except (ValueError, SyntaxError): relations = []
            docs.append({"text": str(row.get('text', "")), "entities": entities, "relations": relations})
        return docs

    gold = process_df(gold_df)
    pred = process_df(pred_df)

    if len(gold) != len(pred): raise ValueError("Length mismatch between gold and pred data.")

    task2_results = _task2(gold, pred, scenario, eval_mode=eval_mode)
    task2_macro = macro_average_task2(task2_results)
    task3_results = _task3(gold, pred, scenario, eval_mode=eval_mode, label_map=label_map)

    return {
        "Task1": _task1(gold, pred),
        "Task2": task2_results,
        "Task2_macro": task2_macro,
        "Task3": task3_results,
        "Total_Macro": total_macro_average(_task1(gold, pred), task2_macro, task3_results)
    }

def macro_average_task2(task2_result):
    keys = ["Precision", "Recall", "F1"]
    macro = {k: (task2_result["cause"][k] + task2_result["effect"][k]) / 2 for k in keys}
    macro["TP"] = task2_result["cause"]["TP"] + task2_result["effect"]["TP"]
    macro["FP"] = task2_result["cause"]["FP"] + task2_result["effect"]["FP"]
    macro["FN"] = task2_result["cause"]["FN"] + task2_result["effect"]["FN"]
    return macro

def total_macro_average(task1, task2_macro, task3):
    keys = ["Precision", "Recall", "F1"]
    return {k: (task1[k] + task2_macro[k] + task3[k]) / 3 for k in keys}

def display_results(results, title_prefix=""):
    border = "=" * 70
    title = f" {title_prefix} Results "
    print(f"\n{border}\n{title:^70}\n{border}")
    
    for task_name, metrics in results.items():
        print(f"\n--- {task_name} ---")
        if isinstance(metrics, dict):
            if any(isinstance(v, dict) for v in metrics.values()):
                for sub_task, sub_metrics in metrics.items():
                    print(f"  Label: {sub_task}")
                    for key, value in sub_metrics.items():
                        print(f"    {key:<12}: {value: >8.4f}" if isinstance(value, float) else f"    {key:<12}: {value: >8}")
            else:
                 for key, value in metrics.items():
                    print(f"  {key:<12}: {value: >8.4f}" if isinstance(value, float) else f"  {key:<12}: {value: >8}")
    print(f"\n{border}\n")

# def main():
#     gold_df = pd.read_csv(r"C:\\Users\\norouzin\\Desktop\\JointLearning\\datasets\\expert_multi_task_data\\test.csv")
#     pred_df = pd.read_csv(r"C:\\Users\\norouzin\\Desktop\\JointLearning\\predictions\\expert_bert_softmax_cls+span_doccano.csv")

#     print("\n" + "#" * 15 + " Running in 'discovery' mode (ignoring extra overlaps) " + "#" * 15)
#     results_discovery = evaluate(gold_df, pred_df, scenario='all_documents', eval_mode='discovery')
#     display_results(results_discovery, title_prefix="All Documents - 'discovery' mode")

#     print("\n" + "#" * 15 + " Running in 'coverage' mode (all overlaps are TPs) " + "#" * 15)
#     results_coverage = evaluate(gold_df, pred_df, scenario='all_documents', eval_mode='coverage')
#     display_results(results_coverage, title_prefix="All Documents - 'coverage' mode")

#     print("\n" + "#" * 15 + " Running in 'discovery' mode for Filtered Causal " + "#" * 15)
#     results_scenario_b = evaluate(gold_df, pred_df, scenario='filtered_causal', eval_mode='discovery')
#     display_results(results_scenario_b, title_prefix="Filtered Causal - 'discovery' mode")
#     print("\n" + "#" * 15 + " Running in 'coverage' mode for Filtered Causal " + "#" * 15)
#     results_scenario_b_coverage = evaluate(gold_df, pred_df, scenario='filtered_causal', eval_mode='coverage')
#     display_results(results_scenario_b_coverage, title_prefix="Filtered Causal - 'coverage' mode")
#     print("\n" + "#" * 15 + " Evaluation completed! " + "#" * 15)

# if __name__ == "__main__":
#     main()

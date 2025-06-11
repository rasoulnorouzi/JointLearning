from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import pathlib
import json

Span = Tuple[int,int]
Entity = Tuple[int,int,str,str]
Relation = Tuple[str,str,str]

def intervals_overlap(a: Span, b: Span) -> bool:
    """
    Check if two span intervals overlap.
    """
    return max(a[0], b[0]) < min(a[1], b[1])

# MODIFIED pair_spans for per-gold (recall) and per-pred (precision) matching
def pair_spans(g: List[Span], p: List[Span]) -> Tuple[int, int, int, int]:
    """
    Match spans under per-gold/per-pred policy:
    - recall_tp: number of gold spans overlapped by any predicted span
    - precision_tp: number of predicted spans overlapped by any gold span
    - fp: false positives = predicted spans with no overlap
    - fn: false negatives = gold spans with no overlap
    Returns:
        recall_tp, fp, fn, precision_tp
    """
    # recall: count gold spans covered by any prediction
    recall_tp = sum(1 for gr in g if any(intervals_overlap(gr, pr) for pr in p))
    fn = len(g) - recall_tp
    # precision: count predicted spans that overlap any gold span
    precision_tp = sum(1 for pr in p if any(intervals_overlap(pr, gr) for gr in g))
    fp = len(p) - precision_tp
    return recall_tp, fp, fn, precision_tp


def _parse_ents(raw, doc_id):
    out = []
    for ent in raw.get("labels", []) or raw.get("entities", []) or raw.get("spans", []):
        if isinstance(ent, dict):
            s = ent.get("start_offset") or ent.get("start"); e = ent.get("end_offset") or ent.get("end")
            lbl = ent.get("label") or ent.get("type") or ""
            eid = str(ent.get("id") or ent.get("entity_id") or f"{doc_id}:{s}-{e}")
        else:
            if len(ent) == 3:
                s, e, lbl = ent
            elif len(ent) == 4:
                _, s, e, lbl = ent
            else:
                continue
            eid = f"{doc_id}:{s}-{e}"
        if s is None or e is None:
            continue
        out.append((int(s), int(e), str(lbl).lower(), eid))
    return out


def _parse_rels(raw):
    out = []
    for rel in raw.get("relations", []) or raw.get("links", []):
        if isinstance(rel, dict):
            src = str(rel.get("from_id") or rel.get("src")); tgt = str(rel.get("to_id") or rel.get("dst"))
            pol = str(rel.get("type") or rel.get("label") or rel.get("relation")).lower()
        else:
            if len(rel) < 3:
                continue
            src, tgt, pol = rel[:3]; pol = str(pol).lower()
        out.append((src, tgt, pol))
    return out


def load_jsonl_list(path) -> List[Dict]:
    docs = []
    with pathlib.Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            raw = json.loads(line)
            did = str(raw.get("id") or raw.get("pk") or raw.get("doc_id") or raw.get("document_id") or len(docs))
            docs.append({
                "text": raw.get("text") or raw.get("content") or "",
                "entities": _parse_ents(raw, did),
                "relations": _parse_rels(raw)
            })
    return docs


def _task1(gold, pred):
    tp = fp = fn = tn = 0
    for g, p in zip(gold, pred):
        gc = any(l in {"cause", "effect"} for *_, l, _ in g["entities"])
        pc = any(l in {"cause", "effect"} for *_, l, _ in p["entities"])
        if gc and pc:
            tp += 1
        elif gc and not pc:
            fn += 1
        elif not gc and pc:
            fp += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0
    acc = (tp + tn) / len(gold)
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Precision": prec, "Recall": rec, "F1": f1, "Accuracy": acc, "N": len(gold)}


def _task2(gold, pred, scenario="A"):
    counts = {lbl: {"recall_tp": 0, "precision_tp": 0, "FP": 0, "FN": 0} for lbl in ("cause", "effect")}
    for g, p in zip(gold, pred):
        gb = defaultdict(list); pb = defaultdict(list)
        for s, e, l, _ in g["entities"]:
            if l in {"cause", "effect"}:
                gb[l].append((s, e))
        for s, e, l, _ in p["entities"]:
            if l in {"cause", "effect"}:
                pb[l].append((s, e))
        gc = bool(gb["cause"] or gb["effect"]); pc = bool(pb["cause"] or pb["effect"])
        if scenario.upper() == "B" and not (gc and pc):
            continue
        for l in ("cause", "effect"):
            recall_tp, fp, fn, precision_tp = pair_spans(gb[l], pb[l])
            c = counts[l]
            c["recall_tp"] += recall_tp
            c["precision_tp"] += precision_tp
            c["FP"] += fp
            c["FN"] += fn
    rep = {}
    for l, c in counts.items():
        prec = c["precision_tp"] / (c["precision_tp"] + c["FP"]) if c["precision_tp"] + c["FP"] else 0
        rec = c["recall_tp"] / (c["recall_tp"] + c["FN"]) if c["recall_tp"] + c["FN"] else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        rep[l] = {
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "TP_precision": c["precision_tp"],
            "TP_recall": c["recall_tp"],
            "FP": c["FP"],
            "FN": c["FN"]
        }
    return rep


def _task3(
        gold_docs,
        pred_docs,
        scenario="A",          # ← new
        label_map=None,
        penalise_orphans=False     # keep the flag you prefer
):
    """
    Relation-classification metric.

    scenario="A"  → score every document  
    scenario="B"  → score *only* documents in which
                    both gold **and** prediction contain
                    at least one Cause or Effect entity
                    (same filter used in Task 2).

    Everything else (one-to-one greedy matching, FP/FN rules)
    is unchanged.
    """
    def norm(lbl):
        return label_map.get(lbl, lbl) if label_map else lbl

    cnt = Counter()

    for g_doc, p_doc in zip(gold_docs, pred_docs):

        # --- scenario filter identical to Task 2 -------------
        if scenario.upper() == "B":
            gc = any(l in {"cause", "effect"} for _, _, l, _ in g_doc["entities"])
            pc = any(l in {"cause", "effect"} for _, _, l, _ in p_doc["entities"])
            if not (gc and pc):
                continue     # skip this document entirely

        # ---- entity maps ------------------------------------
        g_ent = {eid: (s, e, lbl) for s, e, lbl, eid in g_doc["entities"]}
        p_ent = {eid: (s, e, lbl) for s, e, lbl, eid in p_doc["entities"]}

        # ---- collect Cause→Effect links ---------------------
        gold_links = [((g_ent[s][0], g_ent[s][1]),
                       (g_ent[t][0], g_ent[t][1]),
                       norm(lbl))
                      for s, t, lbl in g_doc["relations"]
                      if s in g_ent and t in g_ent
                      and g_ent[s][2] == "cause"
                      and g_ent[t][2] == "effect"]

        pred_links = [((p_ent[s][0], p_ent[s][1]),
                       (p_ent[t][0], p_ent[t][1]),
                       norm(lbl))
                      for s, t, lbl in p_doc["relations"]
                      if s in p_ent and t in p_ent
                      and p_ent[s][2] == "cause"
                      and p_ent[t][2] == "effect"]

        used_pred = set()

        # ---- gold-anchored greedy matching ------------------
        for gs_c, gs_e, g_lab in gold_links:
            hit = -1
            for i_p, (ps_c, ps_e, _) in enumerate(pred_links):
                if i_p in used_pred:
                    continue
                if intervals_overlap(gs_c, ps_c) and intervals_overlap(gs_e, ps_e):
                    hit = i_p
                    break

            if hit == -1:
                cnt["FN"] += 1
            else:
                used_pred.add(hit)
                _, _, p_lab = pred_links[hit]
                if p_lab == g_lab:
                    cnt["TP"] += 1
                else:
                    cnt["FP"] += 1

        if penalise_orphans:
            cnt["FP"] += len(pred_links) - len(used_pred)

    # ---- summary -------------------------------------------
    tp, fp, fn = cnt["TP"], cnt["FP"], cnt["FN"]
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc  = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {"TP": tp, "FP": fp, "FN": fn,
            "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1": f1}



# Driver evaluate function unchanged
def evaluate(gold_path, pred_path, scenario='A', label_map=None, penalise_orphans=False):
    gold = load_jsonl_list(gold_path)
    pred = load_jsonl_list(pred_path)

    if len(gold) != len(pred):
        raise ValueError("Length mismatch")
    if gold[0]["text"][:30] != pred[0]["text"][:30]:
        raise ValueError("Order mismatch")

    return {
        "Task1": _task1(gold, pred),
        "Task2": _task2(gold, pred, scenario),
        "Task2_macro": macro_average_task2(_task2(gold, pred, scenario)),
        "Task3": _task3(gold, pred, scenario,
                        label_map=label_map,
                        penalise_orphans=penalise_orphans),
        "Total_Macro": total_macro_average(
            _task1(gold, pred),
            macro_average_task2(_task2(gold, pred, scenario)),
            _task3(gold, pred, scenario,
                   label_map=label_map,
                   penalise_orphans=penalise_orphans)
        )
    }


def macro_average_task2(task2_result):
    """
    Compute macro average for Task 2 (cause/effect).
    """
    keys = ["Precision", "Recall", "F1"]
    macro = {}
    for k in keys:
        macro[k] = (task2_result["cause"][k] + task2_result["effect"][k]) / 2
    macro["TP_precision"] = task2_result["cause"]["TP_precision"] + task2_result["effect"]["TP_precision"]
    macro["TP_recall"] = task2_result["cause"]["TP_recall"] + task2_result["effect"]["TP_recall"]
    macro["FP"] = task2_result["cause"]["FP"] + task2_result["effect"]["FP"]
    macro["FN"] = task2_result["cause"]["FN"] + task2_result["effect"]["FN"]
    return macro


def total_macro_average(task1, task2_macro, task3):
    """
    Compute macro average across Task1, Task2_macro, and Task3.
    """
    keys = ["Precision", "Recall", "F1"]
    macro = {}
    for k in keys:
        macro[k] = (task1[k] + task2_macro[k] + task3[k]) / 3
    return macro


def display_results(results, scenario):
    """
    Display evaluation results in a clear, structured format.
    
    Args:
        results: Results dictionary returned by the evaluate function
        scenario: Scenario identifier to display in the header
        
    Prints:
        Formatted evaluation results with task information and metrics
    """
    border = "=" * 70
    title = f" Scenario {scenario} Results "
    
    print(f"\n{border}")
    print(f"{title:^70}")
    print(f"{border}")
    
    if isinstance(results, dict):
        for metric_type, metrics in results.items():
            print(f"\n【 {metric_type} 】")
            
            if isinstance(metrics, dict):
                # Find the longest metric name for alignment
                max_len = max(len(str(key)) for key in metrics.keys())
                separator = "-" * 60
                
                # Table header
                print(f"{separator}")
                print(f"{'Metric':<{max_len+5}} | {'Value':>15}")
                print(f"{separator}")
                
                # Print each metric with proper alignment
                for key, value in metrics.items():
                    metric_name = f"{str(key):<{max_len+5}}"
                    
                    if isinstance(value, float):
                        # Format float values with 4 decimal places
                        print(f"{metric_name} | {value:>15.4f}")
                    elif isinstance(value, dict):
                        # Handle nested dictionary values - print each nested value on its own line
                        print(f"{metric_name} |")
                        for nested_key, nested_value in value.items():
                            if isinstance(nested_value, float):
                                print(f"  - {nested_key:<{max_len}} | {nested_value:>15.4f}")
                            else:
                                print(f"  - {nested_key:<{max_len}} | {nested_value:>15}")
                    else:
                        print(f"{metric_name} | {value:>15}")
                        
                print(f"{separator}")
            else:
                print(f"  {metrics}")
    else:
        print(f"{results}")
    
    print(f"\n{border}\n")
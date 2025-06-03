# rule_based_inference.py (v5‑full)
"""
Fully self‑contained inference + post‑processing pipeline for the **JointCausalModel**.
Version 5 (5 Jun 2025) — connector merge excludes “and/or”, keeps separate
“lung cancer” & “heart disease”.

This module provides:
- BIO tag post-processing and span merging
- Causal span and relation extraction (auto/head modes)
- Causality decision logic (cls_only/span_only/cls+span)
- Thresholding and filtering for relation extraction

Key functions:
- run_inference: Main entry for inference and post-processing
- _apply_bio_rules: BIO tag clean-up and correction
- _merge_spans: Merge contiguous BIO spans and connectors
- _decide_causal: Decide if a sentence is causal
- _build_rel: (legacy) Build relation pairs
- _model: Model loader (singleton)

Usage:
    results = run_inference(sentences, rel_mode="auto", rel_threshold=0.5, cause_decision="cls+span")
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import torch
from transformers import AutoTokenizer
import torch.nn.functional as F
try:
    from .config import MODEL_CONFIG, id2label_bio, id2label_cls
    from .model import JointCausalModel
except ImportError:
    from config import MODEL_CONFIG, id2label_bio, id2label_cls  # type: ignore
    from model import JointCausalModel  # type: ignore
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_CONFIG["encoder_name"])
# ———————————————————————————————————————————————————————————
@dataclass
class Span:
    """Represents a contiguous span with a causal role (C/E/CE)."""
    role: str
    start_tok: int
    end_tok: int
    text: str
    is_virtual: bool = False
# ———————————————————————————————————————————————————————————
_PUNCT = {".",",",";",":","?","!","(",")","[","]","{","}"}
_STOPWORD_KEEP = {"this","that","these","those","it","they"}
_CONNECTORS = {"of","to","with","for","the"}
_model_cache: JointCausalModel | None = None
# ———————————————————————————————————————————————————————————
def _apply_bio_rules(tok: List[str], lab: List[str]) -> List[str]:
    """
    Apply post-processing rules to BIO tags to fix inconsistencies and clean up spans.
    - Fixes mixed-role spans, punctuation, short tokens, and CE disambiguation.
    """
    rep, n = lab.copy(), len(tok)
    def blocks():
        i=0
        while i<n:
            if rep[i]=="O": i+=1; continue
            s=i
            while i+1<n and rep[i+1]!="O": i+=1
            yield s,i; i+=1
    # B‑1′: Fix mixed-role spans
    for s,e in list(blocks()):
        roles=[rep[j].split("-")[-1] for j in range(s,e+1)]
        if len(set(roles))>1:
            split=next((j for j in range(s+1,e+1) if roles[j-s]!=roles[j-s-1]),None)
            if split:
                if 1 in {split-s,e-split+1}:
                    maj=roles[0] if split-s>e-split+1 else roles[-1]
                    for j in range(s,e+1): rep[j]=f"B-{maj}" if j==s else f"I-{maj}"
    # B‑2: Remove labels from punctuation
    for i,t in enumerate(tok):
        if rep[i]!="O" and t in _PUNCT: rep[i]="O"
    # helper: extract labeled blocks
    def labeled(v):
        i=0; out=[]
        while i<n:
            if v[i]=="O": i+=1; continue
            s=i; role=v[i].split("-")[-1]
            while i+1<n and v[i+1]!="O": i+=1
            out.append((s,i,role)); i+=1
        return out
    bl=labeled(rep)
    # B‑4: Disambiguate CE to C or E if only one present
    if any(r=="CE" for *_,r in bl):
        cntc=sum(1 for *_,r in bl if r=="C"); cnte=sum(1 for *_,r in bl if r=="E")
        if cntc==0 or cnte==0:
            tr="C" if cntc==0 else "E"
            for s,e,r in bl:
                if r=="CE":
                    for idx in range(s,e+1): rep[idx]=f"B-{tr}" if idx==s else f"I-{tr}"
            bl=labeled(rep)
    # B‑5/6: Remove labels from short/stopword tokens and trailing punctuation
    for s,e,_ in bl:
        if tok[e] in _PUNCT: rep[e]="O"
        if e==s and len(tok[s])<=2 and tok[s].lower() not in _STOPWORD_KEEP: rep[s]="O"
    return rep
# ———————————————————————————————————————————————————————————
def _merge_spans(tok: List[str], lab: List[str]) -> List[Span]:
    """
    Merge contiguous labeled tokens into Span objects, gluing across connectors.
    """
    spans=[]; i=0
    while i<len(tok):
        if lab[i]=="O": i+=1; continue
        role=lab[i].split("-")[-1]; s=i
        while i+1<len(tok) and lab[i+1]!="O": i+=1
        spans.append(Span(role,s,i,TOKENIZER.convert_tokens_to_string(tok[s:i+1])))
        i+=1
    # connector glue: merge across connector tokens
    merged=[spans[0]] if spans else []
    for sp in spans[1:]:
        prv=merged[-1]
        if sp.role==prv.role and sp.start_tok==prv.end_tok+2 and tok[prv.end_tok+1].lower() in _CONNECTORS:
            merged[-1]=Span(prv.role,prv.start_tok,sp.end_tok,TOKENIZER.convert_tokens_to_string(tok[prv.start_tok:sp.end_tok+1]),prv.is_virtual)
        else: merged.append(sp)
    return merged
# ———————————————————————————————————————————————————————————
def _decide_causal(cls,sp,mode):
    """
    Decide if a sentence is causal based on the selected mode:
    - 'cls_only': Only the classification head
    - 'span_only': Only the presence of C/E spans
    - 'cls+span': Both classification and span presence
    """
    span_ok=any(x.role=="C" for x in sp) and any(x.role=="E" for x in sp)
    cls_c=cls.argmax(-1).item()==1
    if mode=="cls_only": return cls_c
    if mode=="span_only": return span_ok
    if mode=="cls+span": return cls_c and span_ok
    raise ValueError(mode)
# ———————————————————————————————————————————————————————————
def _build_rel(sp: List[Span], rel, mode, thr):
    """
    Legacy relation builder (not used in main pipeline).
    For 'auto' mode: pairs single C with all E, or single E with all C.
    For 'head' mode: uses rel logits and threshold.
    """
    c,e=[],[]
    for s in sp:
        if s.role=="C": c.append(s)
        elif s.role=="E": e.append(s)
        else:
            c.append(Span("C",s.start_tok,s.end_tok,s.text,True)); e.append(Span("E",s.start_tok,s.end_tok,s.text,True))
    pairs=[]
    if mode=="head":
        if rel is None: raise ValueError("rel_logits required")
        pair_idx = 0
        for cs in c:
            for es in e:
                if (cs.start_tok,cs.end_tok)==(es.start_tok,es.end_tok): continue
                # rel shape: [num_pairs, num_rel_labels]
                if rel[pair_idx,1].item()>=thr: pairs.append((cs,es))
                pair_idx += 1
    elif mode=="auto":
        if len(c)==1 and e: pairs=[(c[0],x) for x in e]
        elif len(e)==1 and c: pairs=[(x,e[0]) for x in c]
    out=[]
    for cs,es in pairs:
        if any(es==oe and cs.text in oc.text for oc,oe in out): continue
        out.append((cs,es))
    return out
# ———————————————————————————————————————————————————————————
def _model():
    """
    Load and cache the JointCausalModel singleton.
    """
    global _model_cache
    if _model_cache is None:
        m=JointCausalModel(**MODEL_CONFIG)
        m.load_state_dict(torch.load("src\\jointlearning\\expert_bert_GCE_Softmax_Normal\\expert_bert_GCE_Softmax_Normal_model.pt",map_location=DEVICE))
        m.to(DEVICE).eval(); _model_cache=m
    return _model_cache
# ———————————————————————————————————————————————————————————
def run_inference(sents: List[str], *, rel_mode="auto", rel_threshold=0.4, cause_decision="cls+span"):
    """
    Main inference and post-processing pipeline for JointCausalModel.

    Args:
        sents (List[str]):
            List of input sentences to process.
        rel_mode (str, optional):
            Relation extraction mode. Options:
                'auto' (default):
                    - If exactly one C and >=1 E: pair that C with every E.
                    - If exactly one E and >=1 C: pair that E with every C.
                    - If multiple C and multiple E: use the model's relation head (if available) to score all possible C/E pairs and only keep those above rel_threshold. If the relation head is not available, no relations are returned.
                    - If no valid C or E spans: no relations are returned.
                'head':
                    - Always use the model's relation head to score all possible C/E pairs, regardless of the number of C/E spans. Only pairs with probability above rel_threshold are returned.
        rel_threshold (float, optional):
            Probability threshold for relation extraction. Only pairs with a predicted probability for the 'Rel_CE' class strictly greater than this value are returned as relations. Default is 0.4.
        cause_decision (str, optional):
            Causality decision mode. Options:
                'cls_only': Only use the model's classification head to decide if the sentence is causal.
                'span_only': Only require the presence of at least one C and one E span for a sentence to be considered causal.
                'cls+span' (default): Require both the classification head to predict causal and at least one C and one E span to be present.

    Returns:
        List[Dict]:
            A list of dictionaries, one per input sentence, each with the following keys:
                'text': The original sentence.
                'causal': Boolean indicating if the sentence is considered causal (according to the selected cause_decision mode and relation extraction result).
                'relations': List of extracted relations, each a dict with keys 'cause', 'effect', and 'type'.

    Example usage:
        >>> sentences = [
        ...     "Insomnia causes depression and a lack of concentration in children",
        ...     "Exercise improves physical health and mental well-being"
        ... ]
        >>> results = run_inference(sentences, rel_mode="auto", rel_threshold=0.5, cause_decision="cls+span")
        >>> print(json.dumps(results, indent=2, ensure_ascii=False))

    Configuration details:
        - The function will always perform BIO tag clean-up and span merging.
        - In 'auto' mode, relation extraction is rule-based for simple cases and model-based for complex cases (multiple C/E spans).
        - In 'head' mode, relation extraction is always model-based.
        - If no relations are found for a sentence, 'causal' will be set to False regardless of the cause_decision mode.
        - The function is robust to sentences with no valid spans or ambiguous cases.

    """
    mdl=_model(); outs=[]
    for txt in sents:
        enc=TOKENIZER([txt],return_tensors="pt",truncation=True,max_length=512)
        enc={k:v.to(DEVICE) for k,v in enc.items()};
        with torch.no_grad():
            rel_args = {}
            rel_pair_spans = []
            # Always prepare relation extraction arguments if needed (for head mode or auto mode with multi C/E)
            need_rel_head = rel_mode == "head"
            # For auto mode, we need to check after span extraction if multi C/E exists
            res_tmp = None
            spans_tmp = None
            c_spans = None
            e_spans = None
            if rel_mode == "head":
                res_tmp = mdl(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
                bio_tmp = res_tmp["bio_emissions"].squeeze(0).argmax(-1).tolist()
                tok_tmp = TOKENIZER.convert_ids_to_tokens(enc["input_ids"].squeeze(0))
                lab_tmp = [id2label_bio[i] for i in bio_tmp]
                fixed_tmp = _apply_bio_rules(tok_tmp, lab_tmp)
                spans_tmp = _merge_spans(tok_tmp, fixed_tmp)
                c_spans = [s for s in spans_tmp if s.role in ("C", "CE")]
                e_spans = [s for s in spans_tmp if s.role in ("E", "CE")]
                pair_batch = []
                cause_starts = []
                cause_ends = []
                effect_starts = []
                effect_ends = []
                for c in c_spans:
                    for e in e_spans:
                        if c.start_tok == e.start_tok and c.end_tok == e.end_tok:
                            continue
                        pair_batch.append(0)
                        cause_starts.append(c.start_tok)
                        cause_ends.append(c.end_tok)
                        effect_starts.append(e.start_tok)
                        effect_ends.append(e.end_tok)
                        rel_pair_spans.append((c, e))
                if pair_batch:
                    rel_args = {
                        "pair_batch": torch.tensor(pair_batch, device=DEVICE),
                        "cause_starts": torch.tensor(cause_starts, device=DEVICE),
                        "cause_ends": torch.tensor(cause_ends, device=DEVICE),
                        "effect_starts": torch.tensor(effect_starts, device=DEVICE),
                        "effect_ends": torch.tensor(effect_ends, device=DEVICE),
                    }
            res=mdl(input_ids=enc["input_ids"],attention_mask=enc["attention_mask"],**rel_args)
        cls=res["cls_logits"].squeeze(0)
        bio=res["bio_emissions"].squeeze(0).argmax(-1).tolist()
        tok=TOKENIZER.convert_ids_to_tokens(enc["input_ids"].squeeze(0))
        lab=[id2label_bio[i] for i in bio]
        fixed=_apply_bio_rules(tok,lab)
        spans=_merge_spans(tok,fixed)
        # Decide causality based on selected mode
        causal=_decide_causal(cls,spans,cause_decision)
        if not causal:
            outs.append({"text":txt,"causal":False,"relations":[]}); continue
        rels = []
        rel_logits = res.get("rel_logits")
        rel_probs = None
        if rel_logits is not None:
            rel_probs = torch.softmax(rel_logits, dim=-1)
        if rel_mode == "head":
            # Always use third head for all C/E pairs
            for idx, (csp, esp) in enumerate(rel_pair_spans):
                if rel_probs[idx, 1].item() > rel_threshold:
                    rels.append({"cause": csp.text, "effect": esp.text, "type": "Rel_CE"})
        elif rel_mode == "auto":
            c_spans = [s for s in spans if s.role in ("C", "CE")]
            e_spans = [s for s in spans if s.role in ("E", "CE")]
            # No valid spans: do not output any relations
            if not c_spans or not e_spans:
                rels = []
            elif len(c_spans) == 1 and len(e_spans) >= 1:
                for e in e_spans:
                    rels.append({"cause": c_spans[0].text, "effect": e.text, "type": "Rel_CE"})
            elif len(e_spans) == 1 and len(c_spans) >= 1:
                for c in c_spans:
                    rels.append({"cause": c.text, "effect": e_spans[0].text, "type": "Rel_CE"})
            elif len(c_spans) > 1 and len(e_spans) > 1:
                # For multi C/E, run the relation head just like in head mode
                # Prepare rel_args and rel_pair_spans if not already done
                pair_batch = []
                cause_starts = []
                cause_ends = []
                effect_starts = []
                effect_ends = []
                rel_pair_spans = []
                for c in c_spans:
                    for e in e_spans:
                        if (c.start_tok == e.start_tok and e.end_tok == c.end_tok):
                            continue
                        pair_batch.append(0)
                        cause_starts.append(c.start_tok)
                        cause_ends.append(c.end_tok)
                        effect_starts.append(e.start_tok)
                        effect_ends.append(e.end_tok)
                        rel_pair_spans.append((c, e))
                if pair_batch:
                    rel_args = {
                        "pair_batch": torch.tensor(pair_batch, device=DEVICE),
                        "cause_starts": torch.tensor(cause_starts, device=DEVICE),
                        "cause_ends": torch.tensor(cause_ends, device=DEVICE),
                        "effect_starts": torch.tensor(effect_starts, device=DEVICE),
                        "effect_ends": torch.tensor(effect_ends, device=DEVICE),
                    }
                    # Run the model again to get rel_logits for these pairs
                    res_rel = mdl(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], **rel_args)
                    rel_logits = res_rel.get("rel_logits")
                    if rel_logits is not None:
                        rel_probs = torch.softmax(rel_logits, dim=-1)
                        for idx, (csp, esp) in enumerate(rel_pair_spans):
                            if rel_probs[idx, 1].item() > rel_threshold:
                                rels.append({"cause": csp.text, "effect": esp.text, "type": "Rel_CE"})
        # If no relations found, set causal to False
        if cause_decision == "cls_only":
            causal = cls.argmax(-1).item() == 1
        elif cause_decision == "span_only":
            causal = any(x.role == "C" for x in spans) and any(x.role == "E" for x in spans)
        elif cause_decision == "cls+span":
            causal = (cls.argmax(-1).item() == 1) and (any(x.role == "C" for x in spans) and any(x.role == "E" for x in spans))
        else:
            raise ValueError(cause_decision)
        if not rels:
            outs.append({"text": txt, "causal": False, "relations": []})
        else:
            outs.append({"text": txt, "causal": causal, "relations": rels})
    return outs



if __name__ == "__main__":
    test = ["promoting ri might reduce risk factors for drug use and enhance the effects of protective factors (brook et al., 1998).;;",
        "it is also considered that the process could easily be adapted for virtual delivery, thus increasing its accessibility.;;",
        "(corrected for unreliability; Bryk and Raudenbush 1992).;;",
        "big data technologies, however, facilitate the collection and sharing of these data on a large scale.;;",
        "depending on how successful and consistent the analyst was while inventing terms and coupling them with contexts, the resulting network would also become intuitively meaningful to any native speaker (see below).;;",
        "thus, this would intensify interpersonal stress with their family members and increase the risk of relapse.;;",
        "for instance, schneider and turkat (1975) reported that low self-esteem individuals were more dependent on the evaluations of others in determining their sense of self-worth.;;",
        "in many programme areas, this is in fact possible since there are frequently follow-on programmes whose planning stages could deploy a review approach, but of course it is rarely done.;;",
        "Insomnia causes depression and a lack of concentration in children",
        "smoking causes lung cancer and heart disease",
        "exercise improves physical health and mental well-being",
        "Permitting continuous rather than binary ''all-or-nothing'' contributions significantly increases contributions and facilitates provision.",
        "according to smith (2008), there is a need for substantial and consistent support and encouragement for women to participate in studies.",
        "thus, experiencing a task that leads to decrements in feelings of relatedness may affect people's ability to experience intrinsic motivation by also influencing people's mood.;;",
        "It is instructive to contrast this estimate to the one in the previous section, based on the very simple, two-parameter (g, q) model.;;",
        "she also recognized that structurally disadvantaged communities supply the black and brown bodies that fill chicago's jail and illinois's prisons (lavigne et al.;;",
        "the effect of constrained communication and limited information 623 communication, the content of the communication may also reveal which kind of information is more important to the participants.;;",
        "the subjects who were dependent on the other for future aid increased their level of help giving across the trials.;;",
        "instead, depleted participants were more modest in their predictions and more accurate in their predictions than nondepleted participants.;;",
        "the perceived consequences of turning to others for social support, therefore, may influence the expression of pain.;;",
        "moreover, in the context of cooperation in organizational and legal settings, de cremer and tyler (2007) showed that if a party communicates intentions to listen to others and take their interests at heart, cooperative decision making is only promoted if this other is seen as honest and trustworthy.;;",
        "A significant rise in local unemployment rates is a primary driver of increased property crime in the metropolitan area.",
        "Consistent and responsive caregiving in the first year of life is a crucial factor in the development of a secure attachment style in children.",
        "The prolonged drought led to widespread crop failure, which in turn caused a sharp increase in food prices, ultimately contributing to social unrest in the region."
        ]
    res = run_inference(test, rel_mode="auto", rel_threshold=0.5, cause_decision="cls+span")
    print(json.dumps(res, indent=2, ensure_ascii=False))

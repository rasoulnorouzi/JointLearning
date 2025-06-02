# rule_based_inference.py (v5‑full)
"""
Fully self‑contained inference + post‑processing pipeline for the **JointCausalModel**.
Version 5 (5 Jun 2025) — connector merge excludes “and/or”, keeps separate
“lung cancer” & “heart disease”.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import torch
from transformers import AutoTokenizer
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
    role: str; start_tok: int; end_tok: int; text: str; is_virtual: bool = False
# ———————————————————————————————————————————————————————————
_PUNCT = {".",",",";",":","?","!","(",")","[","]","{","}"}
_STOPWORD_KEEP = {"this","that","these","those","it","they"}
_CONNECTORS = {"of","to","with","for","the"}
_model_cache: JointCausalModel | None = None
# ———————————————————————————————————————————————————————————
def _apply_bio_rules(tok: List[str], lab: List[str]) -> List[str]:
    rep, n = lab.copy(), len(tok)
    def blocks():
        i=0
        while i<n:
            if rep[i]=="O": i+=1; continue
            s=i
            while i+1<n and rep[i+1]!="O": i+=1
            yield s,i; i+=1
    # B‑1′
    for s,e in list(blocks()):
        roles=[rep[j].split("-")[-1] for j in range(s,e+1)]
        if len(set(roles))>1:
            split=next((j for j in range(s+1,e+1) if roles[j-s]!=roles[j-s-1]),None)
            if split:
                if 1 in {split-s,e-split+1}:
                    maj=roles[0] if split-s>e-split+1 else roles[-1]
                    for j in range(s,e+1): rep[j]=f"B-{maj}" if j==s else f"I-{maj}"
    # B‑2
    for i,t in enumerate(tok):
        if rep[i]!="O" and t in _PUNCT: rep[i]="O"
    # helper
    def labeled(v):
        i=0; out=[]
        while i<n:
            if v[i]=="O": i+=1; continue
            s=i; role=v[i].split("-")[-1]
            while i+1<n and v[i+1]!="O": i+=1
            out.append((s,i,role)); i+=1
        return out
    bl=labeled(rep)
    # B‑4
    if any(r=="CE" for *_,r in bl):
        cntc=sum(1 for *_,r in bl if r=="C"); cnte=sum(1 for *_,r in bl if r=="E")
        if cntc==0 or cnte==0:
            tr="C" if cntc==0 else "E"
            for s,e,r in bl:
                if r=="CE":
                    for idx in range(s,e+1): rep[idx]=f"B-{tr}" if idx==s else f"I-{tr}"
            bl=labeled(rep)
    # B‑5/6
    for s,e,_ in bl:
        if tok[e] in _PUNCT: rep[e]="O"
        if e==s and len(tok[s])<=2 and tok[s].lower() not in _STOPWORD_KEEP: rep[s]="O"
    return rep
# ———————————————————————————————————————————————————————————
def _merge_spans(tok: List[str], lab: List[str]) -> List[Span]:
    spans=[]; i=0
    while i<len(tok):
        if lab[i]=="O": i+=1; continue
        role=lab[i].split("-")[-1]; s=i
        while i+1<len(tok) and lab[i+1]!="O": i+=1
        spans.append(Span(role,s,i,TOKENIZER.convert_tokens_to_string(tok[s:i+1])))
        i+=1
    # connector glue
    merged=[spans[0]] if spans else []
    for sp in spans[1:]:
        prv=merged[-1]
        if sp.role==prv.role and sp.start_tok==prv.end_tok+2 and tok[prv.end_tok+1].lower() in _CONNECTORS:
            merged[-1]=Span(prv.role,prv.start_tok,sp.end_tok,TOKENIZER.convert_tokens_to_string(tok[prv.start_tok:sp.end_tok+1]),prv.is_virtual)
        else: merged.append(sp)
    return merged
# ———————————————————————————————————————————————————————————
def _decide_causal(cls,sp,mode):
    span_ok=any(x.role=="C" for x in sp) and any(x.role=="E" for x in sp)
    cls_c=cls.argmax(-1).item()==1
    if mode=="cls_only": return cls_c
    if mode=="span_only": return span_ok
    if mode=="cls+span": return cls_c and span_ok
    raise ValueError(mode)
# ———————————————————————————————————————————————————————————
def _build_rel(sp: List[Span], rel, mode, thr):
    c,e=[],[]
    for s in sp:
        if s.role=="C": c.append(s)
        elif s.role=="E": e.append(s)
        else:
            c.append(Span("C",s.start_tok,s.end_tok,s.text,True)); e.append(Span("E",s.start_tok,s.end_tok,s.text,True))
    pairs=[]
    if mode=="head":
        if rel is None: raise ValueError("rel_logits required")
        for cs in c:
            for es in e:
                if (cs.start_tok,cs.end_tok)==(es.start_tok,es.end_tok): continue
                if rel[cs.end_tok,es.end_tok,1].item()>=thr: pairs.append((cs,es))
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
    global _model_cache
    if _model_cache is None:
        m=JointCausalModel(**MODEL_CONFIG)
        m.load_state_dict(torch.load("src/jointlearning/expert_bert_softmax/expert_bert_softmax_model.pt",map_location=DEVICE))
        m.to(DEVICE).eval(); _model_cache=m
    return _model_cache
# ———————————————————————————————————————————————————————————
def run_inference(sents: List[str],*,level=3,rel_mode="auto",cause_decision="cls+span",rel_threshold=0.5):
    mdl=_model(); outs=[]
    for txt in sents:
        enc=TOKENIZER([txt],return_tensors="pt",truncation=True,max_length=512)
        enc={k:v.to(DEVICE) for k,v in enc.items()};
        with torch.no_grad(): res=mdl(input_ids=enc["input_ids"],attention_mask=enc["attention_mask"])
        cls=res["cls_logits"].squeeze(0)
        bio=res["bio_emissions"].squeeze(0).argmax(-1).tolist()
        tok=TOKENIZER.convert_ids_to_tokens(enc["input_ids"].squeeze(0))
        lab=[id2label_bio[i] for i in bio]
        fixed=_apply_bio_rules(tok,lab) if level>=1 else lab
        spans=_merge_spans(tok,fixed)
        causal=_decide_causal(cls,spans,cause_decision) if level>=2 else True
        if not causal:
            outs.append({"text":txt,"causal":False,"relations":[]}); continue
        rels=[]
        if level>=3:
            for csp,esp in _build_rel(spans,res.get("rel_logits"),rel_mode,rel_threshold):
                rels.append({"cause":csp.text,"effect":esp.text,"type":"Rel_CE"})
        outs.append({"text":txt,"causal":True,"relations":rels})
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
        "moreover, in the context of cooperation in organizational and legal settings, de cremer and tyler (2007) showed that if a party communicates intentions to listen to others and take their interests at heart, cooperative decision making is only promoted if this other is seen as honest and trustworthy.;;"
        ]
    res = run_inference(test)
    print(json.dumps(res, indent=2, ensure_ascii=False))

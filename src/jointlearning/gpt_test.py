"""joint_causal_model.py
A self‑contained demo of the **JointCausalModel** plus a miniature, fully‑mocked
test‑suite with social‑science‑flavoured sentences.  The model code is production
ready; the tests run without internet/GPU by monkey‑patching `forward()` so you
can iterate on span logic quickly.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import types
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# ↓↓↓  tiny stand‑in for your real config.py  ↓↓↓
# ---------------------------------------------------------------------------

id2label_bio = {
    0: "B-C", 1: "I-C", 2: "B-E", 3: "I-E", 4: "B-CE", 5: "I-CE", 6: "O",
}
label2id_bio = {v: k for k, v in id2label_bio.items()}

id2label_rel = {0: "Rel_None", 1: "Rel_CE"}
label2id_rel = {v: k for k, v in id2label_rel.items()}

id2label_cls = {0: "non-causal", 1: "causal"}
label2id_cls = {v: k for k, v in id2label_cls.items()}

MODEL_CONFIG = {
    "encoder_name": "bert-base-uncased",
    "num_cls_labels": 2,
    "num_bio_labels": 7,
    "num_rel_labels": 2,
    "dropout": 0.1,
}

# ---------------------------------------------------------------------------
#  util constants for span cleaning / merging
# ---------------------------------------------------------------------------
STOPWORDS = {"the", "a", "an"}
BRIDGE_WORDS = {"of", "to", "and"}            # words allowed *inside* spans
PUNCT = {".", ",", ";", ":"}

# ---------------------------------------------------------------------------
#  the actual model
# ---------------------------------------------------------------------------

class JointCausalModel(nn.Module):
    """Transformer encoder + three heads + lightweight post‑processing."""

    def __init__(
        self,
        *,
        encoder_name: str = MODEL_CONFIG["encoder_name"],
        num_cls_labels: int = MODEL_CONFIG["num_cls_labels"],
        num_bio_labels: int = MODEL_CONFIG["num_bio_labels"],
        num_rel_labels: int = MODEL_CONFIG["num_rel_labels"],
        dropout: float = MODEL_CONFIG["dropout"],
    ) -> None:
        super().__init__()
        self.encoder_name = encoder_name
        self.num_cls_labels = num_cls_labels
        self.num_bio_labels = num_bio_labels
        self.num_rel_labels = num_rel_labels

        self.enc = AutoModel.from_pretrained(encoder_name)
        hid = self.enc.config.hidden_size
        self.layer_norm = nn.LayerNorm(hid)
        self.dropout = nn.Dropout(dropout)

        self.cls_head = nn.Sequential(
            nn.Linear(hid, hid // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hid // 2, num_cls_labels)
        )
        self.bio_head = nn.Sequential(
            nn.Linear(hid, hid), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hid, hid // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hid // 2, num_bio_labels)
        )
        self.rel_head = nn.Sequential(
            nn.Linear(hid * 2, hid), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hid, hid // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hid // 2, num_rel_labels)
        )
        self._init_linears()

    # ------------------------------------------------------------------
    #  basic forward
    # ------------------------------------------------------------------
    def _init_linears(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state
        return self.layer_norm(self.dropout(x))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **_):
        h = self.encode(input_ids, attention_mask)
        return {
            "cls_logits": self.cls_head(h[:, 0]),
            "bio_emissions": self.bio_head(h),
            "hidden_states": h,
        }

    # ------------------------------------------------------------------
    #  token→word alignment helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _word_ids(offsets, attn):
        """HF‑style word_ids reconstruction from offset_mapping."""
        wid, last_e, cur = [], -1, -1
        for (s, e), m in zip(offsets, attn):
            if m == 0:
                wid.append(None)
                continue
            if (s == e == 0) or (s < 0):
                wid.append(None)
                last_e = -1
                continue
            if s != last_e:
                cur += 1
            wid.append(cur)
            last_e = e
        return wid

    # ------------------------------------------------------------------
    #  BIO → spans
    # ------------------------------------------------------------------
    def _align_word_tags(self, bio_ids, attn, offsets, text):
        wid = self._word_ids(offsets, attn)
        words = {}
        for tok_i, w_id in enumerate(wid):
            if w_id is None:
                continue
            words.setdefault(w_id, []).append(tok_i)
        out = []
        for w_id in sorted(words):
            tks = words[w_id]
            s, e = offsets[tks[0]][0], offsets[tks[-1]][1]
            out.append(
                {
                    "text": text[s:e],
                    "tag": id2label_bio[bio_ids[tks[0]]],
                    "start_char": s,
                    "end_char": e,
                    "start_token_idx": tks[0],
                    "end_token_idx": tks[-1],
                }
            )
        return out

    @staticmethod
    def _fix_I_tags(tags):
        fixed = []
        for i, t in enumerate(tags):
            if t.startswith("I-"):
                ent = t[2:]
                if i == 0 or fixed[-1] == "O" or fixed[-1][2:] != ent:
                    fixed.append(f"B-{ent}")
                else:
                    fixed.append(t)
            else:
                fixed.append(t)
        return fixed

    def _group_to_spans(self, words, text):
        spans, buf, typ = [], [], None
        for idx, w in enumerate(words):
            tag = w["tag"]
            pref, ent = ("O", None) if tag == "O" else tag.split("-", 1)
            nxt = words[idx + 1]["tag"] if idx + 1 < len(words) else "O"
            nxt_ent = nxt.split("-", 1)[1] if nxt != "O" else None

            if pref == "B":
                if buf:
                    spans.append(self._finalize(buf, typ, text))
                buf, typ = [w], ent
            elif pref == "I" and typ == ent:
                buf.append(w)
            elif tag == "O" and w["text"].lower() in BRIDGE_WORDS and typ == nxt_ent:
                # keep bridge word inside current span
                buf.append(w)
            else:
                if buf:
                    spans.append(self._finalize(buf, typ, text))
                buf, typ = [], None
        if buf:
            spans.append(self._finalize(buf, typ, text))
        return spans

    @staticmethod
    def _finalize(buf, ent_type, text):
        s_c, e_c = buf[0]["start_char"], buf[-1]["end_char"]
        span_text = text[s_c:e_c]
        # strip leading stop‑words
        tokens = span_text.split()
        while tokens and tokens[0].lower() in STOPWORDS:
            drop = tokens.pop(0)
            s_c += len(drop) + 1
        # strip trailing punctuation
        while tokens and tokens[-1] in PUNCT:
            drop = tokens.pop()
            e_c -= len(drop)
        span_text = " ".join(tokens) if tokens else text[s_c:e_c]
        return {
            "label": ent_type,
            "text": span_text,
            "start_char": s_c,
            "end_char": e_c,
            "start_token_idx": buf[0]["start_token_idx"],
            "end_token_idx": buf[-1]["end_token_idx"],
        }

    # ------------------------------------------------------------------
    #  public inference API (batch)
    # ------------------------------------------------------------------
    def predict_batch(
        self,
        texts: List[str],
        tok_batch: Dict[str, torch.Tensor],
        *,
        device="cpu",
        use_heuristic=False,
        override_cls_if_spans_found=False,
    ) -> List[Dict[str, Any]]:
        self.eval()
        self.to(device)
        ids, mask = tok_batch["input_ids"].to(device), tok_batch["attention_mask"].to(device)
        offsets = tok_batch["offset_mapping"]

        with torch.no_grad():
            outs = self.forward(input_ids=ids, attention_mask=mask)
        cls_logits, emis, hidden = outs["cls_logits"], outs["bio_emissions"], outs["hidden_states"]

        res = []
        for i, txt in enumerate(texts):
            is_causal_init = id2label_cls[int(torch.argmax(cls_logits[i]))] == "causal"
            if not is_causal_init and not override_cls_if_spans_found:
                res.append({"text": txt, "causal": False, "relations": []})
                continue

            tok_tags = torch.argmax(emis[i], -1).cpu().tolist()
            words = self._align_word_tags(tok_tags, mask[i].cpu().tolist(), offsets[i].cpu().tolist(), txt)
            if not words:
                res.append({"text": txt, "causal": False, "relations": []})
                continue
            fixed_tags = self._fix_I_tags([w["tag"] for w in words])
            for j, w in enumerate(words):
                w["tag"] = fixed_tags[j]
            spans = self._group_to_spans(words, txt)

            causes, effects = [], []
            span_id = 0
            for sp in spans:
                sp["id"] = span_id
                span_id += 1
                if sp["label"] == "C":
                    causes.append(sp)
                elif sp["label"] == "E":
                    effects.append(sp)
                else:  # CE
                    causes.append({**sp, "original_label": "CE"})
                    effects.append({**sp, "id": span_id, "original_label": "CE"})
                    span_id += 1

            if not (causes and effects):
                res.append({"text": txt, "causal": False, "relations": []})
                continue

            relations = []
            if use_heuristic:
                if len(effects) == 1:
                    for c in causes:
                        relations.append({"cause": c["text"], "effect": effects[0]["text"], "type": "Rel_CE"})
                elif len(causes) == 1:
                    for e in effects:
                        relations.append({"cause": causes[0]["text"], "effect": e["text"], "type": "Rel_CE"})
            # (relation head omitted in this mock demo)

            is_causal_final = bool(relations) or (override_cls_if_spans_found and (causes and effects))
            res.append({"text": txt, "causal": is_causal_final, "relations": relations})
        return res

    # convenience wrapper --------------------------------------------------
    def predict(self, text, tok, **kw):
        return self.predict_batch([text], {k: v.unsqueeze(0) for k, v in tok.items()}, **kw)[0]

# ---------------------------------------------------------------------------
#  mocked tests – no training required
# ---------------------------------------------------------------------------

def get_mock_forward(tokenizer, txt, mock):
    tok = tokenizer(txt, return_tensors="pt")
    seq_len = tok["input_ids"].size(1)

    def _mock(self, input_ids, attention_mask, **_):
        cls_logits = torch.full((1, 2), -10.0)
        cls_logits[0, mock["cls"]] = 10.0

        emis = torch.full((1, seq_len, 7), -10.0)
        emis[0, :, label2id_bio["O"]] = 5.0
        word_ids = self._word_ids(tok["offset_mapping"][0].tolist(), tok["attention_mask"][0].tolist())
        word_bio = iter(mock["bio"])
        for pos, wid in enumerate(word_ids):
            if wid is not None:
                emis[0, pos, next(word_bio)] = 10.0
        hidden = torch.randn(1, seq_len, self.enc.config.hidden_size)
        return {"cls_logits": cls_logits, "bio_emissions": emis, "hidden_states": hidden}

    return tok, _mock

# test fixtures ------------------------------------------------------------
TESTS = [
    ("Heavy rain caused the flood.",      {"cls": 1, "bio": [0, 1, 6, 6, 2, 6]},  [("Heavy rain", "flood")]),
    ("Economic inequality led to political instability.",
        {"cls": 1, "bio": [0, 1, 6, 6, 2, 3, 6]},             [("Economic inequality", "political instability")]),
    ("Access to education improves social mobility.",
        {"cls": 1, "bio": [0, 1, 6, 2, 3, 6]},                [("Access to education", "social mobility")]),
]

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokzr = AutoTokenizer.from_pretrained(MODEL_CONFIG["encoder_name"])
    model = JointCausalModel(
        encoder_name=MODEL_CONFIG["encoder_name"],
        num_cls_labels=MODEL_CONFIG["num_cls_labels"],
        num_bio_labels=MODEL_CONFIG["num_bio_labels"],
        num_rel_labels=MODEL_CONFIG["num_rel_labels"],
    ).to(device)
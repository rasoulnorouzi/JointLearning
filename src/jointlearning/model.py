from __future__ import annotations
from typing import Dict, Tuple, Optional, Any, List
import torch
import torch.nn as nn
from transformers import AutoModel
from huggingface_hub import PyTorchModelHubMixin
from dataclasses import dataclass

try:
    from .config import MODEL_CONFIG, id2label_bio, id2label_rel, id2label_cls
except ImportError:
    from config import MODEL_CONFIG, id2label_bio, id2label_rel, id2label_cls

# ---------------------------------------------------------------------------
# Type aliases & label maps
# ---------------------------------------------------------------------------
label2id_bio = {v: k for k, v in id2label_bio.items()}
label2id_rel = {v: k for k, v in id2label_rel.items()}
label2id_cls = {v: k for k, v in id2label_cls.items()}

# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------
"""Joint Causal Extraction Model (softmax)
============================================================================

A PyTorch module for joint causal extraction using softmax decoding for BIO tagging.
The model supports class weights for handling imbalanced data.

```python
>>> model = JointCausalModel()        # softmax-based model
```

---------------------------------------------------------------------------
Usage overview
---------------------------------------------------------------------------

**Training**
~~~~~~~~~~~~
(Training code example omitted for brevity, see previous versions)
---------------------------------------------------------------------------
Implementation
---------------------------------------------------------------------------
"""


# ---------------------------------------------------------------------------
# Span dataclass
# ---------------------------------------------------------------------------
@dataclass
class Span:
    role: str
    start_tok: int
    end_tok: int
    text: str
    is_virtual: bool = False


# ---------------------------------------------------------------------------
# Main module
# ---------------------------------------------------------------------------
class JointCausalModel(nn.Module, PyTorchModelHubMixin):
    """Encoder + three heads with **optional CRF** BIO decoder.

    This model integrates a pre-trained transformer encoder with three distinct
    heads for:
    1. Classification (cls_head): Predicts a global label for the input.
    2. BIO tagging (bio_head): Performs sequence tagging using BIO scheme.
       Can operate with a CRF layer or standard softmax.
    3. Relation extraction (rel_head): Identifies relations between entities
       detected by the BIO tagging head.
    """

    # ------------------------------------------------------------------
    # constructor
    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        encoder_name: str = MODEL_CONFIG["encoder_name"],
        num_cls_labels: int = MODEL_CONFIG["num_cls_labels"],
        num_bio_labels: int = MODEL_CONFIG["num_bio_labels"],
        num_rel_labels: int = MODEL_CONFIG["num_rel_labels"],
        dropout: float = MODEL_CONFIG["dropout"],
    ) -> None:
        """Initializes the JointCausalModel.

        Args:
            encoder_name: Name of the pre-trained transformer model to use
                (e.g., "bert-base-uncased").
            num_cls_labels: Number of labels for the classification task.
            num_bio_labels: Number of labels for the BIO tagging task.
            num_rel_labels: Number of labels for the relation extraction task.
            dropout: Dropout rate for regularization.
        """
        super().__init__()

        self.encoder_name = encoder_name
        self.num_cls_labels = num_cls_labels
        self.num_bio_labels = num_bio_labels
        self.num_rel_labels = num_rel_labels
        self.dropout_rate = dropout
        self.enc = AutoModel.from_pretrained(encoder_name)
        self.hidden_size = self.enc.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.cls_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, num_cls_labels),
        )
        self.bio_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, num_bio_labels),
        )
        self.rel_head = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, num_rel_labels),
        )
        self._init_new_layer_weights()

    def get_config_dict(self) -> Dict:
        """Returns the model's configuration as a dictionary."""
        return {
            "encoder_name": self.encoder_name,
            "num_cls_labels": self.num_cls_labels,
            "num_bio_labels": self.num_bio_labels,
            "num_rel_labels": self.num_rel_labels,
            "dropout": self.dropout_rate,
        }

    @classmethod
    def from_config_dict(cls, config: Dict) -> "JointCausalModel":
        """Creates a JointCausalModel instance from a configuration dictionary."""
        return cls(**config)

    def _init_new_layer_weights(self):
        """Initializes the weights of the newly added linear layers.

        Uses Xavier uniform initialization for weights and zeros for biases.
        """
        for mod in [self.cls_head, self.bio_head, self.rel_head]:
            for sub_module in mod.modules():
                if isinstance(sub_module, nn.Linear):
                    nn.init.xavier_uniform_(sub_module.weight)
                    if sub_module.bias is not None:
                        nn.init.zeros_(sub_module.bias)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encodes the input using the transformer model.

        Args:
            input_ids: Tensor of input token IDs.
            attention_mask: Tensor indicating which tokens to attend to.

        Returns:
            Tensor of hidden states from the encoder, passed through dropout
            and layer normalization.
        """
        hidden_states = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.layer_norm(self.dropout(hidden_states))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        bio_labels: torch.Tensor | None = None, 
        pair_batch: torch.Tensor | None = None,
        cause_starts: torch.Tensor | None = None,
        cause_ends: torch.Tensor | None = None,
        effect_starts: torch.Tensor | None = None,
        effect_ends: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor | None]:
        """Performs a forward pass through the model.

        Args:
            input_ids: Tensor of input token IDs.
            attention_mask: Tensor indicating which tokens to attend to.
            bio_labels: Optional tensor of BIO labels for training.
            pair_batch: Optional tensor indicating which hidden states to use
                for relation extraction.
            cause_starts: Optional tensor of start indices for cause spans.
            cause_ends: Optional tensor of end indices for cause spans.
            effect_starts: Optional tensor of start indices for effect spans.
            effect_ends: Optional tensor of end indices for effect spans.

        Returns:
            A dictionary containing:
                - "cls_logits": Logits for the classification task.
                - "bio_emissions": Emissions from the BIO tagging head.
                - "tag_loss": Loss for the BIO tagging task (if bio_labels provided).
                - "rel_logits": Logits for the relation extraction task (if
                  relation extraction inputs provided).
        """
        # Encode input
        hidden = self.encode(input_ids, attention_mask)

        # Classification head
        cls_logits = self.cls_head(hidden[:, 0])  # Use [CLS] token representation

        # BIO tagging head
        emissions  = self.bio_head(hidden)
        tag_loss: Optional[torch.Tensor] = None

        # Calculate BIO tagging loss if labels are provided
        if bio_labels is not None:
            # Softmax loss (typically handled by the training loop's loss function, e.g., CrossEntropyLoss)
            # Here, we initialize it to 0.0 as a placeholder.
            # The actual loss calculation for softmax would compare emissions with bio_labels.
            tag_loss = torch.tensor(0.0, device=emissions.device)

        # Relation extraction head
        rel_logits: torch.Tensor | None = None
        if pair_batch is not None and cause_starts is not None and cause_ends is not None \
           and effect_starts is not None and effect_ends is not None:
            # Select hidden states corresponding to the pairs for relation extraction
            bio_states_for_rel = hidden[pair_batch] 
            seq_len_rel = bio_states_for_rel.size(1)
            pos_rel = torch.arange(seq_len_rel, device=bio_states_for_rel.device).unsqueeze(0)

            # Create masks for cause and effect spans
            c_mask = ((cause_starts.unsqueeze(1) <= pos_rel) & (pos_rel <= cause_ends.unsqueeze(1))).unsqueeze(2)
            e_mask = ((effect_starts.unsqueeze(1) <= pos_rel) & (pos_rel <= effect_ends.unsqueeze(1))).unsqueeze(2)

            # Compute mean-pooled representations for cause and effect spans
            c_vec = (bio_states_for_rel * c_mask).sum(1) / c_mask.sum(1).clamp(min=1) # Average pooling, clamp to avoid div by zero
            e_vec = (bio_states_for_rel * e_mask).sum(1) / e_mask.sum(1).clamp(min=1) # Average pooling, clamp to avoid div by zero
            
            # Concatenate cause and effect vectors and pass through relation head
            rel_logits = self.rel_head(torch.cat([c_vec, e_vec], dim=1))

        return {
            "cls_logits": cls_logits,
            "bio_emissions": emissions,
            "tag_loss": tag_loss, 
            "rel_logits": rel_logits, 
        }

    def predict(self, sents: List[str], tokenizer=None, rel_mode="auto", rel_threshold=0.4, cause_decision="cls+span") -> list:
        """
        HuggingFace-compatible prediction method for causal extraction.
        Args:
            sents (List[str]): List of input sentences.
            tokenizer: Optional HuggingFace tokenizer. If None, uses self.encoder_name.
            rel_mode (str): 'auto' or 'head'.
            rel_threshold (float): Probability threshold for relation extraction.
            cause_decision (str): 'cls_only', 'span_only', or 'cls+span'.
        Returns:
            List of dicts with 'text', 'causal', and 'relations' fields for each sentence.
        """
        # Use id2label_bio from the module-level import instead of importing here
        # Only load tokenizer if not provided
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.encoder_name)
        device = next(self.parameters()).device
        outs = []
        for txt in sents:
            enc = tokenizer([txt], return_tensors="pt", truncation=True, max_length=512)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                rel_args = {}
                rel_pair_spans = []
                # Always prepare relation extraction arguments if needed (for head mode or auto mode with multi C/E)
                if rel_mode == "head":
                    res_tmp = self(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
                    bio_tmp = res_tmp["bio_emissions"].squeeze(0).argmax(-1).tolist()
                    tok_tmp = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze(0))
                    lab_tmp = [id2label_bio[i] for i in bio_tmp]
                    fixed_tmp = JointCausalModel._apply_bio_rules(tok_tmp, lab_tmp)
                    spans_tmp = JointCausalModel._merge_spans(tok_tmp, fixed_tmp)
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
                            "pair_batch": torch.tensor(pair_batch, device=device),
                            "cause_starts": torch.tensor(cause_starts, device=device),
                            "cause_ends": torch.tensor(cause_ends, device=device),
                            "effect_starts": torch.tensor(effect_starts, device=device),
                            "effect_ends": torch.tensor(effect_ends, device=device),
                        }
                res = self(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], **rel_args)
            cls = res["cls_logits"].squeeze(0)
            bio = res["bio_emissions"].squeeze(0).argmax(-1).tolist()
            tok = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze(0))
            lab = [id2label_bio[i] for i in bio]
            fixed = JointCausalModel._apply_bio_rules(tok, lab)
            spans = JointCausalModel._merge_spans(tok, fixed)
            causal = JointCausalModel._decide_causal(cls, spans, cause_decision)
            if not causal:
                outs.append({"text": txt, "causal": False, "relations": []})
                continue
            rels = []
            rel_logits = res.get("rel_logits")
            rel_probs = None
            if rel_logits is not None:
                rel_probs = torch.softmax(rel_logits, dim=-1)
            if rel_mode == "head":
                for idx, (csp, esp) in enumerate(rel_pair_spans):
                    if rel_probs[idx, 1].item() > rel_threshold:
                        rels.append({"cause": csp.text, "effect": esp.text, "type": "Rel_CE"})
            elif rel_mode == "auto":
                c_spans = [s for s in spans if s.role in ("C", "CE")]
                e_spans = [s for s in spans if s.role in ("E", "CE")]
                if not c_spans or not e_spans:
                    rels = []
                elif len(c_spans) == 1 and len(e_spans) >= 1:
                    for e in e_spans:
                        rels.append({"cause": c_spans[0].text, "effect": e.text, "type": "Rel_CE"})
                elif len(e_spans) == 1 and len(c_spans) >= 1:
                    for c in c_spans:
                        rels.append({"cause": c.text, "effect": e_spans[0].text, "type": "Rel_CE"})
                elif len(c_spans) > 1 and len(e_spans) > 1:
                    pair_batch = []
                    cause_starts = []
                    cause_ends = []
                    effect_starts = []
                    effect_ends = []
                    rel_pair_spans = []
                    for c in c_spans:
                        for e in e_spans:
                            if (c.start_tok == e.start_tok and c.end_tok == e.end_tok):
                                continue
                            pair_batch.append(0)
                            cause_starts.append(c.start_tok)
                            cause_ends.append(c.end_tok)
                            effect_starts.append(e.start_tok)
                            effect_ends.append(e.end_tok)
                            rel_pair_spans.append((c, e))
                    if pair_batch:
                        rel_args = {
                            "pair_batch": torch.tensor(pair_batch, device=device),
                            "cause_starts": torch.tensor(cause_starts, device=device),
                            "cause_ends": torch.tensor(cause_ends, device=device),
                            "effect_starts": torch.tensor(effect_starts, device=device),
                            "effect_ends": torch.tensor(effect_ends, device=device),
                        }
                        res_rel = self(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], **rel_args)
                        rel_logits = res_rel.get("rel_logits")
                        if rel_logits is not None:
                            rel_probs = torch.softmax(rel_logits, dim=-1)
                            for idx, (csp, esp) in enumerate(rel_pair_spans):
                                if rel_probs[idx, 1].item() > rel_threshold:
                                    rels.append({"cause": csp.text, "effect": esp.text, "type": "Rel_CE"})
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

    @staticmethod
    def _apply_bio_rules(tok, lab):
        """
        Apply post-processing rules to BIO tags to fix inconsistencies and clean up spans.
        - Fixes mixed-role spans, punctuation, short tokens, and CE disambiguation.
        """
        # Constants for punctuation, stopwords, and connectors
        _PUNCT = {".",",",";",":","?","!","(",")","[","]","{","}"}
        _STOPWORD_KEEP = {"this","that","these","those","it","they"}
        
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

    @staticmethod
    def _merge_spans(tok, lab):
        """
        Merge contiguous labeled tokens into Span objects, gluing across connectors.
        FIX: Start a new span for every B- tag, even if previous span is same type.
        Also trims leading/trailing connectors (e.g., 'and', 'or', 'but') from each span.
        """
        from transformers import AutoTokenizer
        try:
            from .config import MODEL_CONFIG
        except ImportError:
            from config import MODEL_CONFIG
        tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["encoder_name"])
        _CONNECTORS = {"of","to","with","for","the","and","or","but"}
        spans=[]; i=0
        while i<len(tok):
            if lab[i]=="O": i+=1; continue
            tag = lab[i]
            if tag.startswith("B-"):
                role=tag.split("-")[-1]; s=i; e=i
                # collect I- of same type
                while e+1<len(tok) and lab[e+1]==f"I-{role}":
                    e+=1
                spans.append(Span(role,s,e,tokenizer.convert_tokens_to_string(tok[s:e+1])))
                i=e+1
            else:
                i+=1
        merged=[spans[0]] if spans else []
        for sp in spans[1:]:
            prv=merged[-1]
            if sp.role==prv.role and sp.start_tok==prv.end_tok+2 and tok[prv.end_tok+1].lower() in _CONNECTORS:
                merged[-1]=Span(prv.role,prv.start_tok,sp.end_tok,tokenizer.convert_tokens_to_string(tok[prv.start_tok:sp.end_tok+1]),prv.is_virtual)
            else: merged.append(sp)
        # Trim leading/trailing connectors from each span
        trimmed = []
        for sp in merged:
            s, e = sp.start_tok, sp.end_tok
            # Trim from start
            while s <= e and tok[s].lower() in _CONNECTORS:
                s += 1
            # Trim from end
            while e >= s and tok[e].lower() in _CONNECTORS:
                e -= 1
            if s > e:
                continue  # skip empty span
            trimmed.append(Span(sp.role, s, e, tokenizer.convert_tokens_to_string(tok[s:e+1]), sp.is_virtual))
        return trimmed

    @staticmethod
    def _decide_causal(cls, spans, mode):
        if mode == "cls_only":
            return cls.argmax(-1).item() == 1
        elif mode == "span_only":
            return any(x.role == "C" for x in spans) and any(x.role == "E" for x in spans)
        elif mode == "cls+span":
            return (cls.argmax(-1).item() == 1) and (any(x.role == "C" for x in spans) and any(x.role == "E" for x in spans))
        else:
            raise ValueError(mode)

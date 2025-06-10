# debug_causal_detection.py
"""
Detailed debugging script for causal detection rules.
This script investigates the specific issue where the model detects sentences as causal
but no spans are found, even with cls+span rule.
"""
import json
import torch
from transformers import AutoTokenizer
import torch.nn.functional as F

try:
    from .config import MODEL_CONFIG, id2label_bio, id2label_cls
    from .model import JointCausalModel
except ImportError:
    from config import MODEL_CONFIG, id2label_bio, id2label_cls  # type: ignore
    from model import JointCausalModel  # type: ignore

# Load model (same path as test_predict.py)
MODEL_PATH = r"C:\Users\norouzin\Desktop\JointLearning\src\jointlearning\expert_bert_softmax\expert_bert_softmax_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_CONFIG["encoder_name"])

def debug_sentence_prediction(sentence, model, tokenizer):
    """Debug a single sentence prediction step by step."""
    print(f"\n{'='*80}")
    print(f"DEBUGGING SENTENCE: {sentence}")
    print(f"{'='*80}")
    
    # Tokenize
    enc = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze(0))
    print(f"\nTokens: {tokens}")
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    
    # Classification logits
    cls_logits = outputs["cls_logits"].squeeze(0)
    cls_probs = torch.softmax(cls_logits, dim=-1)
    cls_pred = cls_logits.argmax(-1).item()
    
    print(f"\nCLASSIFICATION:")
    print(f"  Logits: {cls_logits.tolist()}")
    print(f"  Probabilities: {cls_probs.tolist()}")
    print(f"  Prediction: {id2label_cls[cls_pred]} (confidence: {cls_probs[cls_pred]:.3f})")
    
    # BIO predictions
    bio_emissions = outputs["bio_emissions"].squeeze(0)
    bio_probs = torch.softmax(bio_emissions, dim=-1)
    bio_preds = bio_emissions.argmax(-1).tolist()
    bio_labels = [id2label_bio[i] for i in bio_preds]
    
    print(f"\nBIO PREDICTIONS (Raw):")
    for i, (token, pred_id, label, probs) in enumerate(zip(tokens, bio_preds, bio_labels, bio_probs)):
        if label != "O":  # Only show non-O predictions
            print(f"  {i:2d}: {token:15s} -> {label:4s} (conf: {probs[pred_id]:.3f})")
    
    # Apply BIO rules
    fixed_labels = model._apply_bio_rules(tokens, bio_labels)
    print(f"\nBIO PREDICTIONS (After rules):")
    for i, (token, orig_label, fixed_label) in enumerate(zip(tokens, bio_labels, fixed_labels)):
        if orig_label != fixed_label or fixed_label != "O":
            change_marker = " *** CHANGED ***" if orig_label != fixed_label else ""
            print(f"  {i:2d}: {token:15s} -> {orig_label:4s} -> {fixed_label:4s}{change_marker}")
    
    # Merge spans
    spans = model._merge_spans(tokens, fixed_labels, tokenizer)
    print(f"\nEXTRACTED SPANS:")
    if spans:
        for span in spans:
            print(f"  Role: {span.role:2s}, Tokens: {span.start_tok:2d}-{span.end_tok:2d}, Text: '{span.text}'")
    else:
        print("  No spans extracted!")
      # Test different causal decision rules
    print(f"\nCAUSAL DECISION RULES:")
    
    # cls_only
    cls_only_result = model._decide_causal(cls_logits, spans, "cls_only")
    print(f"  cls_only:  {cls_only_result} (cls_prob >= 0.5: {cls_probs[1].item() >= 0.5})")
    
    # span_only
    span_only_result = model._decide_causal(cls_logits, spans, "span_only")
    has_cause_spans = any(s.role in ["C", "CE"] for s in spans)
    has_effect_spans = any(s.role in ["E", "CE"] for s in spans)
    print(f"  span_only: {span_only_result} (has_C/CE: {has_cause_spans}, has_E/CE: {has_effect_spans})")
    
    # cls+span
    cls_span_result = model._decide_causal(cls_logits, spans, "cls+span")
    print(f"  cls+span:  {cls_span_result} (cls_only AND span_only)")
    
    # Show relation extraction analysis
    print(f"\nRELATION EXTRACTION ANALYSIS:")
    cause_spans = [s for s in spans if s.role in ("C", "CE")]
    effect_spans = [s for s in spans if s.role in ("E", "CE")]
    print(f"  Cause spans: {[(s.role, s.text) for s in cause_spans]}")
    print(f"  Effect spans: {[(s.role, s.text) for s in effect_spans]}")
    print(f"  Can form relations: {bool(cause_spans and effect_spans)}")
    
    # Call the predict method for comparison
    print(f"\nPREDICT METHOD RESULTS:")
    for cause_decision in ["cls_only", "span_only", "cls+span"]:
        result = model.predict([sentence], tokenizer, cause_decision=cause_decision)
        print(f"  {cause_decision:10s}: causal={result[0]['causal']}, relations={len(result[0]['relations'])}")
        if result[0]['relations']:
            for rel in result[0]['relations']:
                print(f"    - '{rel['cause']}' -> '{rel['effect']}'")
    
    return {
        "sentence": sentence,
        "cls_prediction": cls_pred,
        "cls_probability": cls_probs[1].item(),
        "spans": [(s.role, s.text) for s in spans],
        "cls_only": cls_only_result,
        "span_only": span_only_result,
        "cls_span": cls_span_result,
        "can_form_relations": bool(cause_spans and effect_spans),
    }

def main():
    print("Loading model...")
    model = JointCausalModel(**MODEL_CONFIG)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    print(f"Model loaded on {DEVICE}")
    
    # Test sentences - focusing on the problematic ones mentioned
    test_sentences = [
        "in many programme areas, this is in fact possible since there are frequently follow-on programmes whose planning stages could deploy a review approach, but of course it is rarely done.;;",
        "it is also considered that the process could easily be adapted for virtual delivery, thus increasing its accessibility.;;",
        # Add some clearly causal sentences for comparison
        "smoking causes lung cancer and heart disease",
        "exercise improves physical health and mental well-being",
        # Add a clearly non-causal sentence
        "The book is on the table."    ]
    
    results = []
    for sentence in test_sentences:
        result = debug_sentence_prediction(sentence, model, TOKENIZER)
        results.append(result)
    
    print(f"\n{'='*80}")
    print("SUMMARY OF RESULTS:")
    print(f"{'='*80}")
    
    for result in results:
        print(f"\nSentence: {result['sentence'][:50]}...")
        print(f"  CLS prob: {result['cls_probability']:.3f}")
        print(f"  Spans: {result['spans']}")
        print(f"  Can form relations: {result['can_form_relations']}")
        print(f"  Rules - cls_only: {result['cls_only']}, span_only: {result['span_only']}, cls+span: {result['cls_span']}")
    
    # Look for problematic cases
    print(f"\nPROBLEMATIC CASES:")
    print("1. Classification says causal but no cause/effect span pairs:")
    for result in results:
        if result['cls_only'] and not result['can_form_relations']:
            print(f"  - {result['sentence'][:50]}...")
            print(f"    CLS says causal ({result['cls_probability']:.3f}) but can't form relations")
    
    print("\n2. Has spans but span_only rules say not causal:")
    for result in results:
        if result['spans'] and not result['span_only']:
            print(f"  - {result['sentence'][:50]}...")
            print(f"    Has spans {result['spans']} but span_only=False")

if __name__ == "__main__":
    main()

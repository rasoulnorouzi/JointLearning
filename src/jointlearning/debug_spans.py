from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Tuple, Optional, Any, List
from huggingface_hub import PyTorchModelHubMixin
from model import JointCausalModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    from .config import MODEL_CONFIG, id2label_bio, id2label_rel, id2label_cls
except ImportError:
    from config import MODEL_CONFIG, id2label_bio, id2label_rel, id2label_cls

tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["encoder_name"])
model_path = r"C:\Users\norouzin\Desktop\JointLearning\src\jointlearning\expert_bert_softmax\expert_bert_softmax_model.pt"
model = JointCausalModel(**MODEL_CONFIG)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Test the specific drought example
test_text = "The prolonged drought led to widespread crop failure, which in turn caused a sharp increase in food prices, ultimately contributing to social unrest in the region."

print(f"Input Text: {test_text}")

# Get raw predictions first
tokenized_input = tokenizer(
    [test_text],
    padding=False,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)
tokenized_input = {k: v.to(device) for k, v in tokenized_input.items()}

with torch.no_grad():
    result = model(
        input_ids=tokenized_input["input_ids"],
        attention_mask=tokenized_input["attention_mask"]
    )

# Extract raw BIO predictions
bio_emissions = result["bio_emissions"]
argmax_bio = torch.argmax(bio_emissions, dim=-1)
tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"][0])

seq_len = tokenized_input["attention_mask"][0].sum().item()
input_ids = tokenized_input["input_ids"][0][:seq_len]
bio_emissions_trimmed = bio_emissions[0][:seq_len]
tokens = tokenizer.convert_ids_to_tokens(input_ids)
bio_ids = bio_emissions_trimmed.argmax(-1).tolist()
bio_labels = [id2label_bio[j] for j in bio_ids]

print("\nRaw BIO predictions:")
for token, label in zip(tokens, bio_labels):
    print(f"  {token}: {label}")

# Apply BIO rules
fixed_labels = model._apply_bio_rules(tokens, bio_labels)
print("\nAfter BIO rules:")
for token, label in zip(tokens, fixed_labels):
    print(f"  {token}: {label}")

# Get spans
spans = model._merge_spans(tokens, fixed_labels, tokenizer)
print(f"\nDetected spans ({len(spans)}):")
for span in spans:
    print(f"  '{span.text}' -> {span.role} (tokens {span.start_tok}-{span.end_tok})")

# Categorize spans
pure_cause_spans = [s for s in spans if s.role == "C"]
pure_effect_spans = [s for s in spans if s.role == "E"]
ce_spans = [s for s in spans if s.role == "CE"]

print(f"\nCause spans: {[s.text for s in pure_cause_spans]}")
print(f"Effect spans: {[s.text for s in pure_effect_spans]}")
print(f"CE spans: {[s.text for s in ce_spans]}")

# Now use the predict method with different thresholds
print("\n=== Testing different relation thresholds ===")

for threshold in [0.5, 0.7, 0.8, 0.9]:
    predictions = model.predict(
        sents=[test_text],
        tokenizer=tokenizer,
        rel_mode="neural_only",
        rel_threshold=threshold,
        cause_decision="cls+span"
    )
    
    prediction = predictions[0]
    print(f"\nThreshold {threshold}:")
    print(f"Relations: {prediction['relations']}")

# Also test with auto mode
predictions_auto = model.predict(
    sents=[test_text],
    tokenizer=tokenizer,
    rel_mode="auto",
    rel_threshold=0.8,
    cause_decision="cls+span"
)

prediction_auto = predictions_auto[0]
print(f"\nAuto mode:")
print(f"Relations: {prediction_auto['relations']}")

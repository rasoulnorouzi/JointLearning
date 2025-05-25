
from __future__ import annotations
import torch
import torch.nn as nn
import string
import traceback
import types # Added import for types.MethodType
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Tuple, Optional, Any, List
from huggingface_hub import PyTorchModelHubMixin # Ensure this is imported
from model import JointCausalModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    from .config import MODEL_CONFIG, id2label_bio, id2label_rel, id2label_cls
except ImportError:
    from config import MODEL_CONFIG, id2label_bio, id2label_rel, id2label_cls


tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["encoder_name"])
model_path = "/home/rnorouzini/JointLearning/src/jointlearning/expert_bert_softmax/expert_bert_softmax_model.pt"
model = JointCausalModel(**MODEL_CONFIG)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

test = "i believe that by publishing more auto ethno graph ies in first-person voice, applied linguists will help transform applied linguistics into a more humanized and decolonialized field"

tokenized_input = tokenizer(
    test,
    padding=False,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)
tokenized_input = {k: v.to(device) for k, v in tokenized_input.items()}

print(
    model(
        input_ids=tokenized_input["input_ids"],
        attention_mask=tokenized_input["attention_mask"],
        token_type_ids=tokenized_input["token_type_ids"]
    )
)
result = model(
    input_ids=tokenized_input["input_ids"],
    attention_mask=tokenized_input["attention_mask"],
    token_type_ids=tokenized_input["token_type_ids"]
)
# %%
print(torch.argmax(result["cls_logits"], dim=-1))
# %%
bio_emissions = result["bio_emissions"]
# argmax_bio

argmax_bio = torch.argmax(bio_emissions, dim=-1)
print(argmax_bio)
# %%
# print tokenized words
tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"][0])
print(tokens)
print(len(tokens))
print(len(argmax_bio[0]))
map_tokens_to_bio = {token: id2label_bio[label.item()] for token, label in zip(tokens, argmax_bio[0])}
print(map_tokens_to_bio)
# %%
from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Tuple, Optional, Any, List
from huggingface_hub import PyTorchModelHubMixin # Ensure this is imported
from model import JointCausalModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    from .config import MODEL_CONFIG, id2label_bio, id2label_rel, id2label_cls
except ImportError:
    from config import MODEL_CONFIG, id2label_bio, id2label_rel, id2label_cls

print(f"id2label_bio: {id2label_bio}")
print(f"id2label_rel: {id2label_rel}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["encoder_name"])
model_path = r"C:\Users\norouzin\Desktop\JointLearning\src\jointlearning\expert_bert_softmax\expert_bert_softmax_model.pt"
model = JointCausalModel(**MODEL_CONFIG)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

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
        "thus, experiencing a task that leads to decrements in feelings of relatedness may affect people's ability to experience intrinsic motivation by also influencing people's mood.;;"
        ]

for i, text_sample in enumerate(test):
    print(f"--- Sample {i+1} ---")
    print(f"Input Text: {text_sample}")

    tokenized_input = tokenizer(
        [text_sample],  # Pass a list containing the single sample
        padding=False,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    tokenized_input = {k: v.to(device) for k, v in tokenized_input.items()}

    with torch.no_grad(): # Ensure no gradients are calculated during inference
        result = model(
            input_ids=tokenized_input["input_ids"],
            attention_mask=tokenized_input["attention_mask"],
            token_type_ids=tokenized_input["token_type_ids"]
        )

    # %%
    cls_prediction = torch.argmax(result["cls_logits"], dim=-1)
    print(f"CLS Prediction: {cls_prediction.item()} ({id2label_cls[cls_prediction.item()]})")

    # %%
    bio_emissions = result["bio_emissions"]
    argmax_bio = torch.argmax(bio_emissions, dim=-1)

    print("BIO Predictions:")
    tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"][0])
    
    # Ensure tokens and argmax_bio[0] have the same length for zipping
    min_len = min(len(tokens), len(argmax_bio[0]))
    map_tokens_to_bio = {token: id2label_bio[label.item()] for token, label in zip(tokens[:min_len], argmax_bio[0][:min_len])}

    # Print token-BIO label pairs
    for token, bio_label in map_tokens_to_bio.items():
        print(f"  {token}: {bio_label}")
    
    print("-" * 20)

# %%
# The following lines are commented out as they are processed inside the loop now.
# print(torch.argmax(result["cls_logits"], dim=-1))
# # %%
# bio_emissions = result["bio_emissions"]
# # argmax_bio

# argmax_bio = torch.argmax(bio_emissions, dim=-1)
# print(argmax_bio)
# # %%
# # print tokenized words
# tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"][0])
# print(tokens)
# print(len(tokens))
# print(len(argmax_bio[0]))
# map_tokens_to_bio = {token: id2label_bio[label.item()] for token, label in zip(tokens, argmax_bio[0])}
# print(map_tokens_to_bio)
# %%
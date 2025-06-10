# %%
from transformers import AutoModel, AutoTokenizer
import os
import json
import pandas as pd
import numpy as np
from model import JointCausalModel
from config import MODEL_CONFIG
import torch
# %%
model = JointCausalModel(
    **MODEL_CONFIG
)
# %%
model_weights_path = r"C:\Users\norouzin\Desktop\JointLearning\src\jointlearning\expert_bert_GCE_Softmax_Normal\expert_bert_GCE_Softmax_Normal_model.pt"
model.load_state_dict(torch.load(model_weights_path, map_location="cpu"))
# %%

df_json = pd.read_json(
    r"C:\Users\norouzin\Desktop\JointLearning\datasets\expert_multi_task_data\doccano_test.jsonl",
    orient="records",
    lines=True
)
df_json.head()
# %%
print(f"Number of test samples: {len(df_json)}")
# %%
# get the sentences from the dataframe in the column "text"
test_sents = df_json["text"].tolist()
# %%
print(f"Number of test sentences: {len(test_sents)}")
# %%
tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\norouzin\Desktop\JointLearning\src\jointlearning\expert_bert_GCE_Softmax_Normal\hf_exper_bert_GCE_Softmax_Normal")
# %%
# get the model predictions for the batched sentences
batch_size = 32  # You can adjust this as needed
all_results = []
num_batches = (len(test_sents) + batch_size - 1) // batch_size
for i in range(num_batches):
    batch = test_sents[i * batch_size : (i + 1) * batch_size]
    batch_results = model.predict(
        batch,
        tokenizer=tokenizer,
        rel_mode="auto",           # or "auto"
        rel_threshold=0.5,         # adjust as needed
        cause_decision="cls+span" # or "cls_only", "span_only"
    )
    all_results.extend(batch_results)
    print(f"Processed batch {i+1}/{num_batches} ({len(batch)} sentences)")
# %%
print(f"Total results collected: {len(all_results)}")
# %% 
print(json.dumps(all_results, indent=2, ensure_ascii=False))
# %%
with open("causenet_test_results.jsonl", "w", encoding="utf-8") as f:
    for item in all_results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
# %%
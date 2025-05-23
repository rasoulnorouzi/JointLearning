# %%
from jointlearning.model import JointCausalModel
# import dataloader from pytorch
from torch.utils.data import DataLoader
from dataset_collator import CausalDataset, CausalDatasetCollator
from transformers import AutoTokenizer
import torch
import pandas as pd
import os
import numpy as np
from config import MODEL_CONFIG, TRAINING_CONFIG, DATASET_CONFIG, DEVICE
# %%
model = JointCausalModel(
    encoder_name=MODEL_CONFIG["encoder_name"],
    num_cls_labels=MODEL_CONFIG["num_cls_labels"],
    num_bio_labels=MODEL_CONFIG["num_bio_labels"],
    num_rel_labels=MODEL_CONFIG["num_rel_labels"],
    dropout=MODEL_CONFIG["dropout"]
)
# %%
train_data_path = "C:\\Users\\norouzin\\Desktop\\JointLearning\\datasets\\expert_multi_task_data\\train.csv"

df_train = pd.read_csv(train_data_path)

# %%
train_dataset = CausalDataset(
    df_train,
    tokenizer_name='bert-base-uncased',
    max_length=DATASET_CONFIG["max_length"]
)
# %%
train_collator = CausalDatasetCollator(
    tokenizer=train_dataset.tokenizer
)
# %%
train_dataloader = DataLoader(
    train_dataset,
    batch_size=TRAINING_CONFIG["batch_size"],
    collate_fn=train_collator,
    shuffle=True
)
# %%

for batch in train_dataloader:
    print(batch)
    batch
    break
# %%
batch
# %%
model.to(DEVICE)
print(
model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    bio_labels=batch["bio_labels"],
    pair_batch=batch["pair_batch"],
    cause_starts=batch["cause_starts"],
    cause_ends=batch["cause_ends"],
    effect_starts=batch["effect_starts"],
    effect_ends=batch["effect_ends"]
)   
)
# %%
# %%
from dataset_collator import CausalDatasetCollator, CausalDataset, id2label_bio, id2label_rel, id2label_cls, ignore_id
import pandas as pd
from utility import compute_class_weights, label_value_counts
from torch.utils.data import DataLoader
from model import JointCausalModel
import torch
import numpy as np
from sklearn.metrics import classification_report
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%
train_df = pd.read_csv("datasets/train.csv")
val_df = pd.read_csv("datasets/val.csv")
# %%
checkpoint = "google-bert/bert-base-uncased"
max_length_truncate=256
num_samples_to_report=1
negative_rel_rate_for_report=2.0
# %%
train_dataset_instance = CausalDataset(train_df, checkpoint, max_length_truncate, negative_rel_rate_for_report)
val_dataset_instance = CausalDataset(val_df, checkpoint, max_length_truncate, negative_rel_rate_for_report)
# %%
cls_labels, bio_labels, rel_labels = label_value_counts(train_dataset_instance)
# %%
cls_weights = compute_class_weights(cls_labels, num_classes=len(id2label_cls))
bio_weights = compute_class_weights(bio_labels, num_classes=len(id2label_bio), ignore_index=ignore_id)
rel_weights = compute_class_weights(rel_labels, num_classes=len(id2label_rel), ignore_index=ignore_id)
# %%
collator_fn = CausalDatasetCollator(
    tokenizer = train_dataset_instance.tokenizer,
    num_rel_labels_model_expects= len(id2label_rel)
    )
train_loader = DataLoader(
    train_dataset_instance,
    batch_size=8,
    collate_fn=collator_fn,
    shuffle=True
)
val_loader = DataLoader(
    val_dataset_instance,
    batch_size=8,
    collate_fn=collator_fn,
    shuffle=False
)
# %%
cls_loss_fn = torch.nn.CrossEntropyLoss(weight=cls_weights)
bio_loss_fn = torch.nn.CrossEntropyLoss(weight=bio_weights, ignore_index=ignore_id)
rel_loss_fn = torch.nn.CrossEntropyLoss(weight=rel_weights, ignore_index=ignore_id)
# %%
model = JointCausalModel(
    encoder_name=checkpoint,
    num_cls_labels=len(id2label_cls),
    num_bio_labels=len(id2label_bio),
    num_rel_labels=len(id2label_rel)
).to(device)
# %%
# print 1 sample from dataloader

for batch in train_loader:
    print("#" * 20)
    print(batch)
    print("#" * 20)
    break
# %%
# feed a batch of data to the model
for batch in train_loader:
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    cls_labels = batch["cls_labels"].to(device)
    bio_labels = batch["bio_labels"].to(device)
    rel_labels_gold = batch['rel_labels'].to(device) if batch['rel_labels'] is not None else None

    pair_batch_data = batch['pair_batch'].to(device) if batch['pair_batch'] is not None else None
    cause_starts_data = batch['cause_starts'].to(device) if batch['cause_starts'] is not None else None
    cause_ends_data = batch['cause_ends'].to(device) if batch['cause_ends'] is not None else None
    effect_starts_data = batch['effect_starts'].to(device) if batch['effect_starts'] is not None else None
    effect_ends_data = batch['effect_ends'].to(device) if batch['effect_ends'] is not None else None

    # Forward pass
    output = model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                pair_batch=pair_batch_data, 
                cause_starts=cause_starts_data, 
                cause_ends=cause_ends_data,
                effect_starts=effect_starts_data, 
                effect_ends=effect_ends_data
            )
    print(output)
    print("#" * 20)
    print(f"input_ids: {input_ids.shape}")
    print(f"attention_mask: {attention_mask.shape}")
    print(f"cls_labels: {cls_labels.shape}")
    print(f"bio_labels: {bio_labels.shape}")
    print(f"rel_labels_gold: {rel_labels_gold.shape}")
    print(f"pair_batch_data: {pair_batch_data.shape}")
    print(f"cause_starts_data: {cause_starts_data.shape}")
    print(f"cause_ends_data: {cause_ends_data.shape}")
    print(f"effect_starts_data: {effect_starts_data.shape}")
    print(f"effect_ends_data: {effect_ends_data.shape}")
    print(f"output['cls_logits']: {output['cls_logits'].shape}")
    print(f"output['bio_logits']: {output['bio_logits'].shape}")
    print(f"output['rel_logits']: {output['rel_logits'].shape}")


    cls_loss = cls_loss_fn(output['cls_logits'], cls_labels)
    bio_loss = bio_loss_fn(output['bio_logits'].view(-1, len(id2label_bio)), bio_labels.view(-1))
    rel_loss = rel_loss_fn(output['rel_logits'].view(-1, len(id2label_rel)), rel_labels_gold.view(-1))
   
    print("#" * 20)
    print(f"cls_logits: {output['cls_logits'].shape}")
    print(f"bio_logits: {output['bio_logits'].shape}")
    print(f"rel_logits: {output['rel_logits'].shape}")
    
    print(f"cls_loss: {cls_loss.item()}")
    print(f"bio_loss: {bio_loss.item()}")
    print(f"rel_loss: {rel_loss.item()}")
    break
# %%
cls_loss = cls_loss_fn(output['cls_logits'], cls_labels)
print(f"cls_loss: {cls_loss.item()}")
# %%
bio_loss = bio_loss_fn(output['bio_logits'].view(-1, len(id2label_bio)), bio_labels.view(-1))
print(f"bio_loss: {bio_loss.item()}")
# %%
output['bio_logits'].view(-1, len(id2label_bio)).shape
# %%
rel_loss = rel_loss_fn(output['rel_logits'], rel_labels_gold)
print(f"rel_loss: {rel_loss.item()}")
# %%
output['rel_logits'].shape
# %%
rel_labels_gold.shape
# %%
output['rel_logits'].view(-1, len(id2label_rel)).shape
# %%
rel_labels_gold.view(-1).shape
# %%

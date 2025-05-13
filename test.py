# %%
import random
from dataset_collator import CausalDatasetCollator, CausalDataset, id2label_span, id2label_rel, NEGATIVE_SAMPLE_REL_ID
import pandas as pd
from utility import compute_class_weights
from torch.utils.data import DataLoader
from model import JointCausalModel
import torch
# %%
training_data_path = "datasets/train.csv"
# %%
train_df = pd.read_csv(training_data_path)
# %%
full_dataset_instance = CausalDataset(train_df, "google-bert/bert-base-uncased", 256, 2.0)
# %%
all_cls_labels = []
for i in range(len(full_dataset_instance)):
    sample = full_dataset_instance[i]
    all_cls_labels.append(sample['cls_label'])
# %%
# value counts for cls_labels with numpy
import numpy as np
cls_labels = np.array(all_cls_labels)
unique_cls_labels, cls_counts = np.unique(cls_labels, return_counts=True)
print("Unique cls_labels:", unique_cls_labels)
print("Counts of cls_labels:", cls_counts)
# %%
BIO_IGNORE_INDEX = -100
# %%
# Define num_classes for sentence classification (e.g., 2 for causal/non-causal)
NUM_CLS_CLASSES = 2 # Example
cls_weights = compute_class_weights(all_cls_labels, NUM_CLS_CLASSES, technique='inverse_frequency')
print(f"Sentence Classification Weights ({NUM_CLS_CLASSES} classes): {cls_weights}")
# %%
# 2. For BIO Tagging ('bio_labels')
all_bio_labels_flat = []
for i in range(len(full_dataset_instance)):
    sample = full_dataset_instance[i]
    all_bio_labels_flat.extend(sample['bio_labels']) # .extend() as bio_labels is a list

# Define num_classes for BIO tagging (e.g., len(id2label_span))
NUM_BIO_CLASSES = len(id2label_span) # Example: 7
bio_weights = compute_class_weights(
    all_bio_labels_flat,
    NUM_BIO_CLASSES,
    technique='median_frequency',
    ignore_index=BIO_IGNORE_INDEX
)
print(f"BIO Tagging Weights ({NUM_BIO_CLASSES} classes): {bio_weights}")
# %%
collator_fn = CausalDatasetCollator(
        tokenizer=full_dataset_instance.tokenizer,
        num_rel_labels_model_expects=len(id2label_rel) # Use actual number of defined relation labels
)
# %%
train_loader = DataLoader(
    full_dataset_instance,
    batch_size=5,
    shuffle=True,
    collate_fn=collator_fn
)
# %%
# print the three first batches
for i, batch in enumerate(train_loader):
    if i < 3:
        print(f"Batch {i}:")
        for key, value in batch.items():
            print(f"  {key}: {value}")
    else:
        break

# %%
model = JointCausalModel(
    encoder_name="google-bert/bert-base-uncased",
    num_cls_labels=NUM_CLS_CLASSES,
    num_span_labels=NUM_BIO_CLASSES,
    num_rel_labels=len(id2label_rel) # Use actual number of defined relation labels
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
# %%
# Lets feed a batch to the model
batch = next(iter(train_loader))
# Move the batch to the same device as the model
batch = {k: v.to(device) for k, v in batch.items()}
# Feed the batch to the model

'''
make a dictionary of input_ids=batch['input_ids'],
    attention_mask=batch['attention_mask'],
    pair_batch=batch['pair_batch'],
    cause_starts=batch['cause_starts'],
    cause_ends=batch['cause_ends'],
    effect_starts=batch['effect_starts'],
    effect_ends=batch['effect_ends'],
    rel_labels=batch['rel_labels']

'''
input_dict = {
    "input_ids": batch['input_ids'],
    "attention_mask": batch['attention_mask'],
    "pair_batch": batch['pair_batch'],
    "cause_starts": batch['cause_starts'],
    "cause_ends": batch['cause_ends'],
    "effect_starts": batch['effect_starts'],
    "effect_ends": batch['effect_ends']
}
output = model(
    **input_dict    
)
# Print the output
# %%
print(output)
# %%

batch
# %%

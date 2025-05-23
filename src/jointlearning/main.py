import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import random

from config import DEVICE, SEED, MODEL_CONFIG, TRAINING_CONFIG, DATASET_CONFIG
from model import JointCausalModel
from utility import compute_class_weights, label_value_counts
from dataset_collator import CausalDataset, CausalDatasetCollator
from config import id2label_cls, id2label_bio, id2label_rel
from evaluate_joint_causal_model import evaluate_model, print_eval_report
from trainer import train_model

# %%
train_data_path = "C:\\Users\\norouzin\\Desktop\\JointLearning\\datasets\\expert_multi_task_data\\train.csv"
val_data_path = "C:\\Users\\norouzin\\Desktop\\JointLearning\\datasets\\expert_multi_task_data\\val.csv"
train_df = pd.read_csv(train_data_path)
val_df = pd.read_csv(val_data_path)
# %%
train_dataset = CausalDataset(
    train_df,
    tokenizer_name='bert-base-uncased',
    max_length=DATASET_CONFIG["max_length"],
    
)
# %%
val_dataset = CausalDataset(
    val_df,
    tokenizer_name='bert-base-uncased',
    max_length=DATASET_CONFIG["max_length"],
)
# %%
labels_flat = label_value_counts(train_dataset)
# %%
cls_label_flat = labels_flat["cls_labels_flat"]
bio_label_flat = labels_flat["bio_labels_flat"]
rel_label_flat = labels_flat["rel_labels_flat"]
# %%
# Calculate class weights
cls_weights = compute_class_weights(labels_list=cls_label_flat, num_classes=MODEL_CONFIG["num_cls_labels"])
bio_weights = compute_class_weights(labels_list=bio_label_flat, num_classes=MODEL_CONFIG["num_bio_labels"])
rel_weights = compute_class_weights(labels_list=rel_label_flat, num_classes=MODEL_CONFIG["num_rel_labels"])
# %%
collator = CausalDatasetCollator(
    tokenizer=train_dataset.tokenizer
)
# %%
# take a 100 samples from train_dataset
# train_dataset = torch.utils.data.Subset(train_dataset, random.sample(range(len(train_dataset)), 20))
# val_dataset = torch.utils.data.Subset(val_dataset, random.sample(range(len(val_dataset)), 20))
# %%
train_dataloader = DataLoader(
    train_dataset,
    batch_size=TRAINING_CONFIG["batch_size"],
    collate_fn=collator,
    shuffle=True

)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=TRAINING_CONFIG["batch_size"],
    collate_fn=collator,
    shuffle=False
)
# %%
model = JointCausalModel(
    encoder_name=MODEL_CONFIG["encoder_name"],
    num_cls_labels=MODEL_CONFIG["num_cls_labels"],
    num_bio_labels=MODEL_CONFIG["num_bio_labels"],
    num_rel_labels=MODEL_CONFIG["num_rel_labels"],
    dropout=MODEL_CONFIG["dropout"],
    use_crf=False
)
# %%
optimizer = optim.AdamW(
    model.parameters(),
    lr=TRAINING_CONFIG["learning_rate"],
    weight_decay=TRAINING_CONFIG["weight_decay"]
)
# %%
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=2
)
model_save_path = r"C:\\Users\\norouzin\\Desktop\\JointLearning\\src\\jointlearning\\softmax_model\\best_softmax_model.pt"
# %%
# Train the model
if __name__ == '__main__':
    trained_model, training_history = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        num_epochs=TRAINING_CONFIG["num_epochs"],
        device=DEVICE,
        id2label_cls=id2label_cls,
        id2label_bio=id2label_bio,
        id2label_rel=id2label_rel,
        model_save_path=model_save_path,
        scheduler=scheduler,
        cls_class_weights=cls_weights,
        bio_class_weights=bio_weights, # Only for softmax
        rel_class_weights=rel_weights,
        patience_epochs=TRAINING_CONFIG["patience_epochs"],
        seed=SEED,
        max_grad_norm=TRAINING_CONFIG["gradient_clip_val"],
        eval_fn_metrics=evaluate_model, # Pass your evaluate_model function here
        print_report_fn=print_eval_report # Pass your print_eval_report function here
    )
# %%
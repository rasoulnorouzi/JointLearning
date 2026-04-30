#!/usr/bin/env python3
"""
Train 3 separate single-task models for the ablation study.

Each model uses the same JointCausalModel architecture but with task_loss_weights
that zero out the two unused heads, so the shared encoder optimizes for only one task.

Models produced:
    - separate_cls.pt  : CLS-only  (sentence classification)
    - separate_bio.pt  : BIO-only  (span detection / BIO tagging)
    - separate_rel.pt  : REL-only  (relation extraction, given gold spans)

Usage:
    python train_separate.py [--task cls|bio|rel|all] [--device cuda|cpu]
"""
import sys
import os
import argparse

# Ensure src/ and src/jointlearning/ are importable
# (jointlearning modules use bare imports that need the package dir on sys.path)
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
for d in [SRC_DIR, os.path.join(SRC_DIR, "jointlearning")]:
    if d not in sys.path:
        sys.path.insert(0, d)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

from jointlearning.config import DEVICE as _default_device, MODEL_CONFIG as _default_model_cfg
from jointlearning.model import JointCausalModel
from jointlearning.utility import compute_class_weights, label_value_counts, set_seed, seed_worker
from jointlearning.dataset_collator import CausalDataset, CausalDatasetCollator
from jointlearning.trainer import train_model
from jointlearning.evaluate_joint_causal_model import evaluate_model, print_eval_report

from cmp_config import (
    TRAIN_DATA_PATH, VAL_DATA_PATH, SEED,
    TRAINING_CONFIG, DATASET_CONFIG, TASK_WEIGHTS, MODEL_PATHS, DEVICE,
    id2label_cls, id2label_bio, id2label_rel, TRAINING_LOGS,
)


def make_eval_fn(active_task: str):
    """Wrap evaluate_model so overall_avg_f1 tracks only the active task.

    For single-task models the untrained heads add noise to the average.
    This wrapper overrides ``overall_avg_f1`` to use only the task being trained,
    so early stopping and ReduceLROnPlateau track the right signal.
    """
    def _eval_fn(model, dataloader, device, id2label_cls, id2label_bio, id2label_rel):
        results = evaluate_model(model, dataloader, device, id2label_cls, id2label_bio, id2label_rel)

        task_key = {"cls": "task_cls", "bio": "task_bio", "rel": "task_relation"}[active_task]
        task_metrics = results.get(task_key, {})
        if isinstance(task_metrics, dict) and "macro avg" in task_metrics:
            f1 = task_metrics["macro avg"].get("f1-score", 0.0)
        else:
            f1 = 0.0

        results["overall_avg_f1"] = f1
        return results
    return _eval_fn


def load_data():
    """Load train/val CSVs, build datasets, compute class weights, return loaders."""
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    val_df = pd.read_csv(VAL_DATA_PATH)

    train_dataset = CausalDataset(train_df, tokenizer_name="bert-base-uncased", max_length=DATASET_CONFIG["max_length"])
    val_dataset = CausalDataset(val_df, tokenizer_name="bert-base-uncased", max_length=DATASET_CONFIG["max_length"])

    labels_flat = label_value_counts(train_dataset)

    cls_weights = compute_class_weights(labels_flat["cls_labels_flat"], num_classes=2, technique="ens", ignore_index=-100)
    bio_weights = compute_class_weights(labels_flat["bio_labels_flat"], num_classes=7, technique="ens", ignore_index=-100)
    rel_weights = compute_class_weights(labels_flat["rel_labels_flat"], num_classes=2, technique="ens", ignore_index=-100)

    collator = CausalDatasetCollator(tokenizer=train_dataset.tokenizer)

    _g = torch.Generator()
    _g.manual_seed(SEED)

    train_loader = DataLoader(train_dataset, batch_size=TRAINING_CONFIG["batch_size"], collate_fn=collator,
                              shuffle=True, generator=_g, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_dataset, batch_size=TRAINING_CONFIG["batch_size"], collate_fn=collator,
                            shuffle=False, worker_init_fn=seed_worker)

    return train_loader, val_loader, cls_weights, bio_weights, rel_weights


def train_one(task: str, train_loader, val_loader, cls_weights, bio_weights, rel_weights):
    """Train a single-task model, save checkpoint + log, return model."""
    import sys as _sys

    log_path = TRAINING_LOGS[task]
    log_fh = open(log_path, "w", encoding="utf-8")

    class _Tee:
        """Write to both stdout/stderr and a log file."""
        def __init__(self, fh, orig):
            self.fh = fh
            self.orig = orig
        def write(self, data):
            self.fh.write(data)
            self.fh.flush()
            return self.orig.write(data)
        def flush(self):
            self.fh.flush()
            return self.orig.flush()

    _stdout_orig = _sys.stdout
    _stderr_orig = _sys.stderr
    _sys.stdout = _Tee(log_fh, _stdout_orig)
    _sys.stderr = _Tee(log_fh, _stderr_orig)

    try:
        print(f"\n{'='*60}\n  Training {task.upper()}-only model\n{'='*60}")

        set_seed(SEED)

        model = JointCausalModel(**_default_model_cfg)

        optimizer = optim.AdamW(model.parameters(), lr=TRAINING_CONFIG["learning_rate"],
                                weight_decay=TRAINING_CONFIG["weight_decay"])

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)

        save_path = MODEL_PATHS[task]
        task_weights = TASK_WEIGHTS[task]
        eval_fn = make_eval_fn(task)

        trained_model, history = train_model(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            optimizer=optimizer,
            num_epochs=TRAINING_CONFIG["num_epochs"],
            device=DEVICE,
            id2label_cls=id2label_cls,
            id2label_bio=id2label_bio,
            id2label_rel=id2label_rel,
            model_save_path=save_path,
            scheduler=scheduler,
            cls_class_weights=cls_weights,
            bio_class_weights=bio_weights,
            rel_class_weights=rel_weights,
            patience_epochs=TRAINING_CONFIG["patience_epochs"],
            seed=SEED,
            max_grad_norm=TRAINING_CONFIG["gradient_clip_val"],
            eval_fn_metrics=eval_fn,
            print_report_fn=print_eval_report,
            task_loss_weights=task_weights,
        )

        print(f"  {task.upper()}-only model saved to {save_path}")
        print(f"  Training log saved to {log_path}")
        return trained_model

    finally:
        _sys.stdout = _stdout_orig
        _sys.stderr = _stderr_orig
        log_fh.close()
        # Free GPU memory before next model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[{task.upper()} training done — log at {log_path}]")


def main():
    parser = argparse.ArgumentParser(description="Train separate single-task models")
    parser.add_argument("--task", choices=["cls", "bio", "rel", "all"], default="all",
                        help="Which model to train (default: all)")
    parser.add_argument("--device", default=None, help="Device override (cuda/cpu)")
    args = parser.parse_args()

    if args.device:
        global DEVICE
        DEVICE = torch.device(args.device)

    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")

    train_loader, val_loader, cls_w, bio_w, rel_w = load_data()

    tasks = ["cls", "bio", "rel"] if args.task == "all" else [args.task]

    for task in tasks:
        model = train_one(task, train_loader, val_loader, cls_w, bio_w, rel_w)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nDone. Models saved to:", list(MODEL_PATHS.values()))


if __name__ == "__main__":
    main()

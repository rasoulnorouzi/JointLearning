import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import classification_report
from typing import Dict, List, Tuple # Added for type hinting

# Define a type alias for clarity, assuming JointCausalModel is the class
# If not available, we use nn.Module. You might need to import your model.
# from .crf_model import JointCausalModel
# ModelType = JointCausalModel
ModelType = nn.Module # Use generic nn.Module if import is not feasible here


def evaluate_model(
    model: ModelType,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    id2label_cls: dict,
    id2label_bio: dict,
    id2label_rel: dict,
) -> dict:
    """
    Evaluates the multitask model on a given dataset, focusing on metrics.

    This version removes loss calculation, expecting it to be handled
    during the training loop. It uses softmax BIO predictions for metric calculation by:
    1.  Expecting a dictionary output from the model.
    2.  Using `torch.argmax()` on emissions for BIO predictions.

    Args:
        model: The PyTorch model (JointCausalModel) to evaluate.
        dataloader: DataLoader for the validation or test set.
        device: The device (CPU or CUDA) to run evaluation on.
        id2label_cls: Mapping from class ID to class name for classification task.
        id2label_bio: Mapping from BIO ID to BIO tag name for BIO task.
        id2label_rel: Mapping from relation ID to relation type name for relation task.

    Returns:
        A dictionary containing evaluation metrics (F1, Precision, Recall)
        for each task and an overall average F1 score.
    """
    model.eval()  # Set the model to evaluation mode

    # Lists to store predictions and ground truth labels for all batches
    all_cls_preds, all_cls_labels = [], []
    all_bio_preds, all_bio_labels = [], []  # For token-level BIO evaluation
    all_rel_preds, all_rel_labels = [], []

    val_pbar = tqdm(dataloader, desc="Evaluating", leave=False)

    # Disable gradient calculations during evaluation
    with torch.no_grad():
        for batch in val_pbar:
            # Move batch data to the designated device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            cls_labels_gold = batch["cls_labels"].to(device)
            bio_labels_gold = batch["bio_labels"].to(device)

            # Handle optional relation data (might not be present in every batch)
            pair_batch = batch.get("pair_batch")
            cause_starts = batch.get("cause_starts")
            cause_ends = batch.get("cause_ends")
            effect_starts = batch.get("effect_starts")
            effect_ends = batch.get("effect_ends")
            rel_labels_gold = batch.get("rel_labels")

            # Move optional data to device if it exists
            pair_batch = pair_batch.to(device) if pair_batch is not None else None
            cause_starts = cause_starts.to(device) if cause_starts is not None else None
            cause_ends = cause_ends.to(device) if cause_ends is not None else None
            effect_starts = effect_starts.to(device) if effect_starts is not None else None
            effect_ends = effect_ends.to(device) if effect_ends is not None else None
            rel_labels_gold = rel_labels_gold.to(device) if rel_labels_gold is not None else None

            # --- MODIFIED: Handle Dictionary Output ---
            # Perform forward pass and get the output dictionary
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bio_labels=bio_labels_gold,  # Pass labels in case model uses them
                pair_batch=pair_batch,
                cause_starts=cause_starts,
                cause_ends=cause_ends,
                effect_starts=effect_starts,
                effect_ends=effect_ends,
            )

            # Extract logits/emissions from the dictionary
            cls_logits = outputs["cls_logits"]
            bio_emissions = outputs["bio_emissions"]
            rel_logits = outputs["rel_logits"]

            # --- LOSS CALCULATION REMOVED ---

            # --- Accumulate predictions and labels for metrics ---

            # Task 1: Classification
            all_cls_preds.extend(torch.argmax(cls_logits, dim=-1).cpu().numpy())
            all_cls_labels.extend(cls_labels_gold.cpu().numpy())

            # Task 2: BIO Prediction (Token-level BIO)
            # Create the active mask to ignore padding and special tokens (-100)
            active_mask = attention_mask.bool() & (bio_labels_gold != -100)

            # Softmax Prediction: Use argmax on emissions
            bio_preds_batch = torch.argmax(bio_emissions, dim=-1)
            for i in range(bio_labels_gold.shape[0]):
                # Apply active_mask to both predictions and labels
                active_indices = active_mask[i]
                preds_for_seq = bio_preds_batch[i][active_indices].cpu().numpy()
                labels_for_seq = bio_labels_gold[i][active_indices].cpu().numpy()
                all_bio_preds.extend(preds_for_seq)
                all_bio_labels.extend(labels_for_seq)


            # Task 3: Relation Prediction
            if rel_logits is not None and rel_labels_gold is not None and rel_logits.shape[0] > 0:
                all_rel_preds.extend(torch.argmax(rel_logits, dim=-1).cpu().numpy())
                all_rel_labels.extend(rel_labels_gold.cpu().numpy())

    # --- Calculate and store detailed metrics using sklearn ---
    eval_metrics = {}

    # Task 1: Classification Metrics
    if all_cls_labels:
        cls_labels_set = set(all_cls_labels) | set(all_cls_preds)
        cls_target_names = [id2label_cls.get(i, f"CLS_Label_{i}") for i in sorted(list(cls_labels_set))]
        cls_labels_for_report = sorted(list(cls_labels_set))
        cls_report = classification_report(all_cls_labels, all_cls_preds, labels=cls_labels_for_report, target_names=cls_target_names, output_dict=True, zero_division=0)
        eval_metrics["task_cls"] = cls_report
        f1_cls_macro = cls_report.get("macro avg", {}).get("f1-score", 0.0)
    else:
        eval_metrics["task_cls"] = {"message": "No classification labels to evaluate."}
        f1_cls_macro = 0.0

    # Task 2: BIO Prediction Metrics
    if all_bio_labels:
        bio_labels_set = set(all_bio_labels) | set(all_bio_preds)
        bio_labels_for_report = sorted([l for l in list(bio_labels_set) if l != -100])
        bio_target_names = [id2label_bio.get(i, f"BIO_Label_{i}") for i in bio_labels_for_report]
        bio_report = classification_report(all_bio_labels, all_bio_preds, labels=bio_labels_for_report, target_names=bio_target_names, output_dict=True, zero_division=0)
        eval_metrics["task_bio"] = bio_report
        f1_bio_macro = bio_report.get("macro avg", {}).get("f1-score", 0.0)
    else:
        eval_metrics["task_bio"] = {"message": "No BIO labels to evaluate."}
        f1_bio_macro = 0.0

    # Task 3: Relation Prediction Metrics
    if all_rel_labels:
        rel_labels_set = set(all_rel_labels) | set(all_rel_preds)
        rel_labels_for_report = sorted(list(rel_labels_set))
        rel_target_names = [id2label_rel.get(i, f"REL_Label_{i}") for i in rel_labels_for_report]
        rel_report = classification_report(all_rel_labels, all_rel_preds, labels=rel_labels_for_report, target_names=rel_target_names, output_dict=True, zero_division=0)
        eval_metrics["task_relation"] = rel_report
        f1_rel_macro = rel_report.get("macro avg", {}).get("f1-score", 0.0)
    else:
        eval_metrics["task_relation"] = {"message": "No relation labels to evaluate."}
        f1_rel_macro = 0.0

    # Calculate overall average F1
    num_tasks_evaluated = sum([1 for report in [eval_metrics["task_cls"], eval_metrics["task_bio"], eval_metrics["task_relation"]] if "macro avg" in report])

    if num_tasks_evaluated > 0:
        eval_metrics["overall_avg_f1"] = (f1_cls_macro + f1_bio_macro + f1_rel_macro) / num_tasks_evaluated
    else:
        eval_metrics["overall_avg_f1"] = 0.0

    return eval_metrics


def print_eval_report(
    epoch_num: int,
    num_epochs: int,
    avg_train_loss: float,
    avg_val_loss: float, # MODIFIED: Added to accept validation loss
    eval_results: dict,
    best_overall_f1: float,
    patience_counter: int,
    patience_epochs: int,
    new_best_saved: bool,
):
    """
    Prints a visually structured report of the evaluation results.

    MODIFIED: This version removes the detailed per-task loss breakdown,
    as loss is now calculated externally. It accepts and displays the
    overall training and validation loss passed as arguments.
    """
    header_line = "=" * 80
    section_line = "-" * 80

    print(f"\n{header_line}")
    print(f"Epoch {epoch_num}/{num_epochs} Summary")
    print(section_line)
    # MODIFIED: Display losses passed as arguments
    print(f"  Average Training Loss:           {avg_train_loss:.4f}")
    print(f"  Average Validation Loss:         {avg_val_loss:.4f}")
    print(f"  Overall Validation Avg F1 (Macro): {eval_results.get('overall_avg_f1', 0.0):.4f}")
    print(section_line)
    print("Task-Specific Validation Performance:")

    # Task 1: Classification (Reporting remains the same)
    cls_metrics_dict = eval_results.get("task_cls", {})
    print("\n  [Task 1: Sentence Classification]")
    if isinstance(cls_metrics_dict, dict) and "macro avg" in cls_metrics_dict:
        macro_avg_cls = cls_metrics_dict["macro avg"]
        accuracy_cls = cls_metrics_dict.get("accuracy", "N/A")
        accuracy_cls_str = f"{accuracy_cls:.4f}" if isinstance(accuracy_cls, float) else accuracy_cls

        print(f"    Macro F1-Score:  {macro_avg_cls.get('f1-score', 0.0):.4f}")
        print(f"    Macro Precision: {macro_avg_cls.get('precision', 0.0):.4f}")
        print(f"    Macro Recall:    {macro_avg_cls.get('recall', 0.0):.4f}")
        print(f"    Accuracy:        {accuracy_cls_str}")
        print("    Per-class details:")
        for class_name, metrics in cls_metrics_dict.items():
            if class_name not in ["accuracy", "macro avg", "weighted avg"] and isinstance(metrics, dict):
                print(f"      {class_name:<12}: F1={metrics.get('f1-score',0.0):.4f} (P={metrics.get('precision',0.0):.4f}, R={metrics.get('recall',0.0):.4f}, Support={metrics.get('support',0)})")
    else:
        print(f"    Metrics not available or in unexpected format. Message: {cls_metrics_dict.get('message', 'N/A') if isinstance(cls_metrics_dict, dict) else cls_metrics_dict}")

    # Task 2: BIO (Reporting remains the same)
    bio_metrics_dict = eval_results.get("task_bio", {})
    print("\n  [Task 2: BIO Prediction (Token-BIO)]")
    if isinstance(bio_metrics_dict, dict) and "macro avg" in bio_metrics_dict:
        macro_avg_bio = bio_metrics_dict["macro avg"]
        print(f"    Macro F1-Score:  {macro_avg_bio.get('f1-score', 0.0):.4f}")
        print(f"    Macro Precision: {macro_avg_bio.get('precision', 0.0):.4f}")
        print(f"    Macro Recall:    {macro_avg_bio.get('recall', 0.0):.4f}")
        print("    Per-tag details (P=Precision, R=Recall, F1=F1-Score, S=Support):")
        for tag_name, metrics in bio_metrics_dict.items():
            if tag_name not in ["accuracy", "macro avg", "weighted avg"] and isinstance(metrics, dict):
                print(f"      {tag_name:<10}: F1={metrics.get('f1-score',0.0):.3f} (P={metrics.get('precision',0.0):.3f}, R={metrics.get('recall',0.0):.3f}, S={metrics.get('support',0)})")
    else:
        print(f"    Metrics not available or in unexpected format. Message: {bio_metrics_dict.get('message', 'N/A') if isinstance(bio_metrics_dict, dict) else bio_metrics_dict}")

    # Task 3: Relation (Reporting remains the same)
    rel_metrics_dict = eval_results.get("task_relation", {})
    print("\n  [Task 3: Relation Prediction]")
    if isinstance(rel_metrics_dict, dict) and "macro avg" in rel_metrics_dict:
        macro_avg_rel = rel_metrics_dict["macro avg"]
        print(f"    Macro F1-Score:  {macro_avg_rel.get('f1-score', 0.0):.4f}")
        print(f"    Macro Precision: {macro_avg_rel.get('precision', 0.0):.4f}")
        print(f"    Macro Recall:    {macro_avg_rel.get('recall', 0.0):.4f}")
        print("    Per-relation type details (P=Precision, R=Recall, F1=F1-Score, S=Support):")
        for rel_type_name, metrics in rel_metrics_dict.items():
            if rel_type_name not in ["accuracy", "macro avg", "weighted avg"] and isinstance(metrics, dict):
                print(f"      {rel_type_name:<12}: F1={metrics.get('f1-score',0.0):.3f} (P={metrics.get('precision',0.0):.3f}, R={metrics.get('recall',0.0):.3f}, S={metrics.get('support',0)})")
    else:
        print(f"    Metrics not available or in unexpected format. Message: {rel_metrics_dict.get('message', 'N/A') if isinstance(rel_metrics_dict, dict) else rel_metrics_dict}")

    print(section_line)
    if new_best_saved:
        print(f"Status: 🎉 New best model saved! Overall Avg F1: {best_overall_f1:.4f}")
    else:
        print(f"Status: Overall Avg F1 did not improve. Best: {best_overall_f1:.4f}. Patience: {patience_counter}/{patience_epochs}")
    print(f"{header_line}\n")
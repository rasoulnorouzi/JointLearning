import torch
from tqdm import tqdm
from sklearn.metrics import classification_report

def evaluate_model(model,
                   dataloader,
                   loss_fns: tuple, # (cls_loss_fn, bio_loss_fn, rel_loss_fn)
                   device,
                   id2label_cls: dict,
                   id2label_bio: dict,
                   id2label_rel: dict) -> dict:
    """
    Evaluates the multitask model on a given dataset.

    Args:
        model: The PyTorch model to evaluate.
        dataloader: DataLoader for the validation or test set.
        loss_fns: A tuple containing the loss functions for CLS, BIO, and REL tasks.
        device: The device (CPU or CUDA) to run evaluation on.
        id2label_cls: Mapping from class ID to class name for classification task.
        id2label_bio: Mapping from BIO ID to BIO tag name for BIO task.
        id2label_rel: Mapping from relation ID to relation type name for relation task.

    Returns:
        A dictionary containing evaluation metrics for each task and overall scores.
    """
    model.eval()  # Set the model to evaluation mode
    total_val_loss = 0.0
    total_cls_loss = 0.0
    total_bio_loss = 0.0
    total_rel_loss = 0.0

    all_cls_preds, all_cls_labels = [], []
    all_bio_preds, all_bio_labels = [], [] # For token-level BIO evaluation
    all_rel_preds, all_rel_labels = [], []

    cls_loss_fn, bio_loss_fn, rel_loss_fn = loss_fns
    
    val_pbar = tqdm(dataloader, desc="Evaluating", leave=False)
    
    with torch.no_grad():  # Disable gradient calculations
        for batch_idx, batch in enumerate(val_pbar):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            cls_labels_gold = batch['cls_labels'].to(device)
            bio_labels_gold = batch['bio_labels'].to(device) # (batch_size, seq_len)

            pair_batch_data = batch['pair_batch'].to(device) if batch['pair_batch'] is not None else None
            cause_starts_data = batch['cause_starts'].to(device) if batch['cause_starts'] is not None else None
            cause_ends_data = batch['cause_ends'].to(device) if batch['cause_ends'] is not None else None
            effect_starts_data = batch['effect_starts'].to(device) if batch['effect_starts'] is not None else None
            effect_ends_data = batch['effect_ends'].to(device) if batch['effect_ends'] is not None else None
            rel_labels_gold = batch['rel_labels'].to(device) if batch['rel_labels'] is not None else None

            # Forward pass
            cls_logits, bio_logits, rel_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pair_batch=pair_batch_data,
                cause_starts=cause_starts_data,
                cause_ends=cause_ends_data,
                effect_starts=effect_starts_data,
                effect_ends=effect_ends_data
            )

            # --- Calculate and accumulate loss ---
            loss_cls = cls_loss_fn(cls_logits, cls_labels_gold)
            # Reshape for CrossEntropyLoss: (N, C) for logits, (N) for labels
            loss_bio = bio_loss_fn(bio_logits.view(-1, model.num_bio_labels), bio_labels_gold.view(-1))
            
            loss_rel = torch.tensor(0.0).to(device)
            if rel_logits is not None and rel_labels_gold is not None and rel_logits.shape[0] > 0:
                loss_rel = rel_loss_fn(rel_logits, rel_labels_gold)
            
            # Track individual losses
            total_cls_loss += loss_cls.item()
            total_bio_loss += loss_bio.item()
            total_rel_loss += loss_rel.item()
            
            # Combine losses
            current_batch_loss = loss_cls + loss_bio + loss_rel
            total_val_loss += current_batch_loss.item()
            
            # Update progress bar with current loss values
            if batch_idx > 0:
                val_pbar.set_postfix(
                    total=f"{total_val_loss / (batch_idx + 1):.4f}", 
                    cls=f"{total_cls_loss / (batch_idx + 1):.4f}", 
                    bio=f"{total_bio_loss / (batch_idx + 1):.4f}", 
                    rel=f"{total_rel_loss / (batch_idx + 1):.4f}"
                )

            # --- Accumulate predictions and labels for metrics ---
            # Task 1: Classification
            all_cls_preds.extend(torch.argmax(cls_logits, dim=-1).cpu().numpy())
            all_cls_labels.extend(cls_labels_gold.cpu().numpy())

            # Task 2: BIO Prediction (Token-level BIO)
            bio_preds_batch = torch.argmax(bio_logits, dim=-1) # (batch_size, seq_len)
            for i in range(bio_labels_gold.shape[0]): # Iterate over each sequence in the batch
                # Only evaluate on non атмосфера(-100) labels
                active_indices = (bio_labels_gold[i] != -100)
                preds_for_seq = bio_preds_batch[i][active_indices].cpu().numpy()
                labels_for_seq = bio_labels_gold[i][active_indices].cpu().numpy()
                all_bio_preds.extend(preds_for_seq)
                all_bio_labels.extend(labels_for_seq)

            # Task 3: Relation Prediction
            if rel_logits is not None and rel_labels_gold is not None and rel_logits.shape[0] > 0:
                all_rel_preds.extend(torch.argmax(rel_logits, dim=-1).cpu().numpy())
                all_rel_labels.extend(rel_labels_gold.cpu().numpy())    # Calculate average losses
    avg_val_loss = total_val_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    avg_cls_loss = total_cls_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    avg_bio_loss = total_bio_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    avg_rel_loss = total_rel_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    
    # Store all losses in metrics dictionary
    eval_metrics = {
        "loss": avg_val_loss,
        "cls_loss": avg_cls_loss,
        "bio_loss": avg_bio_loss,
        "rel_loss": avg_rel_loss
    }

    # --- Calculate and store detailed metrics using sklearn.metrics.classification_report ---
    # Task 1: Classification Metrics
    if all_cls_labels: # Ensure there's something to evaluate
        cls_target_names = [id2label_cls.get(i, f"CLS_Label_{i}") for i in sorted(list(set(all_cls_labels)))]
        cls_labels_for_report = sorted(list(set(all_cls_labels)))
        cls_report = classification_report(all_cls_labels, all_cls_preds, labels=cls_labels_for_report, target_names=cls_target_names, output_dict=True, zero_division=0)
        eval_metrics["task_cls"] = cls_report
        f1_cls_macro = cls_report.get("macro avg", {}).get("f1-score", 0.0)
    else:
        eval_metrics["task_cls"] = {"message": "No classification labels to evaluate."}
        f1_cls_macro = 0.0

    # Task 2: BIO Prediction Metrics (Token-level BIO)
    if all_bio_labels: # Ensure there's something to evaluate
        # Get unique labels present in the gold data for spans, excluding any potential -100
        bio_labels_for_report = sorted([l for l in list(set(all_bio_labels)) if l != -100])
        bio_target_names = [id2label_bio.get(i, f"BIO_Label_{i}") for i in bio_labels_for_report]
        bio_report = classification_report(all_bio_labels, all_bio_preds, labels=bio_labels_for_report, target_names=bio_target_names, output_dict=True, zero_division=0)
        eval_metrics["task_bio"] = bio_report
        f1_bio_macro = bio_report.get("macro avg", {}).get("f1-score", 0.0)
    else:
        eval_metrics["task_bio"] = {"message": "No BIO labels to evaluate."}
        f1_bio_macro = 0.0
        
    # Task 3: Relation Prediction Metrics
    if all_rel_labels: # Ensure there's something to evaluate
        rel_labels_for_report = sorted(list(set(all_rel_labels)))
        rel_target_names = [id2label_rel.get(i, f"REL_Label_{i}") for i in rel_labels_for_report]
        rel_report = classification_report(all_rel_labels, all_rel_preds, labels=rel_labels_for_report, target_names=rel_target_names, output_dict=True, zero_division=0)
        eval_metrics["task_relation"] = rel_report
        f1_rel_macro = rel_report.get("macro avg", {}).get("f1-score", 0.0)
    else:
        eval_metrics["task_relation"] = {"message": "No relation labels to evaluate."}
        f1_rel_macro = 0.0

    # Calculate overall average F1 (simple average of macro F1s)
    num_tasks_evaluated = 0
    if all_cls_labels: num_tasks_evaluated += 1
    if all_bio_labels: num_tasks_evaluated += 1
    if all_rel_labels: num_tasks_evaluated += 1
    
    if num_tasks_evaluated > 0:
        eval_metrics["overall_avg_f1"] = (f1_cls_macro + f1_bio_macro + f1_rel_macro) / num_tasks_evaluated
    else:
        eval_metrics["overall_avg_f1"] = 0.0

    return eval_metrics

def print_eval_report(epoch_num: int,
                      num_epochs: int,
                      avg_train_loss: float,
                      eval_results: dict,
                      best_overall_f1: float,
                      patience_counter: int,
                      patience_epochs: int,
                      new_best_saved: bool):
    """
    Prints a visually structured report of the evaluation results,
    including per-tag details for BIO and relation tasks.
    """
    header_line = "=" * 80  # Adjusted width for potentially more details
    section_line = "-" * 80

    print(f"\n{header_line}")
    print(f"Epoch {epoch_num}/{num_epochs} Summary")
    print(section_line)
    print(f"  Average Training Loss:           {avg_train_loss:.4f}")
    print(f"  Validation Loss (Total):         {eval_results.get('loss', float('nan')):.4f}")
    print(f"    - Classification Loss:         {eval_results.get('cls_loss', float('nan')):.4f}")
    print(f"    - BIO Loss:                    {eval_results.get('bio_loss', float('nan')):.4f}")
    print(f"    - Relation Loss:               {eval_results.get('rel_loss', float('nan')):.4f}")
    print(f"  Overall Validation Avg F1 (Macro): {eval_results.get('overall_avg_f1', 0.0):.4f}")
    print(section_line)
    print("Task-Specific Validation Performance:")

    # Task 1: Classification
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

    # Task 2: BIO (Token-BIO)
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
                # Adjust column width if necessary, e.g., tag_name:<8
                print(f"      {tag_name:<10}: F1={metrics.get('f1-score',0.0):.3f} (P={metrics.get('precision',0.0):.3f}, R={metrics.get('recall',0.0):.3f}, S={metrics.get('support',0)})")
    else:
        print(f"    Metrics not available or in unexpected format. Message: {bio_metrics_dict.get('message', 'N/A') if isinstance(bio_metrics_dict, dict) else bio_metrics_dict}")

    # Task 3: Relation
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
                # Adjust column width if necessary, e.g., rel_type_name:<15
                print(f"      {rel_type_name:<12}: F1={metrics.get('f1-score',0.0):.3f} (P={metrics.get('precision',0.0):.3f}, R={metrics.get('recall',0.0):.3f}, S={metrics.get('support',0)})")
    else:
        print(f"    Metrics not available or in unexpected format. Message: {rel_metrics_dict.get('message', 'N/A') if isinstance(rel_metrics_dict, dict) else rel_metrics_dict}")

    print(section_line)
    if new_best_saved:
        print(f"Status: 🎉 New best model saved! Overall Avg F1: {best_overall_f1:.4f}")
    else:
        print(f"Status: Overall Avg F1 did not improve. Best: {best_overall_f1:.4f}. Patience: {patience_counter}/{patience_epochs}")
    print(f"{header_line}\n")
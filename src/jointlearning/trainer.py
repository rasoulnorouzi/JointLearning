# %%
import torch
import torch.nn as nn
import torch.nn.functional as F # Added back for F.softmax
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import os
import copy # For deep copying the best model state
import json # Added back for json.dump
from model import JointCausalModel # Assuming this is your model class
from config import id2label_cls, id2label_bio, id2label_rel
from evaluate_joint_causal_model import evaluate_model, print_eval_report
from loss import GCELoss # Import GCELoss from loss.py


# %%

# Define a type alias for clarity
ModelType = nn.Module # Or specifically JointCausalModel if imported

def train_model(
    model: ModelType,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    id2label_cls: dict,
    id2label_bio: dict,
    id2label_rel: dict,
    model_save_path: str = "best_model.pt",
    scheduler: optim.lr_scheduler._LRScheduler = None, # type: ignore
    cls_class_weights: torch.Tensor = None,
    bio_class_weights: torch.Tensor = None, 
    rel_class_weights: torch.Tensor = None,
    patience_epochs: int = 5, 
    seed: int = 8642,
    max_grad_norm: float | None = 1.0, 
    eval_fn_metrics=None, 
    print_report_fn=None,
    is_silver_training: bool = False, # If True, uses GCE loss for silver (pseudo-annotated) data
    gce_q_value: float = 0.7, # q value for GCE loss if used
    task_loss_weights: dict = None # Weights for cls, bio, rel losses e.g. {"cls": 1.0, "bio": 4.0, "rel": 1.0}
) -> tuple[ModelType, dict]:
    """
    Trains the Joint Causal Model, supporting both standard CrossEntropy and GCE loss,
    class imbalance weighting, task-specific loss weighting, gradient clipping,
    learning rate scheduling, and early stopping based on validation F1-score.

    Args:
        model (ModelType): The PyTorch model to be trained.
        train_dataloader (DataLoader): DataLoader for the training dataset.
        val_dataloader (DataLoader): DataLoader for the validation dataset.
        optimizer (optim.Optimizer): The optimization algorithm.
        num_epochs (int): Total number of epochs for training.
        device (torch.device): The device (CPU or CUDA) to train on.
        id2label_cls (dict): Mapping from ID to label for the classification task.
        id2label_bio (dict): Mapping from ID to label for the BIO task.
        id2label_rel (dict): Mapping from ID to label for the relation task.
        model_save_path (str, optional): Path to save the best model weights.
                                         Defaults to "best_model.pt".
        scheduler (optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
                                                               Defaults to None.
        cls_class_weights (torch.Tensor, optional): Class weights for the CLS loss.
                                                    Defaults to None.
        bio_class_weights (torch.Tensor, optional): Class weights for the BIO loss.
                                                    Defaults to None.
        rel_class_weights (torch.Tensor, optional): Class weights for the REL loss.
                                                    Defaults to None.
        patience_epochs (int, optional): Number of epochs with no F1 improvement
                                         before early stopping. Defaults to 5.
        seed (int, optional): Random seed for reproducibility. Defaults to 8642.
        max_grad_norm (float | None, optional): Maximum norm for gradient clipping.
                                                Set to None to disable. Defaults to 1.0.
        eval_fn_metrics (callable, optional): Function to compute evaluation metrics.
                                              Must be provided.
        print_report_fn (callable, optional): Function to print the epoch summary report.
                                              Must be provided.
        is_silver_training (bool, optional): If True, switches to GCE loss, intended
                                             for training on noisy/pseudo-annotated data.
                                             Defaults to False (uses CrossEntropy).
        gce_q_value (float, optional): The 'q' hyperparameter for GCE loss if used.
                                       Defaults to 0.7.
        task_loss_weights (dict, optional): A dictionary specifying weights for each task's
                                            loss contribution, e.g.,
                                            {"cls": 1.0, "bio": 4.0, "rel": 1.0}.
                                            If None, defaults to 1.0 for all tasks.

    Raises:
        ValueError: If eval_fn_metrics or print_report_fn are not provided, or
                    if an unsupported loss_type is derived.

    Returns:
        tuple[ModelType, dict]: A tuple containing:
            - The best trained model (loaded from the saved state or the best in-memory state).
            - A dictionary containing the history of training/validation losses and F1 scores.
    """

    # --- Pre-flight Checks & Setup ---
    if eval_fn_metrics is None or print_report_fn is None:
        raise ValueError("eval_fn_metrics and print_report_fn must be provided.")

    # Determine loss type based on the training mode flag
    actual_loss_type = "gce" if is_silver_training else "cross_entropy"
    print(f"--- Training Configuration ---")
    print(f"Device: {device}")
    print(f"Number of Epochs: {num_epochs}")
    print(f"Seed: {seed}")
    optimizer_params = optimizer.defaults
    print(f"Optimizer: {optimizer.__class__.__name__} (LR: {optimizer_params.get('lr', 'N/A')}, Weight Decay: {optimizer_params.get('weight_decay', 'N/A')})")
    if scheduler:
        print(f"Scheduler: {scheduler.__class__.__name__}")
    else:
        print(f"Scheduler: None")
    print(f"Gradient Clipping: {'Enabled' if max_grad_norm is not None else 'Disabled'} (Max Norm: {max_grad_norm if max_grad_norm is not None else 'N/A'})")
    print(f"Early Stopping Patience: {patience_epochs if patience_epochs > 0 else 'Disabled'}")
    print(f"Model Save Path: {model_save_path if model_save_path else 'Not saving model to disk'}")
    print(f"Mode: {'Silver Data Training (GCE)' if is_silver_training else 'Standard Training (CrossEntropy)'}")
    if actual_loss_type == "gce":
        print(f"GCE q value: {gce_q_value}")

    # Determine task loss weights, setting defaults if None or incomplete
    _task_loss_weights = {}
    default_weights = {"cls": 1.0, "bio": 1.0, "rel": 1.0}
    if task_loss_weights is None:
        _task_loss_weights = default_weights
        print(f"Task loss weights not provided, using default: {_task_loss_weights}")
    else:
        _task_loss_weights["cls"] = task_loss_weights.get("cls", default_weights["cls"])
        _task_loss_weights["bio"] = task_loss_weights.get("bio", default_weights["bio"])
        _task_loss_weights["rel"] = task_loss_weights.get("rel", default_weights["rel"])
        if len(task_loss_weights) != 3 or not all(k in task_loss_weights for k in default_weights):
             print(f"Warning: Provided task_loss_weights may be incomplete. Using defaults for missing keys.")
        print(f"Using task loss weights: {_task_loss_weights}")
    print(f"CLS Class Weights: {'Provided' if cls_class_weights is not None else 'None'}")
    print(f"BIO Class Weights: {'Provided' if bio_class_weights is not None else 'None'}")
    print(f"REL Class Weights: {'Provided' if rel_class_weights is not None else 'None'}")
    print(f"----------------------------")

    # 1. Reproducibility: Set random seeds for all relevant libraries
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # 2. Device Setup: Move model and class weights (if any) to the target device
    model.to(device)
    if cls_class_weights is not None: cls_class_weights = cls_class_weights.to(device)
    if bio_class_weights is not None: bio_class_weights = bio_class_weights.to(device)
    if rel_class_weights is not None: rel_class_weights = rel_class_weights.to(device)

    # 3. Define Loss Functions: Choose between GCE or CrossEntropy based on config
    if actual_loss_type == "gce":
        cls_loss_fn = GCELoss(q=gce_q_value, num_classes=model.num_cls_labels, weight=cls_class_weights)
        rel_loss_fn = GCELoss(q=gce_q_value, num_classes=model.num_rel_labels, weight=rel_class_weights)
        bio_loss_fn_softmax = GCELoss(q=gce_q_value, num_classes=model.num_bio_labels, weight=bio_class_weights, ignore_index=-100)
    else: # "cross_entropy"
        cls_loss_fn = nn.CrossEntropyLoss(weight=cls_class_weights)
        rel_loss_fn = nn.CrossEntropyLoss(weight=rel_class_weights)
        bio_loss_fn_softmax = nn.CrossEntropyLoss(weight=bio_class_weights, ignore_index=-100)

    # 4. Initialization for Tracking: Setup variables for early stopping and history
    best_overall_f1 = -1.0  # Track the best validation F1 score achieved
    patience_counter = 0    # Track epochs since last F1 improvement
    history = { # Dictionary to store metrics over epochs
        "train_loss_total": [], "train_loss_cls": [], "train_loss_bio": [], "train_loss_rel": [],
        "val_loss_total": [], "val_loss_cls": [], "val_loss_bio": [], "val_loss_rel": [],
        "val_overall_f1": []
    }
    best_model_state = None # To store the best model's state_dict in memory

    # 5. Training Loop: Iterate through each epoch
    for epoch_num in range(1, num_epochs + 1):
        
        # --- Training Phase ---
        model.train() # Set model to training mode
        epoch_train_loss_total, epoch_train_loss_cls, epoch_train_loss_bio, epoch_train_loss_rel = 0,0,0,0
        
        # Use tqdm for a progress bar over the training dataloader
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch_num}/{num_epochs} [Training]", leave=False)
        for batch_idx, batch in enumerate(train_pbar):
            # Move all batch data to the target device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            cls_labels_gold = batch["cls_labels"].to(device)
            bio_labels_gold = batch["bio_labels"].to(device)
            pair_batch = batch.get("pair_batch", None)
            cause_starts, cause_ends = batch.get("cause_starts", None), batch.get("cause_ends", None)
            effect_starts, effect_ends = batch.get("effect_starts", None), batch.get("effect_ends", None)
            rel_labels_gold = batch.get("rel_labels", None)
            if pair_batch is not None: pair_batch = pair_batch.to(device)
            if cause_starts is not None: cause_starts = cause_starts.to(device)
            if cause_ends is not None: cause_ends = cause_ends.to(device)
            if effect_starts is not None: effect_starts = effect_starts.to(device)
            if effect_ends is not None: effect_ends = effect_ends.to(device)
            if rel_labels_gold is not None: rel_labels_gold = rel_labels_gold.to(device)
            
            # Zero out gradients before the forward pass
            optimizer.zero_grad()

            # Forward pass: Get model outputs
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, bio_labels=bio_labels_gold,
                pair_batch=pair_batch, cause_starts=cause_starts, cause_ends=cause_ends,
                effect_starts=effect_starts, effect_ends=effect_ends
            )

            # Calculate individual task losses
            loss_cls = cls_loss_fn(outputs["cls_logits"], cls_labels_gold)
            loss_bio = bio_loss_fn_softmax(outputs["bio_emissions"].view(-1, model.num_bio_labels), bio_labels_gold.view(-1))
            loss_rel = torch.tensor(0.0, device=device)
            if outputs["rel_logits"] is not None and rel_labels_gold is not None and outputs["rel_logits"].shape[0] > 0:
                loss_rel = rel_loss_fn(outputs["rel_logits"], rel_labels_gold)

            # Calculate total loss, applying task weights
            total_loss = (_task_loss_weights["cls"] * loss_cls + 
                          _task_loss_weights["bio"] * loss_bio + 
                          _task_loss_weights["rel"] * loss_rel)
            
            # Check for NaN loss and skip the batch if found
            if torch.isnan(total_loss):
                print(f"Epoch {epoch_num}, Batch {batch_idx}: NaN loss detected! Skipping backward pass.")
                print(f"CLS: {loss_cls.item()}, BIO: {loss_bio.item()}, REL: {loss_rel.item()}")
                continue 

            # Backward pass: Compute gradients
            total_loss.backward()

            # Gradient Clipping: Prevent exploding gradients
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            
            # Optimizer step: Update model weights
            optimizer.step()

            # Accumulate losses for epoch average
            epoch_train_loss_total += total_loss.item()
            epoch_train_loss_cls += loss_cls.item()
            epoch_train_loss_bio += loss_bio.item()
            epoch_train_loss_rel += loss_rel.item()

            # Update progress bar description
            train_pbar.set_postfix(total_loss=f"{total_loss.item():.4f}", cls=f"{loss_cls.item():.4f}", bio=f"{loss_bio.item():.4f}", rel=f"{loss_rel.item():.4f}")
        
        # Calculate average training losses for the epoch
        avg_epoch_train_loss_total = epoch_train_loss_total / len(train_dataloader) if len(train_dataloader) > 0 else 0
        avg_epoch_train_loss_cls = epoch_train_loss_cls / len(train_dataloader) if len(train_dataloader) > 0 else 0
        avg_epoch_train_loss_bio = epoch_train_loss_bio / len(train_dataloader) if len(train_dataloader) > 0 else 0
        avg_epoch_train_loss_rel = epoch_train_loss_rel / len(train_dataloader) if len(train_dataloader) > 0 else 0
        train_pbar.close()

        # --- Validation Phase ---
        model.eval() # Set model to evaluation mode (disables dropout, etc.)
        epoch_val_loss_total, epoch_val_loss_cls, epoch_val_loss_bio, epoch_val_loss_rel = 0,0,0,0
        val_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch_num}/{num_epochs} [Validation]", leave=False)
        with torch.no_grad(): # Disable gradient calculation for validation
            for batch in val_pbar:
                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                cls_labels_gold = batch["cls_labels"].to(device)
                bio_labels_gold = batch["bio_labels"].to(device)
                pair_batch = batch.get("pair_batch", None)
                cause_starts, cause_ends = batch.get("cause_starts", None), batch.get("cause_ends", None)
                effect_starts, effect_ends = batch.get("effect_starts", None), batch.get("effect_ends", None)
                rel_labels_gold = batch.get("rel_labels", None)
                if pair_batch is not None: pair_batch = pair_batch.to(device)
                if cause_starts is not None: cause_starts = cause_starts.to(device)
                if cause_ends is not None: cause_ends = cause_ends.to(device)
                if effect_starts is not None: effect_starts = effect_starts.to(device)
                if effect_ends is not None: effect_ends = effect_ends.to(device)
                if rel_labels_gold is not None: rel_labels_gold = rel_labels_gold.to(device)

                # Forward pass
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, bio_labels=bio_labels_gold,
                    pair_batch=pair_batch, cause_starts=cause_starts, cause_ends=cause_ends,
                    effect_starts=effect_starts, effect_ends=effect_ends
                )
                
                # Calculate validation losses (using same loss functions as training)
                loss_cls_val = cls_loss_fn(outputs["cls_logits"], cls_labels_gold)
                loss_bio_val = bio_loss_fn_softmax(outputs["bio_emissions"].view(-1, model.num_bio_labels), bio_labels_gold.view(-1))
                loss_rel_val = torch.tensor(0.0, device=device)
                if outputs["rel_logits"] is not None and rel_labels_gold is not None and outputs["rel_logits"].shape[0] > 0:
                     if rel_labels_gold.numel() > 0 : 
                        loss_rel_val = rel_loss_fn(outputs["rel_logits"], rel_labels_gold)

                # Calculate total validation loss (typically without task weights, just for monitoring)
                total_loss_val = loss_cls_val + loss_bio_val + loss_rel_val
                
                # Check for NaN and accumulate (avoiding NaNs in sum)
                if torch.isnan(total_loss_val): print(f"Epoch {epoch_num}: NaN validation loss detected.")
                epoch_val_loss_total += total_loss_val.item() if not torch.isnan(total_loss_val) else 0 
                epoch_val_loss_cls += loss_cls_val.item() if not torch.isnan(loss_cls_val) else 0
                epoch_val_loss_bio += loss_bio_val.item() if not torch.isnan(loss_bio_val) else 0
                epoch_val_loss_rel += loss_rel_val.item() if not torch.isnan(loss_rel_val) else 0
                val_pbar.set_postfix(loss=f"{total_loss_val.item() if not torch.isnan(total_loss_val) else float('nan'):.4f}")
        
        # Calculate average validation losses
        avg_epoch_val_loss_total = epoch_val_loss_total / len(val_dataloader) if len(val_dataloader) > 0 else 0
        avg_epoch_val_loss_cls = epoch_val_loss_cls / len(val_dataloader) if len(val_dataloader) > 0 else 0
        avg_epoch_val_loss_bio = epoch_val_loss_bio / len(val_dataloader) if len(val_dataloader) > 0 else 0
        avg_epoch_val_loss_rel = epoch_val_loss_rel / len(val_dataloader) if len(val_dataloader) > 0 else 0
        val_pbar.close()

        # --- Evaluation & Reporting ---
        # Get detailed evaluation metrics using the provided function
        eval_results = eval_fn_metrics(model=model, dataloader=val_dataloader, device=device, id2label_cls=id2label_cls, id2label_bio=id2label_bio, id2label_rel=id2label_rel)
        current_f1 = eval_results.get("overall_avg_f1", 0.0) # Use overall F1 for model selection

        # Store all metrics in the history dictionary
        history["train_loss_total"].append(avg_epoch_train_loss_total); history["train_loss_cls"].append(avg_epoch_train_loss_cls); history["train_loss_bio"].append(avg_epoch_train_loss_bio); history["train_loss_rel"].append(avg_epoch_train_loss_rel)
        history["val_loss_total"].append(avg_epoch_val_loss_total); history["val_loss_cls"].append(avg_epoch_val_loss_cls); history["val_loss_bio"].append(avg_epoch_val_loss_bio); history["val_loss_rel"].append(avg_epoch_val_loss_rel)
        history["val_overall_f1"].append(current_f1)

        # --- Model Saving & Early Stopping ---
        new_best_saved_this_epoch = False
        # Check if the current F1 is better than the best seen so far
        if current_f1 > best_overall_f1:
            best_overall_f1 = current_f1 # Update best F1
            best_model_state = copy.deepcopy(model.state_dict()) # Save model state in memory
            # Save model state to disk if path is provided
            if model_save_path:
                save_dir = os.path.dirname(model_save_path)
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), model_save_path)
            patience_counter = 0 # Reset patience counter
            new_best_saved_this_epoch = True
        else:
            patience_counter += 1 # Increment patience counter

        # Print the epoch summary report
        print_report_fn(epoch_num=epoch_num, num_epochs=num_epochs, avg_train_loss=avg_epoch_train_loss_total, avg_val_loss=avg_epoch_val_loss_total, eval_results=eval_results, best_overall_f1=best_overall_f1, patience_counter=patience_counter, patience_epochs=patience_epochs, new_best_saved=new_best_saved_this_epoch)

        # --- Scheduler Step ---
        # Update learning rate based on scheduler logic
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(current_f1) # Step based on validation F1
            else:
                scheduler.step() # Step based on epoch for other schedulers

        # --- Early Stopping Check ---
        # Stop training if patience limit is reached
        if patience_epochs > 0 and patience_counter >= patience_epochs:
            print(f"Early stopping triggered after {epoch_num} epochs.")
            break
            
    # --- Post-Training ---
    # Load the best model state found during training
    if best_model_state is not None:
        print(f"Loading best model state (in memory) with F1: {best_overall_f1:.4f}")
        model.load_state_dict(best_model_state)
    elif model_save_path and os.path.exists(model_save_path):
        print(f"Loading best model from {model_save_path}")
        model.load_state_dict(torch.load(model_save_path, map_location=device))
    else:
        print("No best model was saved or found. Returning model from last epoch.")

    # Save the training history to a JSON file
    if model_save_path:
        history_save_path = os.path.join(os.path.dirname(model_save_path) or ".", "training_history.json")
        try:
            with open(history_save_path, "w") as f: json.dump(history, f, indent=2)
            print(f"Training history saved to {history_save_path}")
        except Exception as e: print(f"Error saving training history to {history_save_path}: {e}")
    else: print("model_save_path not provided. Training history not saved.")

    # Return the best model and its training history
    return model, history

# Example Usage:
# if __name__ == '__main__':
#     # --- For training on Llama3-annotated (silver) data with GCE and specific task weights ---
#     # silver_task_weights = {"cls": 1.0, "bio": 4.0, "rel": 1.0} # Example weights
#     # trained_model_silver, history_silver = train_model(
#     #     model=my_model_instance,
#     #     train_dataloader=silver_train_loader,
#     #     val_dataloader=expert_val_loader, # Always validate on expert data
#     #     optimizer=my_optimizer,
#     #     num_epochs=20,
#     #     device=DEVICE,
#     #     # ... other necessary params like id2label maps, eval_fn, print_fn ...
#     #     is_silver_training=True, # This will enable GCE
#     #     gce_q_value=0.7,
#     #     task_loss_weights=silver_task_weights,
#     #     cls_class_weights=silver_cls_class_weights, # Class weights for silver data
#     #     bio_class_weights=silver_bio_class_weights,
#     #     rel_class_weights=silver_rel_class_weights
#     # )

#     # --- For fine-tuning on expert-annotated (gold) data with CrossEntropy ---
#     # gold_task_weights = {"cls": 1.0, "bio": 1.0, "rel": 1.0} # Example, could be different
#     # # Re-initialize optimizer for fine-tuning, often with a lower learning rate
#     # finetune_optimizer = torch.optim.AdamW(trained_model_silver.parameters(), lr=2e-5)
#     #
#     # trained_model_gold, history_gold = train_model(
#     #     model=trained_model_silver, # Start from the silver-trained model
#     #     train_dataloader=expert_train_loader,
#     #     val_dataloader=expert_val_loader,
#     #     optimizer=finetune_optimizer,
#     #     num_epochs=10, # Usually fewer epochs for fine-tuning
#     #     device=DEVICE,
#     #     # ... other necessary params ...
#     #     is_silver_training=False, # This will use CrossEntropyLoss
#     #     task_loss_weights=gold_task_weights,
#     #     cls_class_weights=expert_cls_class_weights, # Class weights for expert data
#     #     bio_class_weights=expert_bio_class_weights,
#     #     rel_class_weights=expert_rel_class_weights
#     # )

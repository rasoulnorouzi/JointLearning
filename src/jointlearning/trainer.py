import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import os
import copy # For deep copying the best model state

# Assuming your model and evaluation script are in the same directory or accessible
# Adjust these imports based on your project structure.
# Example:
# from .crf_model import JointCausalModel # If in a separate train.py
# from .evaluate_joint_causal_model import evaluate_model, print_eval_report

# For standalone testing, you might need to define or import these if not already available
# from crf_model import JointCausalModel # Example if files are in the same directory
# from evaluate_joint_causal_model import evaluate_model, print_eval_report


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
    bio_class_weights: torch.Tensor = None, # Only for softmax
    rel_class_weights: torch.Tensor = None,
    patience_epochs: int = 5, # Default patience for early stopping
    seed: int = 8642,
    max_grad_norm: float | None = 1.0, # Max norm for gradient clipping (None to disable)
    eval_fn_metrics=None, # Pass your evaluate_model function here
    print_report_fn=None  # Pass your print_eval_report function here
) -> tuple[ModelType, dict]:
    """
    Trains the JointCausalModel with comprehensive features including
    optional gradient clipping and early stopping.

    Args:
        model: The JointCausalModel instance.
        train_dataloader: DataLoader for the training set.
        val_dataloader: DataLoader for the validation set.
        optimizer: The optimization algorithm.
        num_epochs: Total number of training epochs.
        device: The torch.device (CPU or CUDA).
        id2label_cls: Mapping for classification task labels.
        id2label_bio: Mapping for BIO task labels.
        id2label_rel: Mapping for relation task labels.
        model_save_path (str): Path to save the best performing model.
                               Defaults to "best_model.pt".
        scheduler (Optional): Learning rate scheduler. Defaults to None.
        cls_class_weights (Optional[torch.Tensor]): Class weights for CLS loss.
                                                   Defaults to None.
        bio_class_weights (Optional[torch.Tensor]): Class weights for BIO loss
                                                    (softmax only). Defaults to None.
        rel_class_weights (Optional[torch.Tensor]): Class weights for REL loss.
                                                   Defaults to None.
        patience_epochs (int): Number of epochs to wait for improvement before
                               early stopping. Defaults to 5.
        seed (int): Random seed for reproducibility. Defaults to 42.
        max_grad_norm (Optional[float]): The maximum norm for gradient
                                         clipping. If None, no clipping
                                         is performed. Defaults to 1.0.
        eval_fn_metrics: The function to call for evaluation metrics
                         (e.g., your modified evaluate_model).
        print_report_fn: The function to call for printing epoch reports
                         (e.g., your modified print_eval_report).

    Returns:
        A tuple containing:
            - The best trained model instance (loaded from model_save_path or
              best in-memory state).
            - A history dictionary with training and validation losses and metrics.
    """

    if eval_fn_metrics is None or print_report_fn is None:
        raise ValueError("eval_fn_metrics and print_report_fn must be provided.")

    # 1. Reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # 2. Move model and class weights to device
    model.to(device)
    if cls_class_weights is not None:
        cls_class_weights = cls_class_weights.to(device)
    if bio_class_weights is not None: # Only for softmax
        bio_class_weights = bio_class_weights.to(device)
    if rel_class_weights is not None:
        rel_class_weights = rel_class_weights.to(device)

    # 3. Define Loss Functions
    cls_loss_fn = nn.CrossEntropyLoss(weight=cls_class_weights)
    rel_loss_fn = nn.CrossEntropyLoss(weight=rel_class_weights)
    bio_loss_fn_softmax = None
    # Ensure model has 'use_crf' and 'num_bio_labels' attributes if they are accessed
    if not getattr(model, 'use_crf', False): # Check if CRF is not used
        bio_loss_fn_softmax = nn.CrossEntropyLoss(weight=bio_class_weights, ignore_index=-100)

    # 4. Initialization for Tracking
    best_overall_f1 = -1.0
    patience_counter = 0
    history = {
        "train_loss_total": [], "train_loss_cls": [], "train_loss_bio": [], "train_loss_rel": [],
        "val_loss_total": [], "val_loss_cls": [], "val_loss_bio": [], "val_loss_rel": [],
        "val_overall_f1": []
    }
    best_model_state = None # To store the state_dict of the best model in memory

    # 5. Training Loop
    for epoch_num in range(1, num_epochs + 1):
        # --- Training Phase ---
        model.train()
        epoch_train_loss_total, epoch_train_loss_cls, epoch_train_loss_bio, epoch_train_loss_rel = 0, 0, 0, 0
        
        train_pbar = tqdm(train_dataloader, desc=f"Epoch {epoch_num}/{num_epochs} [Training]", leave=False)
        for batch_idx, batch in enumerate(train_pbar):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            cls_labels_gold = batch["cls_labels"].to(device)
            bio_labels_gold = batch["bio_labels"].to(device)
            
            pair_batch = batch.get("pair_batch")
            cause_starts = batch.get("cause_starts")
            cause_ends = batch.get("cause_ends")
            effect_starts = batch.get("effect_starts")
            effect_ends = batch.get("effect_ends")
            rel_labels_gold = batch.get("rel_labels")

            pair_batch = pair_batch.to(device) if pair_batch is not None else None
            cause_starts = cause_starts.to(device) if cause_starts is not None else None
            cause_ends = cause_ends.to(device) if cause_ends is not None else None
            effect_starts = effect_starts.to(device) if effect_starts is not None else None
            effect_ends = effect_ends.to(device) if effect_ends is not None else None
            rel_labels_gold = rel_labels_gold.to(device) if rel_labels_gold is not None else None
            
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                bio_labels=bio_labels_gold, # Pass gold labels for CRF loss calculation if needed
                pair_batch=pair_batch,
                cause_starts=cause_starts,
                cause_ends=cause_ends,
                effect_starts=effect_starts,
                effect_ends=effect_ends
            )

            # Calculate losses
            loss_cls = cls_loss_fn(outputs["cls_logits"], cls_labels_gold)
            
            loss_bio = torch.tensor(0.0, device=device)
            if getattr(model, 'use_crf', False):
                if outputs["tag_loss"] is not None: # CRF loss is returned directly by the model
                    loss_bio = outputs["tag_loss"]
            elif bio_loss_fn_softmax: # Softmax loss
                loss_bio = bio_loss_fn_softmax(
                    outputs["bio_emissions"].view(-1, getattr(model, 'num_bio_labels', 0)), # Access num_bio_labels safely
                    bio_labels_gold.view(-1)
                )

            loss_rel = torch.tensor(0.0, device=device)
            if outputs["rel_logits"] is not None and rel_labels_gold is not None and outputs["rel_logits"].shape[0] > 0:
                loss_rel = rel_loss_fn(outputs["rel_logits"], rel_labels_gold)

            total_loss = loss_cls + loss_bio + loss_rel
            
            # Check for NaN loss before backward pass
            if torch.isnan(total_loss):
                print(f"Epoch {epoch_num}, Batch {batch_idx}: NaN loss detected! Skipping backward pass for this batch.")
                print(f"Individual losses - CLS: {loss_cls.item()}, BIO: {loss_bio.item()}, REL: {loss_rel.item()}")
                # Optionally, you might want to log more details or even stop training
                continue # Skip to the next batch

            total_loss.backward()

            # Gradient Clipping
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    max_norm=max_grad_norm
                )
            
            optimizer.step()

            # Accumulate losses
            epoch_train_loss_total += total_loss.item()
            epoch_train_loss_cls += loss_cls.item()
            epoch_train_loss_bio += loss_bio.item() if isinstance(loss_bio, torch.Tensor) else loss_bio # Handle if loss_bio is float
            epoch_train_loss_rel += loss_rel.item()

            # Update progress bar
            train_pbar.set_postfix(
                total_loss=f"{total_loss.item():.4f}",
                cls=f"{loss_cls.item():.4f}",
                bio=f"{loss_bio.item() if isinstance(loss_bio, torch.Tensor) else loss_bio:.4f}",
                rel=f"{loss_rel.item():.4f}"
            )
        
        avg_epoch_train_loss_total = epoch_train_loss_total / len(train_dataloader) if len(train_dataloader) > 0 else 0
        avg_epoch_train_loss_cls = epoch_train_loss_cls / len(train_dataloader) if len(train_dataloader) > 0 else 0
        avg_epoch_train_loss_bio = epoch_train_loss_bio / len(train_dataloader) if len(train_dataloader) > 0 else 0
        avg_epoch_train_loss_rel = epoch_train_loss_rel / len(train_dataloader) if len(train_dataloader) > 0 else 0
        train_pbar.close()

        # --- Validation Phase ---
        model.eval()
        epoch_val_loss_total, epoch_val_loss_cls, epoch_val_loss_bio, epoch_val_loss_rel = 0, 0, 0, 0
        
        val_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch_num}/{num_epochs} [Validation]", leave=False)
        with torch.no_grad():
            for batch in val_pbar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                cls_labels_gold = batch["cls_labels"].to(device)
                bio_labels_gold = batch["bio_labels"].to(device)

                pair_batch = batch.get("pair_batch")
                cause_starts = batch.get("cause_starts")
                cause_ends = batch.get("cause_ends")
                effect_starts = batch.get("effect_starts")
                effect_ends = batch.get("effect_ends")
                rel_labels_gold = batch.get("rel_labels")

                pair_batch = pair_batch.to(device) if pair_batch is not None else None
                cause_starts = cause_starts.to(device) if cause_starts is not None else None
                cause_ends = cause_ends.to(device) if cause_ends is not None else None
                effect_starts = effect_starts.to(device) if effect_starts is not None else None
                effect_ends = effect_ends.to(device) if effect_ends is not None else None
                rel_labels_gold = rel_labels_gold.to(device) if rel_labels_gold is not None else None

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bio_labels=bio_labels_gold,
                    pair_batch=pair_batch,
                    cause_starts=cause_starts,
                    cause_ends=cause_ends,
                    effect_starts=effect_starts,
                    effect_ends=effect_ends
                )

                # Calculate validation losses (similar to training)
                loss_cls_val = cls_loss_fn(outputs["cls_logits"], cls_labels_gold)
                
                loss_bio_val = torch.tensor(0.0, device=device)
                if getattr(model, 'use_crf', False):
                    if outputs["tag_loss"] is not None:
                        loss_bio_val = outputs["tag_loss"]
                elif bio_loss_fn_softmax:
                    loss_bio_val = bio_loss_fn_softmax(
                        outputs["bio_emissions"].view(-1, getattr(model, 'num_bio_labels', 0)),
                        bio_labels_gold.view(-1)
                    )

                loss_rel_val = torch.tensor(0.0, device=device)
                if outputs["rel_logits"] is not None and rel_labels_gold is not None and outputs["rel_logits"].shape[0] > 0:
                    loss_rel_val = rel_loss_fn(outputs["rel_logits"], rel_labels_gold)

                total_loss_val = loss_cls_val + loss_bio_val + loss_rel_val
                
                if torch.isnan(total_loss_val): # Check for NaN in validation loss
                    print(f"Epoch {epoch_num}: NaN validation loss detected for a batch. This might affect average validation loss.")
                    # Set to a high number or skip accumulation for this batch if it makes sense
                    # For now, it will contribute to NaN average if not handled
                
                epoch_val_loss_total += total_loss_val.item() if not torch.isnan(total_loss_val) else 0 # Avoid propagating NaN
                epoch_val_loss_cls += loss_cls_val.item() if not torch.isnan(loss_cls_val) else 0
                epoch_val_loss_bio += loss_bio_val.item() if isinstance(loss_bio_val, torch.Tensor) and not torch.isnan(loss_bio_val) else (loss_bio_val if not isinstance(loss_bio_val, torch.Tensor) else 0)
                epoch_val_loss_rel += loss_rel_val.item() if not torch.isnan(loss_rel_val) else 0
                
                val_pbar.set_postfix(
                    total_loss=f"{total_loss_val.item() if not torch.isnan(total_loss_val) else float('nan'):.4f}"
                )
        
        avg_epoch_val_loss_total = epoch_val_loss_total / len(val_dataloader) if len(val_dataloader) > 0 else 0
        avg_epoch_val_loss_cls = epoch_val_loss_cls / len(val_dataloader) if len(val_dataloader) > 0 else 0
        avg_epoch_val_loss_bio = epoch_val_loss_bio / len(val_dataloader) if len(val_dataloader) > 0 else 0
        avg_epoch_val_loss_rel = epoch_val_loss_rel / len(val_dataloader) if len(val_dataloader) > 0 else 0
        val_pbar.close()

        # Get evaluation metrics
        eval_results = eval_fn_metrics(
            model=model,
            dataloader=val_dataloader,
            device=device,
            id2label_cls=id2label_cls,
            id2label_bio=id2label_bio,
            id2label_rel=id2label_rel
        )
        current_f1 = eval_results.get("overall_avg_f1", 0.0)

        # Store history
        history["train_loss_total"].append(avg_epoch_train_loss_total)
        history["train_loss_cls"].append(avg_epoch_train_loss_cls)
        history["train_loss_bio"].append(avg_epoch_train_loss_bio)
        history["train_loss_rel"].append(avg_epoch_train_loss_rel)
        history["val_loss_total"].append(avg_epoch_val_loss_total)
        history["val_loss_cls"].append(avg_epoch_val_loss_cls)
        history["val_loss_bio"].append(avg_epoch_val_loss_bio)
        history["val_loss_rel"].append(avg_epoch_val_loss_rel)
        history["val_overall_f1"].append(current_f1)

        # Print report
        new_best_saved_this_epoch = False
        if current_f1 > best_overall_f1:
            best_overall_f1 = current_f1
            best_model_state = copy.deepcopy(model.state_dict()) # Store best model state in memory
            if model_save_path:
                # Ensure directory exists before saving
                if os.path.dirname(model_save_path): # Check if path includes a directory
                    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                torch.save(model.state_dict(), model_save_path)
            patience_counter = 0
            new_best_saved_this_epoch = True
        else:
            patience_counter += 1
        
        print_report_fn(
            epoch_num=epoch_num,
            num_epochs=num_epochs,
            avg_train_loss=avg_epoch_train_loss_total,
            avg_val_loss=avg_epoch_val_loss_total,
            eval_results=eval_results,
            best_overall_f1=best_overall_f1,
            patience_counter=patience_counter,
            patience_epochs=patience_epochs,
            new_best_saved=new_best_saved_this_epoch
        )

        # Scheduler step
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(current_f1) # Or avg_epoch_val_loss_total
            else:
                scheduler.step()

        # Early stopping
        if patience_epochs > 0 and patience_counter >= patience_epochs: # Check if patience_epochs is positive
            print(f"Early stopping triggered after {epoch_num} epochs.")
            break
            
    # 6. Load Best Model
    # Load from the in-memory state if it exists, otherwise try loading from file
    if best_model_state is not None:
        print(f"Loading best model state (in memory) with F1: {best_overall_f1:.4f}")
        model.load_state_dict(best_model_state)
    elif model_save_path and os.path.exists(model_save_path):
        print(f"Loading best model from {model_save_path}")
        # Ensure map_location is set if loading on a different device than saved
        model.load_state_dict(torch.load(model_save_path, map_location=device))
    else:
        print("No best model was saved or found. Returning the model from the last training epoch.")

    return model, history

# Example of how you might call this (assuming other components are defined):
# if __name__ == '__main__':
    # This is a placeholder for a full runnable example.
    # You would need to:
    # 1. Define/Import JointCausalModel, your evaluate_model, print_eval_report
    # 2. Prepare DataLoaders (train_dataloader, val_dataloader)
    # 3. Define id2label maps
    # 4. Instantiate the model and optimizer
    
    # print("This is a draft training function. See comments for usage.")

    # from crf_model import JointCausalModel # Assuming this is your model class
    # from evaluate_joint_causal_model import evaluate_model as eval_fn_metrics_impl
    # from evaluate_joint_causal_model import print_eval_report as print_report_fn_impl
    # # Dummy Dataloaders and other components would be needed here for a runnable example
    # # ...
# %%
from utility 
import os
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import Subset, DataLoader

from .config import DEVICE, TRAINING_CONFIG

def train_joint_model(
    model, 
    train_dataset,
    val_dataset,
    collator_fn,
    id2label_cls,
    id2label_span,
    id2label_rel,
    evaluate_model_fn,
    print_eval_report_fn,
    device=DEVICE,
    batch_size=TRAINING_CONFIG["batch_size"],
    num_epochs=TRAINING_CONFIG["num_epochs"],
    learning_rate=TRAINING_CONFIG["learning_rate"],
    train_subset_size=None,
    model_save_path=TRAINING_CONFIG["model_save_path"],
    patience_epochs=TRAINING_CONFIG["patience_epochs"],
    weight_decay=TRAINING_CONFIG["weight_decay"],
    gradient_clip_val=TRAINING_CONFIG["gradient_clip_val"],
    apply_gradient_clipping=TRAINING_CONFIG["apply_gradient_clipping"]
):
    """
    Train a joint causal model with flexible configuration options.
    
    Args:
        model: The model to train
        train_dataset: Dataset for training
        val_dataset: Dataset for validation
        collator_fn: Collator function for dataloaders
        id2label_cls: Mapping from class id to label for classification task
        id2label_span: Mapping from class id to label for span detection task
        id2label_rel: Mapping from class id to label for relation task
        evaluate_model_fn: Function to evaluate model performance
        print_eval_report_fn: Function to print evaluation results
        device: Device to use for training ('cuda' or 'cpu')
        batch_size: Batch size for training and validation
        num_epochs: Maximum number of epochs to train
        learning_rate: Learning rate for optimizer
        train_subset_size: If provided, use only this many samples for training
        model_save_path: Path to save the best model
        patience_epochs: Number of epochs to wait before early stopping
        weight_decay: Weight decay for optimizer
        gradient_clip_val: Maximum gradient norm for clipping
        apply_gradient_clipping: Whether to apply gradient clipping
    
    Returns:
        dict: Training statistics including best F1 score and training history
    """
    model.to(device)
    
    # Create subset of training data if specified
    if train_subset_size and train_subset_size < len(train_dataset):
        print(f"Using {train_subset_size} samples out of {len(train_dataset)} for training")
        train_subset_indices = list(range(train_subset_size))
        train_dataset_subset = Subset(train_dataset, train_subset_indices)
    else:
        train_dataset_subset = train_dataset
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator_fn
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Loss functions
    cls_loss_fn = nn.CrossEntropyLoss().to(device)
    span_loss_fn = nn.CrossEntropyLoss(ignore_index=-100).to(device)
    rel_loss_fn = nn.CrossEntropyLoss().to(device)
    training_loss_fns = (cls_loss_fn, span_loss_fn, rel_loss_fn)
    
    # Tracking variables
    best_overall_f1 = 0.0
    patience_counter = 0
    training_history = {
        'train_losses': [],
        'val_losses': [],
        'val_f1_scores': [],
    }
    
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", unit="batch", leave=False)
        
        # Track task-specific losses for running averages
        cls_running_loss = 0.0
        span_running_loss = 0.0
        rel_running_loss = 0.0
        
        for batch_idx, batch in enumerate(train_pbar):
            optimizer.zero_grad()
            
            # Move data to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            cls_labels_gold = batch['cls_labels'].to(device)
            bio_labels_gold = batch['bio_labels'].to(device)
            
            pair_batch_data = batch['pair_batch'].to(device) if batch['pair_batch'] is not None else None
            cause_starts_data = batch['cause_starts'].to(device) if batch['cause_starts'] is not None else None
            cause_ends_data = batch['cause_ends'].to(device) if batch['cause_ends'] is not None else None
            effect_starts_data = batch['effect_starts'].to(device) if batch['effect_starts'] is not None else None
            effect_ends_data = batch['effect_ends'].to(device) if batch['effect_ends'] is not None else None
            rel_labels_gold = batch['rel_labels'].to(device) if batch['rel_labels'] is not None else None
            
            # Forward pass
            cls_logits, span_logits, rel_logits = model(
                input_ids=input_ids, 
                attention_mask=attention_mask,
                pair_batch=pair_batch_data, 
                cause_starts=cause_starts_data, 
                cause_ends=cause_ends_data,
                effect_starts=effect_starts_data, 
                effect_ends=effect_ends_data
            )
            
            # Calculate losses
            loss_cls = cls_loss_fn(cls_logits, cls_labels_gold)
            loss_span = span_loss_fn(span_logits.view(-1, model.num_bio_labels), bio_labels_gold.view(-1))
            loss_rel = torch.tensor(0.0).to(device)
            if rel_logits is not None and rel_labels_gold is not None and rel_logits.shape[0] > 0:
                loss_rel = rel_loss_fn(rel_logits, rel_labels_gold)
            
            total_loss = loss_cls + loss_span + loss_rel
            
            # Update running losses for each task
            cls_running_loss += loss_cls.item()
            span_running_loss += loss_span.item()
            rel_running_loss += loss_rel.item()
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping if enabled
            if apply_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_val)
            
            optimizer.step()
            total_train_loss += total_loss.item()
            
            # Calculate running averages for each task
            avg_cls_loss = cls_running_loss / (batch_idx + 1)
            avg_span_loss = span_running_loss / (batch_idx + 1)
            avg_rel_loss = rel_running_loss / (batch_idx + 1)
            avg_total_loss = total_train_loss / (batch_idx + 1)
            
            # Update progress bar with task-specific losses
            train_pbar.set_postfix(
                total=f"{avg_total_loss:.4f}", 
                cls=f"{avg_cls_loss:.4f}", 
                span=f"{avg_span_loss:.4f}", 
                rel=f"{avg_rel_loss:.4f}", 
                refresh=True
            )
        
        train_pbar.close()  # Close the training progress bar for the epoch
        avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        
        # Validation phase
        eval_results = evaluate_model_fn(
            model, val_loader, training_loss_fns, device,
            id2label_cls, id2label_span, id2label_rel
        )
        
        # Save training history
        training_history['train_losses'].append(avg_train_loss)
        training_history['val_losses'].append(eval_results.get('loss', 0.0))
        training_history['val_f1_scores'].append(eval_results.get('overall_avg_f1', 0.0))
        
        # Check if we have a new best model
        new_best_saved_this_epoch = False
        current_f1 = eval_results.get('overall_avg_f1', 0.0)
        if current_f1 > best_overall_f1:
            best_overall_f1 = current_f1
            save_dir = os.path.dirname(model_save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_overall_f1,
                'training_history': training_history
            }, model_save_path)
            
            new_best_saved_this_epoch = True
            patience_counter = 0  # Reset patience
        else:
            patience_counter += 1
        
        # Print the evaluation report
        print_eval_report_fn(
            epoch + 1, num_epochs, avg_train_loss, eval_results,
            best_overall_f1, patience_counter, patience_epochs,
            new_best_saved_this_epoch
        )
        
        # Early stopping check
        if patience_counter >= patience_epochs:
            print(f"Stopping early after {patience_epochs} epochs without improvement on Overall Avg F1.")
            break
    
    print("Training complete.")
    if os.path.exists(model_save_path):
        print(f"Best model saved at: {model_save_path} with Overall Avg F1: {best_overall_f1:.4f}")
    
    return {
        'best_f1': best_overall_f1,
        'training_history': training_history,
        'model_path': model_save_path if os.path.exists(model_save_path) else None
    }

# Example usage:
# results = train_joint_model(
#     model=model,
#     train_dataset=full_dataset_instance,
#     val_dataset=val_dataset_instance,
#     collator_fn=collator_fn,
#     id2label_cls=id2label_cls, 
#     id2label_span=id2label_span,
#     id2label_rel=id2label_rel,
#     evaluate_model_fn=evaluate_model,
#     print_eval_report_fn=print_eval_report,
#     train_subset_size=200,  # Use only 200 samples
#     num_epochs=3
# )
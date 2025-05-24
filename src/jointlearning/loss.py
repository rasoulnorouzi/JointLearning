import torch
import torch.nn as nn
import torch.nn.functional as F

class GCELoss(nn.Module):
    """
    Generalized Cross Entropy Loss for Robust Learning with Noisy Labels.
    GCE is designed to be more robust to label noise than standard Cross Entropy.
    It works by down-weighting examples with high loss (likely mislabeled)
    based on the 'q' parameter.

    Reference Paper: https://arxiv.org/abs/1805.07836
    """
    def __init__(self, q: float = 0.7, num_classes: int = None, weight: torch.Tensor = None, ignore_index: int = -100, reduction: str = 'mean'):
        """
        Initializes the GCELoss module.

        Args:
            q (float): The 'q' parameter for GCE. Controls robustness.
                       The paper suggests q is in (0, 1]. q = 0.7 is a common starting point.
                       As q -> 0, GCE approaches standard Cross Entropy (handled as a special case).
                       q = 1 results in a Mean Absolute Error-like loss for probabilities (1-p).
            num_classes (int, optional): Number of classes. Useful for context and validation.
            weight (torch.Tensor, optional): A manual rescaling weight given to each class.
                                             If given, must be a Tensor of size C.
                                             Used to handle class imbalance. Defaults to None.
            ignore_index (int, optional): Specifies a target value that is ignored
                                          and does not contribute to the loss or gradient.
                                          Defaults to -100 (standard for PyTorch).
            reduction (str, optional): Specifies the reduction to apply to the output:
                                       'none' | 'mean' | 'sum'. Defaults to 'mean'.
        """
        super(GCELoss, self).__init__()
        # Validate q parameter: GCE is typically defined for 0 < q <= 1.
        # q=0 is handled as a special case (approximates CrossEntropy).
        if q < 0:
            raise ValueError(f"GCE q parameter {q} must be non-negative. For GCE behavior, q should be in (0, 1].")
        if q == 0:
            print(f"Info: GCE q parameter is 0. Loss will behave like Cross Entropy.")
        elif q > 1:
            print(f"Warning: GCE q parameter is {q}. It's typically in (0, 1]. Values > 1 might lead to unexpected behavior.")


        self.q = q
        self.num_classes = num_classes # Store for reference
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.epsilon = 1e-9 # Small constant for numerical stability

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the GCE loss.

        Args:
            logits (torch.Tensor): The predicted logits from the model. Shape (N, C).
            targets (torch.Tensor): The ground truth labels. Shape (N,).

        Returns:
            torch.Tensor: The calculated GCE loss, reduced according to `self.reduction`.
        """
        # Ensure class weights are on the same device as logits
        if self.weight is not None and self.weight.device != logits.device:
            self.weight = self.weight.to(logits.device)

        # Calculate probabilities P(y|x) using softmax
        probs = F.softmax(logits, dim=1)
        
        # Create a mask to identify valid (non-ignored) targets
        valid_mask = (targets != self.ignore_index)
        
        # Initialize loss for all samples to zero (using logits' dtype and device)
        loss_per_sample = torch.zeros_like(targets, dtype=logits.dtype, device=logits.device)

        # Proceed only if there are valid targets to calculate loss for
        if valid_mask.any():
            # Filter targets and probabilities to include only valid ones
            valid_targets = targets[valid_mask]
            valid_probs = probs[valid_mask]

            # Get probabilities corresponding to the true class P(y_true|x)
            prob_of_true_class = valid_probs.gather(1, valid_targets.unsqueeze(1)).squeeze(1)
            
            # Clamp probabilities to avoid log(0) or pow(0, q) with q < 1
            prob_of_true_class_clamped = torch.clamp(prob_of_true_class, min=self.epsilon, max=1.0 - self.epsilon)


            # GCE Loss calculation
            if abs(self.q) < self.epsilon: # Case: q is effectively 0 (CrossEntropy)
                 gce_loss_values = -torch.log(prob_of_true_class_clamped)
            elif abs(self.q - 1.0) < self.epsilon: # Case: q is effectively 1 (MAE-like for probabilities)
                 gce_loss_values = 1.0 - prob_of_true_class_clamped
            else: # Standard GCE formula: (1 - P(y_true|x)^q) / q
                 gce_loss_values = (1. - prob_of_true_class_clamped.pow(self.q)) / self.q
            
            # Apply class weights (for class imbalance) if provided
            if self.weight is not None:
                w = self.weight.gather(0, valid_targets)
                gce_loss_values = gce_loss_values * w
            
            # Store the calculated loss values for the valid samples
            loss_per_sample[valid_mask] = gce_loss_values

        # Apply the specified reduction ('mean', 'sum', or 'none')
        if self.reduction == 'mean':
            # Calculate mean only over the *valid* samples
            if valid_mask.sum() > 0:
                return loss_per_sample.sum() / valid_mask.sum()
            else:
                # If no valid samples, return 0 loss but maintain graph for distributed training
                return logits.sum() * 0.0 
        elif self.reduction == 'sum':
            return loss_per_sample.sum()
        elif self.reduction == 'none':
            return loss_per_sample # Return loss for each sample (with 0 for ignored)
        else:
            raise ValueError(f"Unsupported reduction type: {self.reduction}")

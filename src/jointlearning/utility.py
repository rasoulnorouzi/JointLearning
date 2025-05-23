import numpy as np
import torch
import pandas as pd
from config import DATASET_CONFIG


def compute_class_weights(
    labels_list,
    num_classes: int,
    technique: str = 'inverse_frequency',
    ignore_index: int | None = DATASET_CONFIG.get("ignore_id", -100), # Default to DATASET_CONFIG["ignore_id"], fallback to -100
    beta: float = 0.999, # For ENS
    smoothing_epsilon: float = 1e-9 # To prevent division by zero
):
    """
    Computes class weights for imbalanced datasets based on various techniques.
    These weights can be passed to loss functions like `torch.nn.CrossEntropyLoss`.

    Args:
        labels_list (list, np.ndarray, or torch.Tensor):
            A 1D iterable containing all observed class labels for a specific task
            from the dataset (e.g., [0, 1, 1, 0, 2, ...]).
            For BIO, this should be a flattened list of token labels.
        num_classes (int): The total number of unique classes for this task (e.g., if classes are 0, 1, 2, then num_classes=3).
        technique (str, optional): The weighting technique. Options:
            - 'inverse_count': Weights are 1.0 / (count[c] + epsilon). Less aggressive than inverse_frequency.
            - 'inverse_frequency': Weights are total_valid_labels / (count[c] + epsilon).
                                   This gives significantly higher weights to rarer classes.
            - 'median_frequency': Weights are median_class_frequency / (frequency[c] + epsilon).
                                  Balances based on the median frequency.
            - 'ens': Effective Number of Samples weighting (Cui et al., 2019).
                     Weights are 1.0 / (E_nc + epsilon), where E_nc is the effective number of samples for class c.
            Defaults to 'inverse_frequency'.
        ignore_index (int, optional): A label value to ignore during frequency counting (e.g., -100 for padding/special tokens).
            Defaults to None.
        beta (float, optional): Hyperparameter for the 'ens' technique. Must be in [0, 1).
            Controls how quickly the effective number of samples plateaus. Defaults to 0.999.
        smoothing_epsilon (float, optional): A small epsilon added to denominators
            to prevent division by zero for unobserved or very rare classes.
            Defaults to 1e-9.

    Returns:
        torch.Tensor: A 1D tensor of length `num_classes` containing the calculated weights.
                      Returns a tensor of ones if labels_list is empty or no valid labels found after filtering.

    Examples:
        >>> labels = [0, 1, 1, 0, 2, 1, 1, 0, 0] # Example labels
        >>> num_classes = 3
        >>> weights_inv_freq = compute_class_weights(labels, num_classes, technique='inverse_frequency')
        >>> print(weights_inv_freq)
        tensor([2.2500, 2.2500, 9.0000])

        >>> weights_ens = compute_class_weights(labels, num_classes, technique='ens', beta=0.9)
        >>> print(weights_ens)
        tensor([0.2918, 0.2918, 1.1111])

        >>> bio_labels = [0, 1, 2, 0, 1, 1, -100, 2, 0] # With an ignore_index
        >>> num_bio_classes = 3
        >>> weights_bio = compute_class_weights(bio_labels, num_bio_classes, ignore_index=-100)
        >>> print(weights_bio)
        tensor([2.6667, 2.6667, 4.0000])
    """
    if not isinstance(labels_list, (list, np.ndarray, torch.Tensor)):
        raise TypeError("labels_list must be a list, NumPy array, or PyTorch tensor.")
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError("num_classes must be a positive integer.")

    if isinstance(labels_list, list):
        labels_array = np.array(labels_list, dtype=int)
    elif isinstance(labels_list, torch.Tensor):
        labels_array = labels_list.cpu().numpy().astype(int)
    else: # np.ndarray
        labels_array = labels_list.astype(int)

    # Filter out ignore_index
    if ignore_index is not None:
        labels_array = labels_array[labels_array != ignore_index]

    if labels_array.size == 0:
        # print(f"Warning: No valid labels found (possibly all were ignore_index or list was empty). Returning uniform weights for {num_classes} classes.")
        return torch.ones(num_classes, dtype=torch.float32)

    # Filter out labels that are out of the expected range [0, num_classes-1]
    # This ensures np.bincount works correctly and handles unexpected label values.
    valid_mask = (labels_array >= 0) & (labels_array < num_classes)
    # if not np.all(valid_mask): # More thorough check if all original labels were supposed to be in range
    #     out_of_range_labels = labels_array[~valid_mask]
    #     print(f"Warning: Labels found outside the range [0, {num_classes-1}]: {np.unique(out_of_range_labels)}. These will be ignored for weight calculation.")
    labels_array = labels_array[valid_mask]
    
    if labels_array.size == 0:
        # print(f"Warning: No valid labels found after filtering out-of-range ones. Returning uniform weights for {num_classes} classes.")
        return torch.ones(num_classes, dtype=torch.float32)

    # Calculate class counts
    counts = np.bincount(labels_array, minlength=num_classes)
    
    weights = np.ones(num_classes, dtype=np.float32) # Default for safety

    if technique == 'inverse_count':
        # Weight is inversely proportional to the raw count of the class.
        weights = 1.0 / (counts + smoothing_epsilon)

    elif technique == 'inverse_frequency':
        total_valid_labels = np.sum(counts)
        # Weight is total samples / count for class c. Rarer classes get higher weights.
        for c in range(num_classes):
            weights[c] = total_valid_labels / (counts[c] + smoothing_epsilon)

    elif technique == 'median_frequency':
        total_valid_labels = np.sum(counts)
        if total_valid_labels > 0:
            # Calculate frequencies only for classes that actually appear
            observed_class_indices = np.where(counts > 0)[0]
            if len(observed_class_indices) > 0:
                class_frequencies = counts[observed_class_indices] / total_valid_labels
                median_freq = np.median(class_frequencies)
                
                for c in range(num_classes):
                    if counts[c] > 0:
                        weights[c] = median_freq / ((counts[c] / total_valid_labels) + smoothing_epsilon)
                    else:
                        # For unobserved classes, assign a high weight (median_freq / epsilon)
                        # This represents extreme rarity.
                        weights[c] = median_freq / smoothing_epsilon
            # else: all counts are 0, already handled by labels_array.size == 0 check
        # else: all counts are 0, already handled

    elif technique == 'ens': # Effective Number of Samples
        if not (0 <= beta < 1):
            raise ValueError("beta for ENS must be in [0, 1).")
        
        for c in range(num_classes):
            n_c = counts[c]
            # Effective number E_nc = (1 - beta^n_c) / (1-beta)
            # If n_c = 0, E_nc = 0. Weight = 1 / (E_nc + epsilon) -> high weight
            # If beta = 0, E_nc = 1 (for n_c > 0), or 0 (for n_c = 0)
            if beta == 0.0: # Avoids 0/0 if n_c is also 0, though beta**0 = 1
                effective_num_c = 1.0 if n_c > 0 else 0.0
            else:
                effective_num_c = (1.0 - beta**n_c) / (1.0 - beta)
            weights[c] = 1.0 / (effective_num_c + smoothing_epsilon)
    else:
        raise ValueError(f"Unknown weighting technique: {technique}. "
                         "Options: 'inverse_count', 'inverse_frequency', 'median_frequency', 'ens'.")

    # PyTorch CrossEntropyLoss weights do not need to be normalized to sum to 1 or N.
    # The loss is internally computed as: loss(x, class) = weight[class] * (-x[class] + log(sum_j exp(x_j)))
    # So, the absolute scale of weights matters.
    
    return torch.tensor(weights, dtype=torch.float32)




def label_value_counts(dataset_instance):
    """
    Computes the value counts for 'cls_label', 'bio_labels', 
    and 'relation_tuples' from the given dataset instance.

    Args:
        dataset_instance: An instance of a dataset (e.g., CausalDataset) 
                          that contains 'cls_label', 'bio_labels', and 
                          'relation_tuples' keys in its items.

    Returns:
        tuple: A tuple containing three lists:
            - cls_labels_flat (list): A flat list of all cls_labels.
            - bio_labels_flat (list): A flat list of all bio_labels.
            - rel_labels_flat (list): A flat list of all relation labels.

    Examples:
        >>> # Assuming CausalDataset is defined and dataset_instance is an instance of it
        >>> # For demonstration, let's mock a dataset_instance:
        >>> class MockDataset:
        ...     def __init__(self, data):
        ...         self.data = data
        ...     def __len__(self):
        ...         return len(self.data)
        ...     def __getitem__(self, idx):
        ...         return self.data[idx]
        >>> mock_data = [
        ...     {'cls_label': 0, 'bio_labels': ['O', 'B-C', 'I-C'], 'relation_tuples': [('entity1', 'entity2', 'causes')]},
        ...     {'cls_label': 1, 'bio_labels': ['O', 'B-E', 'I-E'], 'relation_tuples': [('entity3', 'entity4', 'mitigates')]},
        ...     {'cls_label': 0, 'bio_labels': ['O', 'O'], 'relation_tuples': []}
        ... ]
        >>> dataset_instance = MockDataset(mock_data)
        >>> cls_flat, bio_flat, rel_flat = label_value_counts(dataset_instance)
        cls_labels_value_counts:
         0    2
        1    1
        Name: count, dtype: int64
        bio_labels_value_counts:
         O      4
        B-C    1
        I-C    1
        B-E    1
        I-E    1
        Name: count, dtype: int64
        rel_labels_value_counts:
         causes       1
        mitigates    1
        Name: count, dtype: int64
        >>> print(cls_flat)
        [0, 1, 0]
        >>> print(bio_flat)
        ['O', 'B-C', 'I-C', 'O', 'B-E', 'I-E', 'O', 'O']
        >>> print(rel_flat)
        ['causes', 'mitigates']
    """
    # --- Classification (cls) Labels ---
    # Initialize an empty list to store all cls_labels
    cls_labels_flat = []
    # Iterate over each item in the dataset
    for i in range(len(dataset_instance)):
        # Append the 'cls_label' of the current item to the list
        cls_labels_flat.append(dataset_instance[i]['cls_label'])
    # Convert the list of cls_labels to a pandas Series for easy value counting
    cls_labels_flat_series = pd.Series(cls_labels_flat)
    # Compute the value counts for each class
    cls_labels_value_counts = cls_labels_flat_series.value_counts()
    # Print the value counts for cls_labels
    print(f"cls_labels_value_counts:\n {cls_labels_value_counts}")

    # --- BIO (span) Labels ---
    # Initialize an empty list to store all bio_labels
    bio_labels_flat = []
    # Iterate over each item in the dataset
    for i in range(len(dataset_instance)):
        # Extend the list with the 'bio_labels' of the current item
        # 'extend' is used because 'bio_labels' is a list itself
        bio_labels_flat.extend(dataset_instance[i]['bio_labels'])

    # Convert the list of bio_labels to a pandas Series
    bio_labels_flat_series = pd.Series(bio_labels_flat)
    # Compute the value counts for each BIO tag
    bio_labels_value_counts = bio_labels_flat_series.value_counts()
    # Print the value counts for bio_labels
    print(f"bio_labels_value_counts:\n {bio_labels_value_counts}")

    # --- Relation (rel) Labels ---
    # Initialize an empty list to store all relation labels
    rel_labels_flat = []
    # Iterate over each item in the dataset
    for i in range(len(dataset_instance)):
        # Check if 'relation_tuples' exist and is not empty for the current item
        if dataset_instance[i]['relation_tuples']:
            relation_tuples = dataset_instance[i]['relation_tuples']
            # Iterate over each tuple in 'relation_tuples'
            for tp in relation_tuples:
                # The relation label is the third element (index 2) in the tuple
                rel_labels_flat.append(tp[2])
    # Convert the list of relation labels to a pandas Series
    rel_labels_flat_series = pd.Series(rel_labels_flat)
    # Compute the value counts for each relation type
    rel_labels_value_counts = rel_labels_flat_series.value_counts()
    # Print the value counts for rel_labels
    print(f"rel_labels_value_counts:\n {rel_labels_value_counts}")

    # Return the flat lists of cls_labels, bio_labels, and relation labels
    return {
        "cls_labels_flat": cls_labels_flat,
        "bio_labels_flat": bio_labels_flat,
        "rel_labels_flat": rel_labels_flat
    }
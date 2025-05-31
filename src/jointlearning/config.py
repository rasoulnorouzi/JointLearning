"""
Configuration settings for the joint causal learning model.
"""

import torch

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed for reproducibility
SEED = 8642

# Model configuration
MODEL_CONFIG = {
    "encoder_name": "bert-base-uncased",  # Default encoder
    "num_cls_labels": 2,                 # Binary classification for causal/non-causal
    "num_bio_labels": 7,                 # BIO labels for span detection
    "num_rel_labels": 2,                 # Relation labels (updated from 3 to 2)
    "dropout": 0.2,                      # Dropout rate
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 16,
    "num_epochs": 10,
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "gradient_clip_val": None,
    "apply_gradient_clipping": False,
    "patience_epochs": 10,
    "model_save_path": "best_joint_causal_model.pt",
}

# Dataset configuration
DATASET_CONFIG = {
    "max_length": 512,                    # Maximum sequence length for tokenization
    "negative_relation_rate": 2.0,        # Rate of negative relation samples to generate
    "max_random_span_len": 5,            # Maximum length for random negative spans
    "ignore_id": -100,                   # ID to ignore in loss computation
}

# Label mappings
# BIO labels for span detection
id2label_bio = {
    0: "B-C",    # Beginning of Cause
    1: "I-C",    # Inside of Cause
    2: "B-E",    # Beginning of Effect
    3: "I-E",    # Inside of Effect
    4: "B-CE",   # Beginning of Cause-Effect
    5: "I-CE",   # Inside of Cause-Effect
    6: "O"       # Outside
}
label2id_bio = {v: k for k, v in id2label_bio.items()}

# Entity label to BIO prefix mapping
entity_label_to_bio_prefix = {
    "cause": "C",
    "effect": "E",
    "internal_CE": "CE",
    "non-causal": "O"
}

# Relation labels
id2label_rel = {
    0: "Rel_None",
    1: "Rel_CE"
}
label2id_rel = {
    "Rel_None": 0,
    "Rel_CE": 1
}

# Classification labels
id2label_cls = {
    0: "non-causal",
    1: "causal"
}
label2id_cls = {v: k for k, v in id2label_cls.items()}

# Relation type mappings
POSITIVE_RELATION_TYPE_TO_ID = {
    "Rel_CE": 1
}
NEGATIVE_SAMPLE_REL_ID = label2id_rel["Rel_None"]

# Inference configuration
INFERENCE_CONFIG = {
    "cls_threshold": 0.5,  # Threshold for causal/non-causal classification
}
"""
Shared paths and constants for the comparison learning ablation study.
"""
import os
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

# Data
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, "datasets", "expert_multi_task_data", "train.csv")
VAL_DATA_PATH = os.path.join(PROJECT_ROOT, "datasets", "expert_multi_task_data", "val.csv")
TEST_DATA_PATH = os.path.join(PROJECT_ROOT, "datasets", "expert_multi_task_data", "test.csv")

# Model save directory
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Outputs
PREDICTIONS_DIR = os.path.join(BASE_DIR, "predictions")
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Device & seed
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 8642

# ---------------------------------------------------------------------------
# Model config (mirrors src/jointlearning/config.py)
# ---------------------------------------------------------------------------
MODEL_CONFIG = {
    "encoder_name": "bert-base-uncased",
    "num_cls_labels": 2,
    "num_bio_labels": 7,
    "num_rel_labels": 2,
    "dropout": 0.2,
}

# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------
TRAINING_CONFIG = {
    "batch_size": 16,
    "num_epochs": 20,
    "learning_rate": 1e-5,
    "weight_decay": 0.1,
    "gradient_clip_val": 1.0,
    "patience_epochs": 10,
}

# ---------------------------------------------------------------------------
# Dataset config
# ---------------------------------------------------------------------------
DATASET_CONFIG = {
    "max_length": 512,
    "negative_relation_rate": 2.0,
    "max_random_span_len": 5,
    "ignore_id": -100,
}

# ---------------------------------------------------------------------------
# Task loss weights for each separate model
# ---------------------------------------------------------------------------
TASK_WEIGHTS = {
    "cls": {"cls": 1.0, "bio": 0.0, "rel": 0.0},
    "bio": {"cls": 0.0, "bio": 1.0, "rel": 0.0},
    "rel": {"cls": 0.0, "bio": 0.0, "rel": 1.0},
}

# Model save paths (subdirs so each model keeps its own training_history.json)
MODEL_PATHS = {
    "cls": os.path.join(MODELS_DIR, "cls", "separate_cls.pt"),
    "bio": os.path.join(MODELS_DIR, "bio", "separate_bio.pt"),
    "rel": os.path.join(MODELS_DIR, "rel", "separate_rel.pt"),
}
for _p in MODEL_PATHS.values():
    os.makedirs(os.path.dirname(_p), exist_ok=True)

# Training log files
TRAINING_LOGS = {
    "cls": os.path.join(MODELS_DIR, "cls", "training_log.txt"),
    "bio": os.path.join(MODELS_DIR, "bio", "training_log.txt"),
    "rel": os.path.join(MODELS_DIR, "rel", "training_log.txt"),
}

# ---------------------------------------------------------------------------
# Label mappings (must match src/jointlearning/config.py)
# ---------------------------------------------------------------------------
id2label_cls = {0: "non-causal", 1: "causal"}
label2id_cls = {v: k for k, v in id2label_cls.items()}

id2label_bio = {
    0: "B-C", 1: "I-C", 2: "B-E", 3: "I-E",
    4: "B-CE", 5: "I-CE", 6: "O",
}
label2id_bio = {v: k for k, v in id2label_bio.items()}

id2label_rel = {0: "Rel_None", 1: "Rel_CE"}
label2id_rel = {"Rel_None": 0, "Rel_CE": 1}

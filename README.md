# Joint Causal Learning Model

A neural network model for **joint causal extraction** from text, combining three interconnected tasks:
1. **Sentence Classification**: Determining if a sentence contains causal relationships
2. **Span Detection**: Identifying cause and effect phrases using BIO tagging  
3. **Relation Extraction**: Linking cause-effect pairs with typed relations

The model is built on BERT and trained end-to-end for robust causal reasoning in scientific and academic texts.

🤗 **[View Model on Hugging Face Hub](https://huggingface.co/rasoultilburg/SocioCausaNet)** - Pre-trained model, examples, and documentation

## 🚀 Quick Start

### Using the Pre-trained Model from Hugging Face

```python
from transformers import AutoModel, AutoTokenizer

# Load the model from Hugging Face Hub
model = AutoModel.from_pretrained(
    "rasoultilburg/SocioCausaNet",  # Official pre-trained model
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "rasoultilburg/SocioCausaNet",
    trust_remote_code=True
)

# Analyze causal relationships
sentences = [
    "Smoking causes lung cancer and heart disease",
    "Exercise improves physical health and mental well-being",
    "This is a non-causal sentence about weather"
]

results = model.predict(
    sentences,
    tokenizer=tokenizer,
    rel_mode="neural",        # "neural", "auto", or "heuristic"
    rel_threshold=0.8,             # Confidence threshold for relations
    cause_decision="cls+span"      # "cls+span", "cls_only", or "span_only"
)

# Results contain causal analysis for each sentence
print(results)
```

### Example Output

```json
[
  {
    "text": "Smoking causes lung cancer and heart disease",
    "causal": true,
    "relations": [
      {
        "cause": "Smoking",
        "effect": "lung cancer",
        "type": "Rel_CE"
      },
      {
        "cause": "Smoking", 
        "effect": "heart disease",
        "type": "Rel_CE"
      }
    ]
  },
  {
    "text": "Exercise improves physical health and mental well-being",
    "causal": true,
    "relations": [
      {
        "cause": "Exercise",
        "effect": "physical health",
        "type": "Rel_CE"
      },
      {
        "cause": "Exercise",
        "effect": "mental well-being",
        "type": "Rel_CE"
      }
    ]
  },
  {
    "text": "This is a non-causal sentence about weather",
    "causal": false,
    "relations": []
  }
]
```
```

### Installation for Development

```bash
# Clone the repository
git clone <repository-url>
cd JointLearning

# Create virtual environment
python -m venv myenv
myenv\Scripts\activate  # Windows
# source myenv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements_cpu.txt  # For CPU
# pip install -r requirements_gpu.txt  # For GPU with CUDA

# Install the package in development mode
pip install -e src/
```

### For R Language Users 📊

🚧 **Coming Soon!** We're developing an R wrapper package that leverages Python's transformers library behind the scenes.

**Our Approach**: Instead of a native R implementation, we're creating an **automated R interface** that handles all the Python/transformers complexity for you, making causal analysis seamless for the R community.

**Key Benefits:**
- **Zero Python knowledge required** - everything handled automatically
- **Automatic environment management** - Python setup done behind the scenes  
- **Native R data structures** - work with familiar data.frames and tibbles
- **R ecosystem integration** - compatible with tidyverse, ggplot2, and Shiny
- **One-command installation** - no manual Python environment setup

**Expected R workflow:**
```r
# Future R usage (under development)
# install.packages("jointcausal")  # Will auto-handle Python dependencies
library(jointcausal)

# First run automatically sets up Python environment
model <- load_causal_model("rasoultilburg/SocioCausaNet")

# Native R data frame input/output
sentences_df <- data.frame(
  text = c(
    "Smoking causes lung cancer and heart disease",
    "Exercise improves physical health and mental well-being"
  )
)

# Get results as R data frame
results_df <- predict_causal(model, sentences_df)

# R-style analysis and visualization
library(ggplot2)
library(dplyr)

results_df %>%
  filter(causal == TRUE) %>%
  plot_causal_network() +
  theme_minimal()

# Seamless Shiny integration
causal_analysis_app(results_df)
```

**Technical Implementation:**
- Uses `reticulate` package to interface with Python transformers
- Automated conda/pip environment management
- Transparent data conversion between R and Python
- Error handling and user-friendly R messages
- Memory-efficient batch processing for large R datasets

📧 **Stay Updated**: Watch this repository or follow our [Hugging Face model page](https://huggingface.co/rasoultilburg/SocioCausaNet) for R package release announcements.

## 📊 Model Architecture

The **JointCausalModel** uses a multi-task learning approach:

```
Input Text → BERT Encoder → Hidden States
                           ├── Classification Head → Causal/Non-causal
                           ├── BIO Tagging Head → C/E/CE/O tags  
                           └── Relation Head → Cause-Effect pairs
```

### Task Details

- **Classification**: Binary classification (causal vs non-causal sentences)
- **BIO Tagging**: 7 labels for Named Entity Recognition of causal spans
- **Relation Extraction**: Links cause-effect pairs (`Rel_CE` vs `Rel_None`)

## 🏷️ Understanding BIO Tags

**BIO (Beginning-Inside-Outside)** is a tagging scheme used for Named Entity Recognition to identify causal spans in text:

### BIO Tag Structure
- **B-** : **Beginning** of an entity span
- **I-** : **Inside** (continuation) of an entity span  
- **O** : **Outside** any entity (non-causal tokens)

### Our 7 BIO Labels
| Tag | Full Name | Description | Example |
|-----|-----------|-------------|---------|
| `B-C` | Beginning-Cause | First token of a cause phrase | **Smoking** causes cancer |
| `I-C` | Inside-Cause | Continuation of cause phrase | Excessive **smoking** causes cancer |
| `B-E` | Beginning-Effect | First token of an effect phrase | Smoking causes **cancer** |
| `I-E` | Inside-Effect | Continuation of effect phrase | Smoking causes lung **cancer** |
| `B-CE` | Beginning-CauseEffect | First token of combined cause-effect | **Stress** can worsen itself |
| `I-CE` | Inside-CauseEffect | Continuation of combined span | **Chronic** **stress** worsens itself |
| `O` | Outside | Non-causal tokens | Smoking **causes** cancer |

### BIO Tagging Examples

**Example 1**: "Smoking causes lung cancer"
```
Token:  Smoking  causes  lung    cancer
BIO:    B-C      O       B-E     I-E
```

**Example 2**: "Heavy alcohol consumption leads to liver damage"
```
Token:  Heavy  alcohol  consumption  leads  to  liver  damage  
BIO:    B-C    I-C      I-C          O      O   B-E    I-E
```

**Example 3**: "Stress can worsen stress levels" (self-referential)
```
Token:  Stress  can  worsen  stress  levels
BIO:    B-CE    O    O       B-CE    I-CE
```

## 🏗️ Project Structure & Directory Guide

```
JointLearning/
├── README.md
├── requirements_cpu.txt         # CPU-only dependencies (PyTorch CPU, transformers, etc.)
├── requirements_gpu.txt         # GPU dependencies with CUDA support
│
├── datasets/                    # Training and evaluation data
│   ├── expert_multi_task_data/  # Human-annotated causal relationship data
│   │   ├── train.csv           # Training set with expert annotations
│   │   ├── val.csv             # Validation set for model tuning
│   │   ├── test.csv            # Test set for final evaluation
│   │   ├── doccano_train.jsonl # Training data in Doccano annotation format
│   │   ├── doccano_val.jsonl   # Validation data in Doccano format
│   │   └── doccano_test.jsonl  # Test data in Doccano format
│   ├── pseudo_annotate_data/    # LLM-generated training data for data augmentation
│   │   ├── llama3_8b_processed.csv     # Processed pseudo-labels from Llama3-8B
│   │   ├── llama3_8b_processed.jsonl  # Same data in JSONL format
│   │   └── llama3_8b_raw.jsonl        # Raw LLM outputs before processing
│   └── raw_100k_sentences_for_pseudo_annotation/  # Unlabeled data for pseudo-labeling
│       ├── 100k_sentences.csv  # Large corpus for pseudo-annotation
│       └── doccano_*.zip       # Annotation project backups
│
├── src/jointlearning/           # Core library (pip-installable package)
│   ├── __init__.py             # Package initialization
│   ├── model.py                # JointCausalModel class definition
│   ├── config.py               # Configuration settings (model, training, dataset)
│   ├── dataset_collator.py     # Data loading, preprocessing, and batch collation
│   ├── trainer.py              # Training loop with multi-task loss and early stopping
│   ├── evaluate_joint_causal_model.py  # Evaluation metrics and reporting
│   ├── main.py                 # Main training script
│   ├── loss.py                 # Custom loss functions (e.g., GCE loss)
│   └── utility.py              # Helper functions (class weights, label counting)
│
├── hf_port/                     # Hugging Face Hub integration
│   ├── modeling_joint_causal.py     # HF-compatible model implementation
│   ├── configuration_joint_causal.py # Model configuration for HF
│   ├── config.py               # Configuration bridge for HF compatibility
│   ├── save_for_hf.py          # Export trained model to HF format
│   ├── upload_to_hf.py         # Upload model to Hugging Face Hub
│   ├── automodel_test.py       # Test script for HF AutoModel integration
│   └── joint_causal_model_for_hf/   # Generated HF model directory
│       ├── config.json         # Model configuration
│       ├── model.safetensors   # Model weights in SafeTensors format
│       ├── modeling_joint_causal.py # Model code
│       ├── configuration_joint_causal.py # Config code
│       └── tokenizer files     # Tokenizer configuration and vocab
│
└── Notebooks/                   # Jupyter notebooks for analysis and experimentation
   ├── expert_bert_softmax_test.ipynb      # Model testing and validation
   ├── llm_testing.ipynb                   # LLM pseudo-labeling experiments
   ├── model_evaluation_analysis.ipynb    # Detailed performance analysis
   ├── evaluation_report.md               # Evaluation summary report
   └── predictions/                       # Saved model predictions

```

## 🔧 Key Scripts & Their Functions

### Core Training Scripts
- **`src/jointlearning/main.py`**: Main training pipeline
  - Loads datasets and creates data loaders
  - Initializes the JointCausalModel
  - Runs multi-task training with early stopping
  - Saves best model based on validation F1 score

- **`src/jointlearning/trainer.py`**: Training loop implementation
  - Multi-task loss computation (classification + BIO + relation)
  - Gradient clipping and optimization
  - Learning rate scheduling
  - Early stopping with patience mechanism

- **`src/jointlearning/evaluate_joint_causal_model.py`**: Comprehensive evaluation
  - Computes metrics for all three tasks
  - Generates detailed classification reports
  - Calculates overall F1 scores
  - Provides task-specific performance analysis

### Hugging Face Integration Scripts
- **`hf_port/save_for_hf.py`**: Model export for Hugging Face
  - Converts trained PyTorch model to HF format
  - Registers custom model and configuration classes
  - Saves model weights, config, and tokenizer files
  - Creates HF-compatible model directory

- **`hf_port/upload_to_hf.py`**: Upload to Hugging Face Hub
  - Authenticates with HF Hub
  - Uploads model files to specified repository
  - Handles model versioning and metadata
  - Creates model cards and documentation

- **`hf_port/automodel_test.py`**: Test AutoModel integration
  - Loads model using HF AutoModel API
  - Tests inference pipeline
  - Validates model outputs
  - Demonstrates usage examples

### Utility Scripts
- **`src/jointlearning/dataset_collator.py`**: Data preprocessing
  - Tokenizes input text
  - Aligns BIO labels with tokenized text
  - Creates relation extraction pairs
  - Handles padding and batching

- **`src/jointlearning/utility.py`**: Helper functions
  - Computes class weights for imbalanced data
  - Label distribution analysis
  - Data statistics and validation

## 📊 Dataset Format & Requirements

### Expected CSV Format

The model expects data in the format used in `datasets\expert_multi_task_data\train.csv`:

```csv
id,text,entities,relations,Comments
9296,"Table 1 displays the different treatments employed.","[{'id': 9622, 'label': 'non-causal', 'start_offset': 0, 'end_offset': 53}]",[],[]
6908,"because the messages are sent anonymously, the decision only depends on the number of messages.","[{'id': 6298, 'label': 'cause', 'start_offset': 0, 'end_offset': 42}, {'id': 6299, 'label': 'effect', 'start_offset': 44, 'end_offset': 95}]","[{'id': 2857, 'from_id': 6298, 'to_id': 6299, 'type': 'Rel_CE'}]",[]
7434,"social rejection evokes responses because it threatens this basic need.","[{'id': 7615, 'label': 'effect', 'start_offset': 0, 'end_offset': 31}, {'id': 7616, 'label': 'cause', 'start_offset': 32, 'end_offset': 70}]","[{'id': 3436, 'from_id': 7616, 'to_id': 7615, 'type': 'Rel_CE'}]",[]
```

### Column Descriptions

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `id` | Integer | Unique identifier for each sentence | 9296 |
| `text` | String | Input sentence | "Smoking causes lung cancer" |
| `entities` | JSON Array | Entity annotations with labels and positions | `[{'id': 123, 'label': 'cause', 'start_offset': 0, 'end_offset': 7}]` |
| `relations` | JSON Array | Relations between entities | `[{'id': 456, 'from_id': 123, 'to_id': 124, 'type': 'Rel_CE'}]` |
| `Comments` | Array | Additional comments (usually empty) | `[]` |

### Entity Labels

The `entities` column contains annotations with these labels:
- **`cause`**: Spans that represent causes
- **`effect`**: Spans that represent effects  
- **`non-causal`**: Entire sentences that are non-causal

### Entity Structure
```json
{
  "id": 123,
  "label": "cause",           # "cause", "effect", or "non-causal"
  "start_offset": 0,          # Character position where span starts
  "end_offset": 7             # Character position where span ends
}
```

### Relation Structure
```json
{
  "id": 456,
  "from_id": 123,             # ID of the cause entity
  "to_id": 124,               # ID of the effect entity  
  "type": "Rel_CE"            # Relation type: "Rel_CE" 
}
```

### Data Preprocessing

The training pipeline automatically converts this format to the internal representation:
- **Entities** → **BIO tags** aligned with tokenized text
- **Relations** → **Relation pairs** for the relation extraction head
- **Entity labels** → **Classification labels** (causal vs non-causal)

### Supported Relation Types
- **`Rel_CE`**: Positive causal relation (cause → effect)

## 🎯 Prediction Modes

The model supports different prediction strategies:

### Relation Extraction Modes
- **`neural`**: Always use the neural relation head for all valid cause-effect pairs
- **`auto`**: Simplified relation extraction for sentences with single causes/effects  

### Causality Decision Strategies  
- **`cls+span`** (default): Sentence is causal if both classification score ≥ 0.5 AND valid cause+effect spans exist
- **`cls_only`**: Based only on classification head output
- **`span_only`**: Based only on presence of cause+effect spans

## 📈 Training Your Own Model

### 1. Prepare Data

The model expects data in the format used in `datasets\expert_multi_task_data\train.csv`.

#### Required CSV Columns:
- **`id`**: Unique identifier for each sentence
- **`text`**: Input sentences (string)
- **`entities`**: JSON array of entity annotations with labels and character offsets
- **`relations`**: JSON array of relations between entities
- **`Comments`**: Additional comments (usually empty array)

#### Example Training Data:
```csv
id,text,entities,relations,Comments
1,"Smoking causes lung cancer","[{'id': 1, 'label': 'cause', 'start_offset': 0, 'end_offset': 7}, {'id': 2, 'label': 'effect', 'start_offset': 15, 'end_offset': 26}]","[{'id': 1, 'from_id': 1, 'to_id': 2, 'type': 'Rel_CE'}]",[]
2,"Regular sentence about weather","[{'id': 3, 'label': 'non-causal', 'start_offset': 0, 'end_offset': 29}]",[],[]
3,"Stress can worsen anxiety levels","[{'id': 4, 'label': 'cause', 'start_offset': 0, 'end_offset': 6}, {'id': 5, 'label': 'effect', 'start_offset': 18, 'end_offset': 32}]","[{'id': 2, 'from_id': 4, 'to_id': 5, 'type': 'Rel_CE'}]",[]
```

#### Entity Annotation Guidelines:
- Use **character offsets** (not token positions) for `start_offset` and `end_offset`
- **Label types**: `cause`, `effect`, or `non-causal` (for entire non-causal sentences)
- Each entity needs a unique `id` within the sentence
- **Non-causal sentences**: Annotate the entire sentence with `non-causal` label

#### Relation Annotation Guidelines:
- `from_id` refers to the cause entity ID
- `to_id` refers to the effect entity ID  
- `type` should be `Rel_CE` for positive causal relations
- Use `Rel_Zero` for negative/hard negative relations (optional)

#### Data Preprocessing Pipeline:
The training system automatically converts this format to internal representations:
1. **Entities** → **BIO tags** aligned with BERT tokens
2. **Character offsets** → **Token positions** 
3. **Relations** → **Relation extraction pairs**
4. **Entity presence** → **Classification labels** (causal/non-causal)

### 2. Configure Training

Edit `src/jointlearning/config.py`:

```python
MODEL_CONFIG = {
    "encoder_name": "bert-base-uncased",
    "num_cls_labels": 2,
    "num_bio_labels": 7, 
    "num_rel_labels": 2,
    "dropout": 0.2,
}

TRAINING_CONFIG = {
    "batch_size": 16,
    "num_epochs": 20,
    "learning_rate": 1e-5,
    "weight_decay": 0.1,
}
```

### 3. Train the Model

```bash
cd src/jointlearning
python main.py
```

The training script will:
- Load and preprocess your data
- Train the multi-task model with early stopping
- Save the best model based on validation F1 score
- Generate detailed evaluation metrics

## 🤗 Publishing to Hugging Face Hub

### 1. Prepare Model for Hub

```bash
cd hf_port
# Edit save_for_hf.py to set your model path
python save_for_hf.py
```

### 2. Upload to Hub

```bash
# Login to Hugging Face
huggingface-cli login

# Edit upload_to_hf.py to set your repository ID
python upload_to_hf.py
```

### 3. Test Your Published Model

```bash
# Edit automodel_test.py to use your repo ID
python automodel_test.py
```

## 📋 Example Output Format

The model returns a list of dictionaries, one for each input sentence:

```json
{
  "text": "Smoking causes lung cancer and heart disease",
  "causal": true,
  "relations": [
    {
      "cause": "Smoking",
      "effect": "lung cancer",
      "type": "Rel_CE"
    },
    {
      "cause": "Smoking", 
      "effect": "heart disease",
      "type": "Rel_CE"
    }
  ]
}
```

## 🔧 Advanced Configuration

### BIO Tag Post-processing Rules

The model applies sophisticated rules to clean up BIO predictions:
- Merge adjacent B-tags of the same type
- Bridge small gaps with connector words ("of", "to", "and", etc.)
- Resolve mixed cause-effect spans
- Handle overlapping entity boundaries

### Relation Extraction Thresholds

Adjust `rel_threshold` to control relation extraction sensitivity:
- **0.5**: Balanced precision/recall
- **0.8**: Higher precision, fewer false positives
- **0.3**: Higher recall, more potential relations

## 📊 Evaluation Metrics

The model reports detailed metrics for all three tasks:
- **Classification**: Precision, Recall, F1 for causal/non-causal
- **BIO Tagging**: Token-level metrics for each entity type
- **Relation Extraction**: Relation-level precision, recall, F1

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Citation

If you use this model in your research, please cite:

```bibtex
@misc{jointcausal2025,
  title={Joint Causal Learning: Multi-task Neural Networks for Causal Relationship Extraction},
  author={Your Name},
  year={2025},
  publisher={GitHub},
  url={https://github.com/yourusername/JointLearning}
}
```

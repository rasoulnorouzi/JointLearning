---
license: gpl-2.0
language: en
base_model: google-bert/bert-base-uncased
pipeline_tag: token-classification
tags:
- causal-extraction
- relation-extraction
- bert
- pytorch
- causality
library_name: transformers
---

# JointCausalModel for Causal Extraction

This repository contains JointCausalModel, a PyTorch-based model for joint causal extraction, optimized for use with the Hugging Face transformers library. The model is built upon `google-bert/bert-base-uncased` and is designed to identify and structure causal relationships within text.

**GitHub Repository**: [https://github.com/rasoulnorouzi/JointLearning](https://github.com/rasoulnorouzi/JointLearning/tree/main)

## Model Description

This model performs three tasks simultaneously:

1. **Sentence-level Causal Classification**: Determines whether a sentence contains a causal statement.
2. **Span Extraction**: Identifies the specific Cause, Effect, and combined Cause-Effect spans within the text using a BIO tagging scheme.
3. **Relation Extraction**: Establishes the relationships between the identified cause and effect spans.

> **Note**: This model uses a custom implementation and requires `trust_remote_code=True` when loading with AutoModel.

## How to Use

To get started, load the model and tokenizer from the Hugging Face Hub:

```python
from transformers import AutoModel, AutoTokenizer

repo_id = "rasoultilburg/SocioCausaNet"

model = AutoModel.from_pretrained(
    repo_id,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    repo_id
)
```

### Inference API

The primary method for inference is `model.predict()`, which processes a list of sentences and returns detailed causal information:

```python
# Example of a simple prediction call
results = model.predict(
    sents=["The heavy rainfall led to severe flooding in the coastal regions."],
    tokenizer=tokenizer,
    rel_mode="neural",
    rel_threshold=0.5,
    cause_decision="cls+span"
)
```

### Understanding the predict() Parameters

Think of this model as a "Causality Detective." The parameters are the instructions you give the detective on how to investigate the text.

| Parameter | What it is & How it works | Analogy |
|-----------|---------------------------|---------|
| `sents` | The list of sentences you want the model to analyze. | The "case files" you give to the detective. |
| `rel_mode` | Strategy for finding relationships.<br/>- `'auto'`: A smart, efficient mode. For simple cases (one cause-one effect, one cause-multiple effects, multiple causes-one effect), it automatically connects them using rules. For complex cases (multiple causes and multiple effects), it uses a neural network to determine connections.<br/>- `'neural_only'`: Uses a neural network to validate every potential cause-effect connection, checking whether there is a relationship between each pair of entities. More thorough but slower. | The Detective's Strategy<br/>- `'auto'` is the Smart Detective who uses simple logic for obvious cases but calls in the expert (neural network) for complex situations.<br/>- `'neural_only'` is the Expert Detective who carefully analyzes every possible connection using advanced techniques (neural network) regardless of complexity. |
| `rel_threshold` | The confidence score needed to report a relationship (from 0.0 to 1.0).<br/>- High value (e.g., 0.8): Only reports relationships it's very sure about. Fewer, but more accurate results.<br/>- Low value (e.g., 0.3): Reports any potential link, even hunches. More results, but some may be incorrect. | The Detective's "Burden of Proof."<br/>- High value: Needs a lot of evidence before making an accusation.<br/>- Low value: Follows up on even the smallest lead. |
| `cause_decision` | The criteria for deciding if a sentence is causal.<br/>- `'cls_only'`: Decides based on overall sentence meaning.<br/>- `'span_only'`: Decides only if it finds distinct "cause" and "effect" phrases.<br/>- `'cls+span'`: Strictest mode. Sentence must have causal meaning AND contain distinct cause/effect phrases. | The Panel of Judges<br/>- `'cls_only'` is the "Big Picture" Judge.<br/>- `'span_only'` is the "Evidence-Focused" Judge.<br/>- `'cls+span'` requires both judges to agree. Most reliable option. |

## Complete Example

Here is a complete, runnable example demonstrating how to use the model and format the output:

```python
from transformers import AutoModel, AutoTokenizer
import json

# 1. Load the model and tokenizer
repo_id = "rasoultilburg/SocioCausaNet"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(repo_id)

# 2. Define input sentences
sentences = [
    "Insomnia causes depression and a lack of concentration in children.",
    "Due to the new regulations, the company's profits declined sharply.",
    "The sun rises in the east."  # Non-causal example
]

# 3. Get predictions from the model
results = model.predict(
    sentences,
    tokenizer=tokenizer,
    rel_mode="neural",
    rel_threshold=0.5,
    cause_decision="cls+span"
)

# 4. Print the results in a readable format
print(json.dumps(results, indent=2, ensure_ascii=False))
```

### Example Output

The predict method returns a list of dictionaries, where each dictionary corresponds to an input sentence:

```json
[
  {
    "text": "Insomnia causes depression and a lack of concentration in children.",
    "causal": true,
    "relations": [
      {
        "cause": "Insomnia",
        "effect": "depression",
        "type": "Rel_CE"
      },
      {
        "cause": "Insomnia",
        "effect": "a lack of concentration in children",
        "type": "Rel_CE"
      }
    ]
  },
  {
    "text": "Due to the new regulations, the company's profits declined sharply.",
    "causal": true,
    "relations": [
      {
        "cause": "the new regulations",
        "effect": "the company's profits declined sharply",
        "type": "Rel_CE"
      }
    ]
  },
  {
    "text": "The sun rises in the east.",
    "causal": false,
    "relations": [],
    "spans": []
  }
]
```

## Model Architecture

The JointCausalModel requires custom code, which is why `trust_remote_code=True` is necessary. The architecture consists of a BERT encoder followed by three specialized heads for the joint tasks.

The key files defining the model are:

- `modeling_joint_causal.py`: Contains the main JointCausalModel class which defines the model's architecture. It inherits from `transformers.PreTrainedModel` to ensure compatibility with the Hugging Face ecosystem.
- `configuration_joint_causal.py`: Defines the JointCausalConfig class, which stores the model's configuration and hyperparameters.

## Citation

If you use this model in your work, please consider citing this repository.

```bibtex
@misc{jointcausalmodel2024,
  title={JointCausalModel: Joint Learning for Causal Extraction},
  author={Rasoul Norouzi},
  year={2024},
  howpublished={GitHub Repository},
  url={https://github.com/rasoulnorouzi/JointLearning/tree/main}
}
```

For more details and source code, visit the [GitHub repository](https://github.com/rasoulnorouzi/JointLearning/tree/main)
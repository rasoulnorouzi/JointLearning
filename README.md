## Project Structure

```
JointLearning/
├── README.md
├── requirements.txt
├── .gitignore
|
├── datasets/                # (git-ignored) raw & processed data
│   └── …
|
├── src/                     # Core library (pip-installable)
│   └── jointlearning/
│       ├── __init__.py
│       ├── dataset.py       # Data loading & preprocessing
│       ├── collator.py      # Batch collation & padding
│       ├── model.py         # Model definition & post-processing rules
│       ├── evaluation.py    # Metrics & annotator-agreement
│       ├── training.py      # Training loops & checkpoints
│       └── utils.py         # Helpers (logging, config)
|
├── scripts/                 # CLI tools
│   ├── train.py             # `python scripts/train.py --config config.yaml`
│   ├── infer.py             # Load model + predict
│   ├── evaluate.py          # Evaluate on held-out set
│   └── agreement.py         # Compute annotator agreement
|
└── analysis/                # (optional) Jupyter notebooks for EDA & error analyses
    └── …
```

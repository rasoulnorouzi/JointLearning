# JointLearning
JointLearning


JointLearning/
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   └── jointlearning/
│       ├── loader.py
│       ├── preprocess.py
│       ├── model.py
│       ├── inference.py
│       ├── rules.py
│       ├── eval.py
│       └── utils.py
│
├── hf/                          # Hugging Face integrations
│   ├── __init__.py
│   ├── hf_config.py            # load & manage HF configs
│   ├── hf_model.py             # AutoModel-based model wrappers
│   └── hf_pipeline.py          # Pipeline API wrappers
│
├── scripts/
│   ├── train.py
│   ├── hf_train.py             # optional: HF-specific training
│   ├── infer.py
│   ├── hf_infer.py             # uses hf_pipeline
│   └── compute_agreement.py
│
└── analysis/
    ├── agreement.ipynb
    ├── eda.ipynb
    └── error_analysis.ipynb

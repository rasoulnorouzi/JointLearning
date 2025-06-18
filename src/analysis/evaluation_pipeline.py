from __future__ import annotations
"""evaluation_utils.py
Reusable evaluation routine for JointCausalModel experiments.

The main entry point is :func:`run_full_evaluation`, which performs the full
predict → convert → evaluate pipeline for every combination of models,
classification modes, thresholds, scenarios and evaluation modes supplied by
the caller. Results are returned as a tidy :class:`pandas.DataFrame` and, if
requested, written to a Markdown report alongside per-configuration prediction
CSVs.

Example
-------
>>> from evaluation_utils import run_full_evaluation
>>> models = {
...     "bert-softmax": "/path/to/bert_softmax.pt",
...     "bert-gce"    : "/path/to/bert_gce.pt",
... }
>>> df = run_full_evaluation(
...     model_paths=models,
...     thresholds=[0.7, 0.8],
...     classification_modes=["cls+span", "span_only"],
...     scenarios=["all_documents", "filtered_causal"],
...     write_markdown=True,
... )

Dependencies
------------
- torch
- transformers
- pandas
- tqdm
- jointlearning (MODEL_CONFIG, JointCausalModel)
- llm2doccano.convert_llm_output_to_doccano
- analysis.causal_eval.evaluate, analysis.causal_eval.display_results
"""

import sys
import os

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
import json
import tempfile
from pathlib import Path
from typing import Mapping, Sequence, Dict, Any, List
import random
import numpy as np

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

# Project-specific imports
from jointlearning.config import MODEL_CONFIG
from jointlearning.model import JointCausalModel
from analysis.llm2doccano import convert_llm_output_to_doccano
from analysis.causal_eval import evaluate, display_results

# ---------------------------------------------------------------------------
# Reproducibility utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible results across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make CUDA operations deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _prepare_device(device: str | torch.device | None = None) -> torch.device:
    """Return a valid :class:`torch.device`."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    return device


def _ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def run_full_evaluation(
    *,
    model_paths: Mapping[str, str],
    thresholds: Sequence[float],
    classification_modes: Sequence[str],
    scenarios: Sequence[str],
    eval_modes: Sequence[str] = ("coverage", "discovery"),
    test_data_path: str = "../../datasets/expert_multi_task_data/test.csv",
    tokenizer_name: str = MODEL_CONFIG["encoder_name"],
    batch_size: int = 32,
    device: str | torch.device | None = None,
    write_markdown: bool = True,
    save_report_dir: str = "evaluation_report.md",
    save_predictions_dir: str = "analysis/predictions",
    seed: int = 42,
) -> pd.DataFrame:
    """Run prediction and evaluation over a grid of settings.

    Parameters
    ----------
    model_paths
        Keys are *model nicknames* (str) and values are paths to ``.pt`` weight
        files. Each model will be evaluated independently.
    thresholds
        List of relation-confidence cut-offs passed to ``model.predict``.
    classification_modes
        List of ``cause_decision`` modes (e.g. ``"cls+span"``).
    scenarios
        List of corpus subsets accepted by :func:`analysis.causal_eval.evaluate`.
    eval_modes
        Evaluation perspectives (default ``("coverage", "discovery")``).
    test_data_path
        CSV with at least a ``text`` column and gold labels.
    tokenizer_name
        HuggingFace tokeniser ID for the encoder.
    batch_size
        Mini-batch size for forward passes.    device
        GPU/CPU spec. ``None`` chooses CUDA if available.
    write_markdown
        If *True*, write a narrative report to *save_report_dir*.
    save_report_dir
        Destination Markdown file path where the evaluation report will be saved.
    save_predictions_dir
        Directory path where per-configuration predictions (CSV) are stored.
    seed
        Random seed for reproducible results (default: 42).

    Returns
    -------
    pandas.DataFrame
        One row per combination of configuration values with the metrics
        produced by :func:`evaluate` as columns.
    """
    # ---------------------------------------------------------------------
    # Reproducibility setup
    # ---------------------------------------------------------------------
    set_seed(seed)
    
    # ---------------------------------------------------------------------
    # Initialisation
    # ---------------------------------------------------------------------
    device = _prepare_device(device)
    save_predictions_dir = _ensure_dir(save_predictions_dir)

    # Load resources shared across loops
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    gold_df = pd.read_csv(test_data_path)
    texts: List[str] = gold_df["text"].tolist()

    # Store metrics here for final DataFrame
    metrics_records: List[Dict[str, Any]] = []

    # ---------------------------------------------------------------------    # Nested evaluation loops
    # ---------------------------------------------------------------------
    for model_name, model_path in model_paths.items():
        # Load model once per weight file
        model = JointCausalModel(**MODEL_CONFIG)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device).eval()

        # Optional: Clear GPU cache between models if memory is limited
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

        for class_mode in classification_modes:
            for threshold in thresholds:
                # ------------------------------------------------------
                # Prediction step (fresh run for each threshold & mode)
                # ------------------------------------------------------
                all_results = []
                num_batches = (len(texts) + batch_size - 1) // batch_size

                for i in tqdm(range(num_batches),
                              desc=f"{model_name} | {class_mode} | thr={threshold:.2f}"):
                    batch = texts[i * batch_size : (i + 1) * batch_size]
                    with torch.no_grad():
                        batch_results = model.predict(
                            batch,
                            tokenizer=tokenizer,
                            rel_mode="neural_only",
                            rel_threshold=threshold,
                            cause_decision=class_mode,
                        )
                    all_results.extend(batch_results)

                # Save raw prediction JSON for reproducibility
                raw_json_path = save_predictions_dir / f"{model_name}__{class_mode}__thr{threshold:.2f}.json"
                with open(raw_json_path, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, ensure_ascii=False, indent=2)

                # Doccano-style CSV conversion
                pred_df = convert_llm_output_to_doccano(all_results)
                csv_path = save_predictions_dir / f"{model_name}__{class_mode}__thr{threshold:.2f}.csv"
                pred_df.to_csv(csv_path, index=False)

                # ------------------------------------------------------
                # Evaluation across scenarios × eval_modes
                # ------------------------------------------------------
                for scenario in scenarios:
                    for eval_mode in eval_modes:
                        banner = (
                            "#"*12 +
                            f" Model={model_name} | class={class_mode} | thr={threshold:.2f} | "
                            f"scenario={scenario} | eval={eval_mode} " +
                            "#"*12
                        )
                        print("\n" + banner)

                        result_metrics = evaluate(
                            gold_df,
                            pred_df,
                            scenario=scenario,
                            eval_mode=eval_mode,
                        )
                        if isinstance(result_metrics, pd.DataFrame):
                            metric_dict = result_metrics.to_dict(orient="records")[0]
                        elif isinstance(result_metrics, pd.Series):
                            metric_dict = result_metrics.to_dict()
                        else:
                            metric_dict = dict(result_metrics)

                        display_results(result_metrics,
                                        title_prefix=f"{scenario} — {eval_mode}")

                        metric_dict.update(
                            model=model_name,
                            classification_mode=class_mode,
                            threshold=threshold,
                            scenario=scenario,
                            eval_mode=eval_mode,
                        )
                        metrics_records.append(metric_dict)

    # ---------------------------------------------------------------------
    # Assemble DataFrame
    # ---------------------------------------------------------------------
    metrics_df = pd.DataFrame(metrics_records)

    # ---------------------------------------------------------------------
    # Markdown report
    # ---------------------------------------------------------------------
    if write_markdown:
        _write_markdown_report(
            metrics_df,
            model_paths=model_paths,
            thresholds=thresholds,
            classification_modes=classification_modes,
            scenarios=scenarios,
            eval_modes=eval_modes,
            batch_size=batch_size,            tokenizer_name=tokenizer_name,
            save_report_dir=save_report_dir,
            save_predictions_dir=save_predictions_dir,
        )

    return metrics_df


# ---------------------------------------------------------------------------
# Markdown writer
# ---------------------------------------------------------------------------

def _write_markdown_report(
    df: pd.DataFrame,
    *,
    model_paths: Mapping[str, str],
    thresholds: Sequence[float],
    classification_modes: Sequence[str],
    scenarios: Sequence[str],
    eval_modes: Sequence[str],
    batch_size: int,
    tokenizer_name: str,
    save_report_dir: str | Path,
    save_predictions_dir: Path,
) -> None:
    """Generate a human-readable Markdown summary of *df*."""
    save_report_dir = Path(save_report_dir)
    save_report_dir.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []

    lines.append("# JointCausalModel Evaluation Report\n")
    lines.append("## Configuration\n")
    lines.append("| Setting | Value |\n|---------|-------|")
    lines.append(f"| Tokenizer | `{tokenizer_name}` |")
    lines.append(f"| Batch size | {batch_size} |")
    lines.append(f"| Thresholds | {', '.join(map(str, thresholds))} |")
    lines.append(f"| Classification modes | {', '.join(classification_modes)} |")
    lines.append(f"| Scenarios | {', '.join(scenarios)} |")
    lines.append(f"| Eval modes | {', '.join(eval_modes)} |")
    lines.append("\n")

    for model_name in model_paths:
        sub_df = df[df["model"] == model_name]
        lines.append(f"## {model_name}\n")
        for class_mode in classification_modes:
            sub_sub_df = sub_df[sub_df["classification_mode"] == class_mode]
            lines.append(f"### Classification: `{class_mode}`\n")
            for thr in thresholds:
                table_df = sub_sub_df[sub_sub_df["threshold"] == thr]
                if table_df.empty:
                    continue
                lines.append(f"#### Threshold = {thr:.2f}\n")
                metric_cols = [c for c in table_df.columns if c not in (
                    "model", "classification_mode", "threshold")]
                lines.append(table_df[metric_cols].to_markdown(index=False) + "\n")

                csv_file = save_predictions_dir / f"{model_name}__{class_mode}__thr{thr:.2f}.csv"
                if csv_file.exists():
                    rel_path = os.path.relpath(csv_file, save_report_dir.parent)
                    lines.append(f"[Predictions CSV]({rel_path})\n")

    save_report_dir.write_text("\n".join(lines), encoding="utf-8")

# # # ---------------------------------------------------------------------------
# # # Smoke test executed when run as a script
# # # ---------------------------------------------------------------------------

# def _smoke_test() -> None:
#     """Lightweight self-test verifying that run_full_evaluation executes.

#     The test builds a one-row dummy dataset, invokes the evaluation with an
#     empty model dictionary (thereby skipping heavy model loading) and asserts
#     that the returned DataFrame is empty. This ensures the function and its
#     argument plumbing work without depending on external files.
#     """
#     print("[Smoke-Test] Starting...", flush=True)

#     # Create a temporary CSV with the compulsory 'text' column
#     with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv") as tmp:
#         tmp.write("text\nThis is a dummy sentence.\n")
#         dummy_csv_path = tmp.name

#     df = run_full_evaluation(
#         model_paths={
#         'bert-softmax': r"C:\Users\norouzin\Desktop\JointLearning\src\jointlearning\expert_bert_softmax\expert_bert_softmax_model.pt",
#         'bert-gce-softmax-freeze': r"C:\Users\norouzin\Desktop\JointLearning\src\jointlearning\expert_bert_GCE_Softmax_Freeze\expert_bert_GCE_Softmax_Freeze_model.pt"    },
#         thresholds=[0.9],
#         classification_modes=['cls+span', 'span_only'],
#         scenarios=['all_documents', 'filtered_causal'],
#         eval_modes=["coverage", "discovery"],
#         test_data_path=r'C:\Users\norouzin\Desktop\JointLearning\datasets\expert_multi_task_data\test.csv',
#         save_predictions_dir=r'C:\Users\norouzin\Desktop\JointLearning\src\analysis\predictions',
#         save_report_dir=r'C:\Users\norouzin\Desktop\JointLearning\src\analysis\evaluation_report.md',
#         write_markdown=True,
#         seed=42,
#     )

#     assert not df.empty, "Expected non-empty DataFrame when models are supplied."
#     print(f"[Smoke-Test] Passed - Evaluated {len(df)} configurations", flush=True)


# if __name__ == "__main__":
#     _smoke_test()

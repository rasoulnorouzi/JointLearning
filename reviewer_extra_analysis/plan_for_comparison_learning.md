# Plan: Separate & Sequential Learning Baselines for Reviewer

## Context
A reviewer asked to demonstrate the advantages of joint learning over alternatives:
- **Separate**: Each task trained with its own encoder, evaluated independently
- **Sequential (pipeline)**: Train 3 separate models, chain at **test time**: CLS → BIO → REL. No shared encoder, errors propagate, **no cleanup** between stages.

Joint model results already exist — only producing baselines.

## What gets created under `reviewer_extra_analysis/sequential_learning/`

| File | Purpose |
|------|---------|
| `config.py` | Shared paths and constants |
| `train_separate.py` | Train 3 separate models (CLS-only, BIO-only, REL-only) |
| `evaluate_sequential.py` | Chain 3 models as pipeline at test time, save predictions |
| `evaluation_report.ipynb` | **Main deliverable** — notebook with all results, saved cell outputs |
| `models/` | Auto-created, holds `.pt` checkpoints |

## Training (`train_separate.py`)

Train 3 independent `JointCausalModel` instances with `task_loss_weights`:

| Model | `task_loss_weights` | Encoder optimized for |
|-------|-------------------|-----------------------|
| CLS-only | `{cls:1.0, bio:0.0, rel:0.0}` | Sentence classification |
| BIO-only | `{cls:0.0, bio:1.0, rel:0.0}` | Span detection (BIO tagging) |
| REL-only | `{cls:0.0, bio:0.0, rel:1.0}` | Relation extraction (given gold spans) |

**BIO weight = 1.0** (the 4.0 was for multitask balancing, irrelevant here).

Training setup: `bert-base-uncased`, 20 epochs, lr=1e-5, AdamW, batch=16, ReduceLROnPlateau, class weights (ENS), seed=8642.

**Model selection fix**: `train_model()` uses `overall_avg_f1` (avg of 3 tasks). For separate models, untrained heads add noise. Pass a wrapper `eval_fn_metrics` that overrides `overall_avg_f1` to track only the active task's F1. No changes to existing code.

## The Notebook: `evaluation_report.ipynb`

The main deliverable. Structured in 3 sections, each with saved outputs so the audience sees full results on open:

### Section 1: Per-task evaluation of separate models
- Load `separate_cls.pt` → `evaluate_model()` → CLS macro F1
- Load `separate_bio.pt` → `evaluate_model()` → BIO macro F1
- Load `separate_rel.pt` → `evaluate_model()` → REL macro F1 (uses gold spans from batch)
- **Table 1**: Per-task comparison

### Section 2: Sequential pipeline evaluation
Loads all 3 models, chains them sentence-by-sentence:

```
Input sentence
    │
    ▼
┌─────────────────────┐
│ CLS model (forward) │──→ is_causal (cls_only, threshold 0.5)
└─────────────────────┘
    │  if not causal → output {causal:false, spans:[], relations:[]}
    │  if causal → continue
    ▼
┌─────────────────────┐
│ BIO model (forward) │──→ bio_emissions → _apply_bio_rules()
│                     │                     _merge_spans() → spans
└─────────────────────┘
    │  Build ALL cause×effect pair combinations from found spans
    │  (spans may be empty — measure as error, don't fix)
    ▼
┌─────────────────────┐
│ REL model (forward  │──→ rel_logits → softmax → filter → relations
│  with span pairs)   │
└─────────────────────┘
    │  (relations may be empty — measure as error, don't fix)
    ▼
 Output {causal, spans, relations} — AS IS, NO CLEANUP
```

- Convert predictions to Doccano via `convert_llm_output_to_doccano()`
- Evaluate via `causal_eval.evaluate()` across all 4 combos
- **Table 2**: Sequential end-to-end results

### Section 3: Comparison with joint model
- Joint model results hardcoded from existing evaluation (no re-run)
- Combined Table 3 showing Sequential vs Joint

### Table 3 (final comparison)
| Model | Scenario | Eval Mode | Task1 F1 | Task2 Macro | Task3 F1 | Total Macro |
|-------|----------|-----------|----------|-------------|----------|-------------|
| Joint | all_docs | discovery | (existing) | | | |
| Joint | all_docs | coverage | (existing) | | | |
| Joint | filtered | discovery | (existing) | | | |
| Joint | filtered | coverage | (existing) | | | |
| Sequential | all_docs | discovery | X | X | X | X |
| Sequential | all_docs | coverage | X | X | X | X |
| Sequential | filtered | discovery | X | X | X | X |
| Sequential | filtered | coverage | X | X | X | X |

The gap between Joint and Sequential = cost of error propagation / benefit of joint learning.

## Files to read during implementation
- `src/jointlearning/model.py` — `predict()`, `_apply_bio_rules()`, `_merge_spans()`, `_decide_causal()` (lines 263–740)
- `src/jointlearning/main.py` — training entry point pattern
- `src/jointlearning/trainer.py` — `task_loss_weights` and `eval_fn_metrics` interface
- `src/jointlearning/evaluate_joint_causal_model.py` — `evaluate_model()` output format
- `src/analysis/causal_eval.py` — `evaluate()` function
- `src/analysis/llm2doccano.py` — `convert_llm_output_to_doccano()` interface

## Verification
1. `train_separate.py` completes for all 3 models without errors
2. Notebook Section 1 cell outputs show per-task F1 for each separate model
3. Notebook Section 2 produces Doccano-format predictions for all 452 test sentences and evaluates them
4. Notebook Section 3 loads or hardcodes joint results and shows combined comparison
5. Sequential Total Macro < Joint Total Macro (demonstrates error propagation)
6. All cell outputs saved so the notebook is watchable offline

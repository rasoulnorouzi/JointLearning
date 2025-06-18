# Evaluation Methodology

To provide a robust and multi-faceted assessment of model performance, we established a comprehensive evaluation framework with three hierarchical tasks:

1. **Task 1 — Document-level classification**  
2. **Task 2 — Causal span extraction**  
3. **Task 3 — End-to-end relation extraction**

Our methodology is grounded in two distinct **evaluation scenarios** so we can report both full-pipeline performance and focused, task-specific capabilities. For extraction tasks we chose the **coverage** evaluation mode, which fairly credits models that correctly identify all parts of a causal argument even when their segmentation differs from the gold standard.

---

## Evaluation Scenarios: A Two-Pronged Approach

### 1. `all_documents` Scenario — Assessing the Overall Pipeline  

*Evaluates the entire test corpus (452 documents).*  
The model must filter a mixed collection of texts, identify causal documents, and then extract spans and relations. The score therefore combines:

- Classification accuracy on non-causal texts, and  
- Extraction quality on causal texts.  

This yields a single end-to-end metric that reflects “real-world” usage.

### 2. `filtered_causal` Scenario — Isolating Extraction Performance  

*Evaluates only documents that **both** the gold standard **and** the model label as causal.*  
By restricting to the intersection set, we measure pure extraction skill without dilution from classification errors.

| Case | Document Content | Ground Truth | Model Prediction | Included in `filtered_causal`? | Rationale |
|------|-----------------|--------------|------------------|--------------------------------|-----------|
| **A** | “The drought led to crop failure.” | Causal | Causal | ✔ Yes | Both agree the doc is causal. |
| **B** | “The sky is blue today.” | Not causal | Not causal | ✘ No | Correctly identified as non-causal; extraction not relevant. |
| **C** | “The drought led to crop failure.” | Causal | Not causal | ✘ No | Model missed causality, so extraction isn’t scored. |
| **D** | “His speech had a strong effect.” | Not causal | Causal | ✘ No | Model’s false positive hurts Task 1 but is excluded here. |

This focused approach ensures Task 2 and Task 3 metrics aren’t distorted by documents where extraction is irrelevant.

---

## Evaluation Mode: Why **coverage**?

In Tasks 2–3 a single gold span can correspond to multiple valid, non-overlapping model spans (e.g., with conjunctions such as “and”, “or”). **Coverage** treats every predicted span that overlaps a gold span as a true positive, giving appropriate credit.

### Illustrative Example

> **Gold Cause:** `[drought and subsequent wildfires]`  
> **Model predictions:** `[drought]`, `[subsequent wildfires]`

Under **coverage**, both predictions count as **TP** because each overlaps the larger gold concept.

---

## Task-Specific Metric Computation

### Task 1 — Document-Level Classification  
- **TP**: causal doc correctly predicted causal  
- **FP**: non-causal doc predicted causal  
- **FN**: causal doc predicted non-causal  

### Task 2 — Cause/Effect Span Extraction  
- **TP**: predicted span overlaps a gold span of the same label  
- **FP**: predicted span overlaps no gold span of that label  
- **FN**: gold span overlapped by no prediction of that label  

### Task 3 — Cause → Effect Relation Extraction  
- **TP**: predicted relation whose cause & effect spans each overlap the correct gold spans **and** whose relation type matches  
- **FP**: predicted relation with no matching gold relation  
- **FN**: gold relation missed by the model  

---

## Performance Metrics

Using counts of **TP**, **FP**, and **FN** we compute:

- **Precision**  
  \[
  \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
  \]

- **Recall**  
  \[
  \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
  \]

- **F1-Score** (harmonic mean)  
  \[
  \text{F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
  \]

For **Task 2**, we report the **macro-average F1** across the *cause* and *effect* classes, giving equal weight to each entity type.

---



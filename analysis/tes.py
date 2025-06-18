# # %%
import json
import torch
from transformers import AutoTokenizer
from src.jointlearning.config import MODEL_CONFIG
from src.jointlearning.model import JointCausalModel
from src.causal_pseudo_labeling.llm2doccano import convert_llm_output_to_doccano
from analysis.causal_eval import evaluate, display_results
import pandas as pd
import tqdm
# %%
models = {
        'bert-softmax': "src/jointlearning/expert_bert_softmax/expert_bert_softmax_model.pt",
        'bert-gce': "src/jointlearning/expert_bert_gce/expert_bert_gce_model.pt",
        'bert-gce-softmax': "src/jointlearning/expert_bert_gce_softmax/expert_bert_gce_softmax_model.pt",
        'bert-gce-freeze-softmax': "src/jointlearning/expert_bert_gce_freeze_softmax/expert_bert_gce_freeze_softmax_model.pt",
    }

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%    
# Evaluation parameters
thresholds = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
causal_classification_modes = ['cls+span', 'span_only']
batch_size = 32
scenarios = ['all_documents', 'filtered_causal']
eval_modes = ['coverage', 'discovery']
test_data_dir = 'datasets/expert_multi_task_data/test.csv'
save_dir = 'analysis/predictions'
# %%
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_CONFIG["encoder_name"])
model=JointCausalModel(**MODEL_CONFIG)
model.load_state_dict(torch.load(models['bert-softmax'],map_location=DEVICE))
model.to(DEVICE).eval();
# %%
gold_df = pd.read_csv(test_data_dir)
texts = gold_df['text'].tolist()
# %%
batch_size = 32  # You can adjust this as needed
all_results = []
num_batches = (len(texts) + batch_size - 1) // batch_size
for i in tqdm.tqdm(range(num_batches), desc="Processing batches"):
    batch = texts[i * batch_size : (i + 1) * batch_size]
    batch_results = model.predict(
        batch,
        tokenizer=TOKENIZER,
        rel_mode="neural_only",           # or "auto"
        rel_threshold=0.75,         # adjust as needed
        cause_decision="cls+span" # or "cls_only", "span_only"
    )
    all_results.extend(batch_results)

print(json.dumps(all_results, indent=2, ensure_ascii=False))

# %%
pred_df = convert_llm_output_to_doccano(all_results)

# %%
print("\n" + "#" * 15 + " Running in 'discovery' mode (ignoring extra overlaps) " + "#" * 15)
results_discovery = evaluate(gold_df, pred_df, scenario='all_documents', eval_mode='discovery')
display_results(results_discovery, title_prefix="All Documents - 'discovery' mode")

print("\n" + "#" * 15 + " Running in 'coverage' mode (all overlaps are TPs) " + "#" * 15)
results_coverage = evaluate(gold_df, pred_df, scenario='all_documents', eval_mode='coverage')
display_results(results_coverage, title_prefix="All Documents - 'coverage' mode")

print("\n" + "#" * 15 + " Running in 'discovery' mode for Filtered Causal " + "#" * 15)
results_scenario_b = evaluate(gold_df, pred_df, scenario='filtered_causal', eval_mode='discovery')
display_results(results_scenario_b, title_prefix="Filtered Causal - 'discovery' mode")
print("\n" + "#" * 15 + " Running in 'coverage' mode for Filtered Causal " + "#" * 15)
results_scenario_b_coverage = evaluate(gold_df, pred_df, scenario='filtered_causal', eval_mode='coverage')
display_results(results_scenario_b_coverage, title_prefix="Filtered Causal - 'coverage' mode")
print("\n" + "#" * 15 + " Evaluation completed! " + "#" * 15)
# %%

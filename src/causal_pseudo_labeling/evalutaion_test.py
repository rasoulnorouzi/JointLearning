#%%
import llm_causal_metrics

gold_path = r"datasets\training_test_samples_gold\doccano_test.jsonl"
pred_path = r"weak_supervision/dataset/llama3_8b_output_test.jsonl"

# Evaluate scenario A
performance_A = llm_causal_metrics.evaluate(
    gold_path=gold_path,
    pred_path=pred_path,
    scenario_task2="A"
)
llm_causal_metrics.display_results(performance_A, "A")

# Evaluate scenario B
performance_B = llm_causal_metrics.evaluate(
    gold_path=gold_path,
    pred_path=pred_path,
    scenario_task2="B"
)
llm_causal_metrics.display_results(performance_B, "B")

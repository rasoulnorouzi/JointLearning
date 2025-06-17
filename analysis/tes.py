
def main():
    # Model configurations - UPDATE THESE PATHS TO YOUR ACTUAL MODEL PATHS
    models = {
        'bert-softmax': r"C:\Users\norouzin\Desktop\JointLearning\src\jointlearning\expert_bert_softmax\expert_bert_softmax_model.pt",
        'bert-gce': "path/to/bert-gce.pt",
        'bert-gce-softmax': "path/to/bert-gce-softmax.pt", 
        'bert-gce-freeze-softmax': "path/to/bert-gce-freeze-softmax.pt",
    }
    
    # Evaluation parameters
    thresholds = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    causal_classification_modes = ['cls+span', 'span_only']
    batch_size = 32
    scenarios = ['all_documents', 'filtered_causal']
    eval_modes = ['coverage', 'discovery']
    
    # For quick testing, you might want to use fewer configurations:
    # thresholds = [0.8, 0.9]
    # causal_classification_modes = ['span_only']
    


if __name__ == "__main__":
    main()
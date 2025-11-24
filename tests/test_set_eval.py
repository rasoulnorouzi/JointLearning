import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# --- Assuming these modules are in your src/jointlearning directory ---
# Make sure your PYTHONPATH is set correctly or run this script from
# a location where these imports work.
try:
    from model import JointCausalModel
    from dataset_collator import CausalDataset, CausalDatasetCollator
    from evaluate_joint_causal_model import evaluate_model, print_eval_report
    from config import (
        MODEL_CONFIG,
        id2label_cls,
        id2label_bio,
        id2label_rel,
        DATASET_CONFIG,
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(
        "Please ensure 'model.py', 'dataset_collator.py', 'evaluate_joint_causal_model.py',"
    )
    print(
        "and 'config.py' are in your Python path or the same directory."
    )
    exit()

# --- Configuration ---
MODEL_PATH = r"C:\Users\norouzin\Desktop\JointLearning\src\jointlearning\expert_bert_softmax\expert_bert_softmax_model.pt"
TEST_DATA_PATH = (
    r"C:\Users\norouzin\Desktop\JointLearning\datasets\expert_multi_task_data\test.csv"  # Path to your test CSV
)
TOKENIZER_NAME = MODEL_CONFIG.get(
    "encoder_name", "bert-base-uncased"
)  # Ensure this matches your trained model
BATCH_SIZE = 16  # Adjust as needed based on your resources
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")
print(f"Loading model from: {MODEL_PATH}")
print(f"Loading test data from: {TEST_DATA_PATH}")

# --- 1. Load the Model ---
try:
    # Instantiate the model using the configuration
    model = JointCausalModel()
    print("Model instantiated.")

    # Load the saved state dictionary
    # Use map_location to ensure it loads correctly whether you have a GPU or not
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)  # Move the model to the evaluation device
    model.eval()  # Set the model to evaluation mode
    print("Model weights loaded and set to evaluation mode.")

except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    exit()
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    exit()

# --- 2. Load the Test Dataset ---
try:
    test_df = pd.read_csv(TEST_DATA_PATH)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # Create the dataset
    test_dataset = CausalDataset(
        dataframe=test_df, tokenizer_name=TOKENIZER_NAME
    )

    # Create the collator
    collator = CausalDatasetCollator(tokenizer=tokenizer)

    # Create the DataLoader
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, collate_fn=collator, shuffle=False
    )
    print(f"Test dataset loaded with {len(test_dataset)} samples.")

except FileNotFoundError:
    print(f"Error: Test data file not found at {TEST_DATA_PATH}")
    exit()
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    exit()


# --- 3. Evaluate the Model ---
print("\nStarting evaluation...")
try:
    # Call the evaluation function (assuming it doesn't calculate loss)
    eval_metrics = evaluate_model(
        model=model,
        dataloader=test_dataloader,
        device=DEVICE,
        id2label_cls=id2label_cls,
        id2label_bio=id2label_bio,
        id2label_rel=id2label_rel,
    )
    print("Evaluation finished.")

    # --- 4. Print the Evaluation Report ---
    # We can use the print_eval_report function. Since it expects epoch info
    # and losses (which we don't have here), we'll pass placeholder values.
    print_eval_report(
        epoch_num=1,  # Placeholder
        num_epochs=1,  # Placeholder
        avg_train_loss=0.0,  # Placeholder
        avg_val_loss=0.0,  # Placeholder (or you could modify print_report)
        eval_results=eval_metrics,
        best_overall_f1=eval_metrics.get(
            "overall_avg_f1", 0.0
        ),  # Show current F1
        patience_counter=0,  # Placeholder
        patience_epochs=0,  # Placeholder
        new_best_saved=False,  # Placeholder
    )

except Exception as e:
    print(f"An error occurred during evaluation: {e}")
    # You might want to print traceback here for debugging
    # import traceback
    # traceback.print_exc()
    exit()

print("\nEvaluation script completed.")
# Uploading the Joint Causal Model to Hugging Face Hub

This guide explains how to upload the trained `JointCausalModel` to the Hugging Face Hub, making it compatible with the `AutoModel` API.

## Files in this Directory

*   `configuration_joint_causal.py`: Defines the model's configuration class, `JointCausalConfig`. This class holds the model's hyperparameters and architecture details.
*   `modeling_joint_causal.py`: Contains the implementation of the `JointCausalModel` itself.
*   `save_for_hf.py`: A script to save the trained model in a format compatible with the Hugging Face Hub and the `AutoModel` API.
*   `upload_to_hf.py`: A script to upload the saved model files to a repository on the Hugging Face Hub.
*   `automodel_test.py`: A script to test the uploaded model from the hub.
*   `config.py`: This file is not used.
*   `joint_causal_model_for_hf/`: This directory is created by `save_for_hf.py` and contains the saved model, tokenizer, and configuration files.

## Step-by-Step Guide to Upload Your Model

### Step 1: Save the Model in Hugging Face Format

The `save_for_hf.py` script is responsible for saving your trained model in the correct format.

**Nuances and Arguments:**

*   **`model_weights_path`**: Inside the script, you need to set the pytorch `model_weights_path` variable to the correct path of your trained model's weights file (`.pt` file). 
*   **`save_directory`**: This variable determines where the script will save the Hugging Face-compatible model files. The default is `"hf_port/joint_causal_model_for_hf"`.

**To run the script:**

```bash
python hf_port/save_for_hf.py
```

After running this script, the `hf_port/joint_causal_model_for_hf` directory will be created, containing the following files:

*   `config.json`: The model's configuration.
*   `model.safetensors`: The model's weights.
*   `modeling_joint_causal.py`: A copy of the model's implementation.
*   `configuration_joint_causal.py`: A copy of the model's configuration class.
*   `special_tokens_map.json`, `tokenizer_config.json`, `tokenizer.json`, `vocab.txt`: Tokenizer files.

### Step 2: Upload the Model to the Hugging Face Hub

The `upload_to_hf.py` script uploads the contents of the `joint_causal_model_for_hf` directory to a repository on the Hugging Face Hub.

**Nuances and Arguments:**

*   **`REPO_ID`**: **This is the most important variable to change.** You must replace `"rasoultilburg/SocioCausaNet"` with your own Hugging Face repository ID, in the format `"YourUsername/YourRepoName"`. You need to create this repository on the Hugging Face website first.
*   **`LOCAL_FOLDER`**: This variable points to the directory containing the files to be uploaded. The default is `"hf_port/joint_causal_model_for_hf"`, which is the output of the `save_for_hf.py` script.
*   **`commit_message`**: You can change the commit message to describe the version of the model you are uploading.

**Before running the script, you need to be logged into your Hugging Face account in your terminal:**

```bash
huggingface-cli login
```

You will be prompted to enter your Hugging Face API token.

**To run the script:**

```bash
python hf_port/upload_to_hf.py
```

### Step 3: Test the Uploaded Model

The `automodel_test.py` script shows how to load your model from the Hugging Face Hub using the `AutoModel` API.

**Nuances and Arguments:**

*   **`REPO_ID`**: You need to change the `REPO_ID` variable in this script to your repository ID.

**To run the script:**

```bash
python hf_port/automodel_test.py
```

This will download the model from the hub and run a simple test to ensure it's working correctly.

## Summary of Actions

1.  **Modify `save_for_hf.py`**: Set the correct `model_weights_path`.
2.  **Run `save_for_hf.py`**: This will save your model in the correct format.
3.  **Create a repository on the Hugging Face Hub.**
4.  **Modify `upload_to_hf.py`**: Set your `REPO_ID`.
5.  **Login to Hugging Face**: Use `huggingface-cli login`.
6.  **Run `upload_to_hf.py`**: This will upload your model.
7.  **Modify `automodel_test.py`**: Set your `REPO_ID`.
8.  **Run `automodel_test.py`**: This will test your uploaded model.

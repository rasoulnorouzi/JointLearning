from huggingface_hub import HfApi

# --- IMPORTANT ---
# Replace this with the repository ID you just created on the Hub.
# It should be in the format "YourUsername/YourRepoName".
REPO_ID = "rasoultilburg/SocioCausaNet"

# This is the local folder containing your model, tokenizer, and .py files.
LOCAL_FOLDER = "hf_port/joint_causal_model_for_hf"

# Create an instance of the HfApi client
api = HfApi()

print(f"Uploading files from '{LOCAL_FOLDER}' to '{REPO_ID}'...")

# Upload the entire folder to your repository
api.upload_folder(
    folder_path=LOCAL_FOLDER,
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="adding neural only mode for relation extraction",
)

print("\nUpload complete!")
print(f"Check out your model at: https://huggingface.co/{REPO_ID}")
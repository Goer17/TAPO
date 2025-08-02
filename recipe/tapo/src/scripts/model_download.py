from huggingface_hub import snapshot_download
import os

model_name = "Qwen/Qwen2.5-0.5B" # Or "Qwen/Qwen2.5-0.5B-Chat"
local_model_path = "./qwen2.5-0.5b_model" # Define your desired local path

# Create the directory if it doesn't exist
os.makedirs(local_model_path, exist_ok=True)

print(f"Downloading {model_name} to {local_model_path}...")
try:
    snapshot_download(
        repo_id=model_name,
        local_dir=local_model_path,
        local_dir_use_symlinks=False # Set to True if you prefer symlinks, False to download actual files
    )
    print(f"Model {model_name} downloaded successfully to {local_model_path}")
except Exception as e:
    print(f"Error downloading model {model_name}: {e}")
    print("Please ensure you have an internet connection, the model name is correct, and Git LFS is installed.")
    print("You might also need to log in to Hugging Face: `huggingface-cli login`")
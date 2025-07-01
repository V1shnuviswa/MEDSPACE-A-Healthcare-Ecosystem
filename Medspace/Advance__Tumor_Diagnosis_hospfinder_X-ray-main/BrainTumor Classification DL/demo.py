from huggingface_hub import HfApi

# Define your repo ID and paths
repo_id = "manoj2423/medi"  # Your Hugging Face repo
local_folder = "C:/Users/smano/OneDrive/Desktop/hyd3/token"  # Local token folder

# Initialize the API
api = HfApi()

# Upload `token/` folder into `model/checkpoint-14649/` in Hugging Face
api.upload_folder(
    folder_path=local_folder,
    repo_id=repo_id,
    repo_type="model",
    path_in_repo="model/checkpoint-14649"  # Upload inside checkpoint-14649
)

print("Upload complete!")

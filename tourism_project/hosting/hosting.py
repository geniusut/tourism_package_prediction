from huggingface_hub import HfApi
import os

# Replace with your actual Hugging Face dataset repo
#repo_id = "geniusut/tourism-package-prediction-project"
repo_id = "geniusut/tourism-project-package-prediction"
repo_type = "space"


api = HfApi(token=os.getenv("HF_TOKEN"))

#Upload
api.upload_folder(
    folder_path="tourism_project/deployment",     # the local folder containing your files
    repo_id=repo_id,          # the target repo
    repo_type=repo_type,                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)

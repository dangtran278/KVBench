from huggingface_hub import snapshot_download
import os

repositories = {
    "wikitext": "Salesforce/wikitext",
    "coqa": "stanfordnlp/coqa",
    "truthfulqa": "domenicrosati/TruthfulQA"
}

base_path = "./datasets"

for folder_name, repo_id in repositories.items():
    target_dir = os.path.join(base_path, folder_name)
    
    print(f"Downloading {repo_id} to {target_dir}")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            revision="main",
            repo_type="dataset"
        )
        print(f"Downloaded {folder_name}!\n")
    except Exception as e:
        print(f"Failed to download {repo_id}: {e}")

print("Finished")
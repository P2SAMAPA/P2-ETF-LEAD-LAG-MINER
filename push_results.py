"""
Push daily results to Hugging Face dataset.
"""
import json
import pandas as pd
from datetime import datetime
from huggingface_hub import HfApi, upload_file
import config
import tempfile
import os


def push_daily_result(results: dict):
    """Upload today's results to the output dataset."""
    api = HfApi(token=config.HF_TOKEN)

    # Create a timestamped file
    today = datetime.utcnow().strftime("%Y-%m-%d")
    filename = f"lead_lag_{today}.json"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(results, f, indent=2, default=str)
        temp_path = f.name

    try:
        upload_file(
            path_or_fileobj=temp_path,
            path_in_repo=filename,
            repo_id=config.HF_OUTPUT_DATASET,
            repo_type="dataset",
            token=config.HF_TOKEN,
        )
        print(f"Uploaded results to {config.HF_OUTPUT_DATASET}/{filename}")
    except Exception as e:
        print(f"Upload failed: {e}")
    finally:
        os.unlink(temp_path)


def load_latest_result() -> dict:
    """Load the most recent result file from the dataset."""
    api = HfApi(token=config.HF_TOKEN)
    files = api.list_repo_files(repo_id=config.HF_OUTPUT_DATASET, repo_type="dataset")
    json_files = [f for f in files if f.startswith("lead_lag_") and f.endswith(".json")]
    if not json_files:
        return {}
    json_files.sort(reverse=True)
    latest_file = json_files[0]

    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id=config.HF_OUTPUT_DATASET,
        filename=latest_file,
        repo_type="dataset",
        token=config.HF_TOKEN,
    )
    with open(path, "r") as f:
        return json.load(f)

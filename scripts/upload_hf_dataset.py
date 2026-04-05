from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create and upload a dataset repository to the Hugging Face Hub.")
    parser.add_argument("--repo-id", required=True, help="Dataset repo id, for example username/my-dataset")
    parser.add_argument("--folder", default="hf_dataset_repo", help="Local folder to upload")
    parser.add_argument("--private", action="store_true", help="Create the repo as private")
    parser.add_argument("--token", default=None, help="Optional HF token. Falls back to HF_TOKEN or HUGGINGFACE_HUB_TOKEN")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = args.token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    from huggingface_hub import HfApi, create_repo

    create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
        token=token,
    )

    api = HfApi(token=token)
    api.upload_folder(
        folder_path=str(folder),
        repo_id=args.repo_id,
        repo_type="dataset",
    )

    visibility = "private" if args.private else "public"
    print(f"Uploaded {folder} to https://huggingface.co/datasets/{args.repo_id} ({visibility})")


if __name__ == "__main__":
    main()

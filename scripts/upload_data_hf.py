from __future__ import annotations

"""
Upload the data/ directory to a HuggingFace dataset repository.

Usage:
    pip install huggingface_hub
    huggingface-cli login
    python scripts/upload_data_hf.py --repo <your-hf-username>/<repo-name>
"""

import argparse

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True,
                        help="HuggingFace repo id, e.g. username/microscopy-sr-data")
    parser.add_argument("--data_dir", default="data",
                        help="Local data directory to upload (default: data/)")
    parser.add_argument("--private", action="store_true", default=True,
                        help="Make the repository private (default: True)")
    parser.add_argument("--token", default=None,
                        help="HuggingFace API token (alternative to huggingface-cli login)")
    args = parser.parse_args()

    api = HfApi(token=args.token)

    # Create repo if it doesn't exist
    api.create_repo(repo_id=args.repo, repo_type="dataset",
                    private=args.private, exist_ok=True)
    print(f"Repository: https://huggingface.co/datasets/{args.repo}")

    print(f"Uploading {args.data_dir}/ ...")
    api.upload_folder(
        folder_path=args.data_dir,
        repo_id=args.repo,
        repo_type="dataset",
        ignore_patterns=["*.DS_Store"],
    )
    print("Done.")


if __name__ == "__main__":
    main()

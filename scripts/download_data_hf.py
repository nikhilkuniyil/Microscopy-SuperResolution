from __future__ import annotations

"""
Download the dataset from HuggingFace onto a remote machine (e.g. DataHub).

Usage:
    pip install huggingface_hub
    huggingface-cli login          # only needed if repo is private
    python scripts/download_data_hf.py --repo <your-hf-username>/<repo-name>
"""

import argparse

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True,
                        help="HuggingFace repo id, e.g. username/microscopy-sr-data")
    parser.add_argument("--dest", default="data",
                        help="Local destination directory (default: data/)")
    parser.add_argument("--token", default=None,
                        help="HuggingFace API token (alternative to huggingface-cli login)")
    args = parser.parse_args()

    print(f"Downloading {args.repo} -> {args.dest}/ ...")
    snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        local_dir=args.dest,
        token=args.token,
        ignore_patterns=["*.gitattributes", ".gitattributes"],
    )
    print("Done.")


if __name__ == "__main__":
    main()

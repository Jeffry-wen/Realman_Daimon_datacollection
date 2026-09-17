#!/usr/bin/env python3
"""Upload a local dataset into a subfolder of a Hugging Face dataset repo.

The access token is read from stdin so it is not placed in the command line,
process list, shell history, or logs.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

from huggingface_hub import HfApi


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--folder-path", type=Path, required=True)
    parser.add_argument("--path-in-repo", required=True)
    parser.add_argument("--commit-message", required=True)
    args = parser.parse_args()

    token = (
        getpass.getpass("Hugging Face token: ")
        if sys.stdin.isatty()
        else sys.stdin.readline().strip()
    )
    if not token:
        raise RuntimeError("No Hugging Face token received on stdin")

    api = HfApi(token=token)
    identity = api.whoami(token=token)
    print(f"Authenticated as {identity.get('name', '<unknown>')}", flush=True)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        exist_ok=True,
        token=token,
    )
    print(
        f"Uploading {args.folder_path} to {args.repo_id}/{args.path_in_repo}",
        flush=True,
    )
    result = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=args.folder_path,
        path_in_repo=args.path_in_repo,
        commit_message=args.commit_message,
        ignore_patterns=[
            ".aligned_recorder_snapshots/**",
            "**/.aligned_recorder_snapshots/**",
            "**/__pycache__/**",
            "**/.DS_Store",
        ],
        token=token,
    )
    token = ""
    print(f"Commit: {result.commit_url}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Safely remove only trailing LeRobot episodes and rebuild aggregate metadata."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.io_utils import write_info, write_stats
from lerobot.datasets.utils import unflatten_dict


def _episode_stats(path: Path) -> dict[str, dict[str, np.ndarray]]:
    row = pq.read_table(path).to_pydict()
    flat = {
        key.removeprefix("stats/"): np.asarray(value[0])
        for key, value in row.items()
        if key.startswith("stats/")
    }
    return unflatten_dict(flat)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--first-delete", type=int, required=True)
    args = parser.parse_args()

    root = args.dataset_root.expanduser().resolve()
    meta = LeRobotDatasetMetadata("local/tail-edit", root=root)
    first_delete = args.first_delete
    total = int(meta.total_episodes)
    if not 0 < first_delete < total:
        raise ValueError(f"first-delete must be in 1..{total - 1}, got {first_delete}")

    # This utility deliberately supports tail deletion only; that preserves all
    # existing indices and lets DatasetWriter.resume append without re-encoding.
    removed = list(range(first_delete, total))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    quarantine = root.parent / f".{root.name}.deleted-{first_delete}-{total - 1}-{stamp}"
    quarantine.mkdir(parents=True)

    paths: set[Path] = set()
    for episode_index in removed:
        paths.add(meta.get_data_file_path(episode_index))
        episode = meta.episodes[episode_index]
        chunk = int(episode["meta/episodes/chunk_index"])
        file = int(episode["meta/episodes/file_index"])
        paths.add(Path(f"meta/episodes/chunk-{chunk:03d}/file-{file:03d}.parquet"))
        paths.add(Path(f"meta/episode_start_poses/episode-{episode_index:06d}.json"))
        for video_key in meta.video_keys:
            paths.add(meta.get_video_file_path(episode_index, video_key))

    snapshot_root = root / ".aligned_recorder_snapshots"
    if snapshot_root.exists():
        for snapshot in snapshot_root.glob("episode-*"):
            paths.add(snapshot.relative_to(root))

    moved: list[str] = []
    for relative in sorted(paths):
        source = root / relative
        if not source.exists():
            continue
        destination = quarantine / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        moved.append(str(relative))

    kept_episode_files = [
        root / f"meta/episodes/chunk-000/file-{index:03d}.parquet"
        for index in range(first_delete)
    ]
    missing = [str(path) for path in kept_episode_files if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing kept episode metadata: {missing[:3]}")

    lengths = [int(pq.read_table(path, columns=["length"])["length"][0].as_py()) for path in kept_episode_files]
    stats = aggregate_stats([_episode_stats(path) for path in kept_episode_files])
    info = dict(meta.info)
    info["total_episodes"] = first_delete
    info["total_frames"] = sum(lengths)
    info["splits"] = {"train": f"0:{first_delete}"}
    write_info(info, root)
    write_stats(stats, root)

    latest_pose = root / "meta/episode_start_poses" / f"episode-{first_delete - 1:06d}.json"
    if latest_pose.is_file():
        payload = json.loads(latest_pose.read_text(encoding="utf-8"))
        payload["source"] = str(latest_pose.relative_to(root))
        (root / "meta/inference_start_pose.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    manifest = {
        "dataset_root": str(root),
        "removed_episode_indices": removed,
        "kept_episodes": first_delete,
        "kept_frames": sum(lengths),
        "moved_paths": moved,
    }
    (quarantine / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**manifest, "quarantine": str(quarantine)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

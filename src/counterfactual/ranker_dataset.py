from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.counterfactual.candidate_ranker import phase_one_hot


@dataclass
class RankerStats:
    privileged_mean: np.ndarray
    privileged_std: np.ndarray
    temporal_mean: np.ndarray
    temporal_std: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key).tolist() for key in self.__annotations__}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RankerStats":
        return cls(**{key: np.asarray(value[key], np.float32) for key in cls.__annotations__})


def _privileged(feature: dict[str, Any]) -> np.ndarray:
    return np.concatenate([feature["simulator_qpos"], feature["simulator_qvel"],
                           feature["target_orientation"]]).astype(np.float32)


def _temporal(feature: dict[str, Any]) -> np.ndarray:
    return np.concatenate([feature["frozen_visual_latent_history"],
                           feature["proprio_history"], feature["previous_base_actions"],
                           feature["previous_executed_actions"]], axis=-1).astype(np.float32)


def load_candidate_data(path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data["root_features"], data["records"], data["metadata"]


def assert_source_split_isolated(features: list[dict[str, Any]]) -> None:
    train = {x["source_episode_id"] for x in features if x["split"] == "train"}
    dev = {x["source_episode_id"] for x in features if x["split"] == "dev"}
    overlap = train & dev
    if overlap:
        raise AssertionError(f"Source trajectories leak across train/dev: {sorted(overlap)}")


def training_stats(features: list[dict[str, Any]]) -> RankerStats:
    train = [x for x in features if x["split"] == "train"]
    privileged = np.stack([_privileged(x) for x in train])
    temporal = np.concatenate([_temporal(x) for x in train], axis=0)
    return RankerStats(privileged.mean(0), np.maximum(privileged.std(0), 1e-5),
                       temporal.mean(0), np.maximum(temporal.std(0), 1e-5))


class CandidateDataset(Dataset):
    def __init__(self, features: list[dict[str, Any]], records: list[dict[str, Any]],
                 split: str, stats: RankerStats):
        feature_by_id = {x["root_id"]: x for x in features}
        self.rows: list[dict[str, Any]] = []
        for root_index, root in enumerate(records):
            if root["split"] != split:
                continue
            feature = feature_by_id[root["root_id"]]
            privileged = (_privileged(feature)-stats.privileged_mean)/stats.privileged_std
            temporal = (_temporal(feature)-stats.temporal_mean)/stats.temporal_std
            for candidate in root["candidates"]:
                label = candidate["labels"]
                self.rows.append({
                    "root_index": root_index, "root_id": root["root_id"],
                    "root_type": root["root_type"],
                    "candidate_id": int(candidate["candidate_id"]),
                    "privileged": privileged, "phase": phase_one_hot(root["root_type"]),
                    "temporal": temporal,
                    "descriptor": np.asarray(candidate["descriptor"], np.float32).reshape(-1),
                    "strong_gain": float(label["strong_gain"]), "harm": float(label["harm"]),
                    "progress": np.asarray(label["progress_delta"], np.float32)/20.0,
                    "mask": float(label["mask"]),
                })

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {key: torch.as_tensor(value) if key not in ("root_id", "root_type") else value
                for key, value in row.items()}

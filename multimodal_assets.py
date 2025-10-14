"""Dataloaders for asset-backed multimodal survival training.

This module exposes :class:`AssetBackedMultiModalDataset` which keeps the
tabular modality in-memory while streaming image and sensor assets from disk
per batch.  The dataset returns a dictionary of modality tensors so that the
model can apply learnable encoders inside the computational graph, enabling
end-to-end optimisation across all modalities.

Utility helpers are provided to analyse manifests, resolve asset paths, and
collate batches into tensors suitable for the upgraded multimodal MySA model.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

try:  # torchvision is optional during documentation builds
    from torchvision import transforms
except Exception as exc:  # pragma: no cover - torchvision is required at runtime
    raise RuntimeError(
        "torchvision is required for the asset-backed multimodal pipeline. "
        "Install it with `pip install torchvision`."
    ) from exc


def _resolve_asset_paths(
    manifest: pd.DataFrame,
    *,
    id_col: str,
    path_col: str,
    root: str,
) -> Dict[str, str]:
    """Map identifier → absolute asset path.

    Paths inside the manifest may be absolute or relative to ``root``.  The
    resolver normalises both styles and keeps only entries that resolve to
    existing files.
    """

    lookup: Dict[str, str] = {}
    norm_root = os.path.abspath(root)
    for _, row in manifest.iterrows():
        key = str(row[id_col]).strip()
        if not key:
            continue
        raw_path = str(row[path_col]).strip()
        if not raw_path:
            continue
        if os.path.isabs(raw_path):
            abs_path = raw_path
        else:
            abs_path = os.path.normpath(os.path.join(norm_root, raw_path))
        if os.path.exists(abs_path):
            lookup[key] = abs_path
    return lookup


def _find_sensor_columns(path: str) -> Tuple[List[str], int]:
    """Return the numeric columns and sequence length of the first sensor file."""

    df = pd.read_csv(path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return [], 0
    length = int(len(df))
    return numeric_cols, length


def _resample_sequence(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Resample a ``[C, L]`` array to ``target_len`` using linear interpolation."""

    if target_len <= 0 or arr.shape[1] == target_len:
        return arr
    if arr.shape[1] == 0:
        return np.zeros((arr.shape[0], target_len), dtype=arr.dtype)
    old_idx = np.linspace(0.0, 1.0, arr.shape[1], dtype=np.float32)
    new_idx = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    out = np.vstack([np.interp(new_idx, old_idx, channel) for channel in arr])
    return out.astype(arr.dtype, copy=False)


@dataclass
class SensorSpec:
    columns: Sequence[str]
    target_len: int

    @property
    def channels(self) -> int:
        return len(self.columns)


class AssetBackedMultiModalDataset(Dataset):
    """Dataset that streams raw assets while keeping supervision tensors in RAM."""

    def __init__(
        self,
        *,
        ids: Sequence[str],
        tabular: np.ndarray,
        labels: np.ndarray,
        masks: np.ndarray,
        modality_mask: np.ndarray,
        tabular_names: Sequence[str],
        image_paths: Optional[Sequence[Optional[str]]] = None,
        image_size: Tuple[int, int] = (224, 224),
        sensor_paths: Optional[Sequence[Optional[str]]] = None,
        sensor_spec: Optional[SensorSpec] = None,
    ) -> None:
        if tabular.shape[0] != len(ids):
            raise ValueError("Tabular matrix and id list must align in length.")

        self.ids = [str(i) for i in ids]
        self.tabular = torch.from_numpy(tabular.astype(np.float32, copy=False))
        self.labels = torch.from_numpy(labels.astype(np.float32, copy=False))
        self.masks = torch.from_numpy(masks.astype(np.float32, copy=False))
        self.modality_mask = torch.from_numpy(modality_mask.astype(np.float32, copy=False))
        self.tabular_names = list(tabular_names)

        self.modalities: List[str] = ["tabular"]
        self.image_paths: Optional[List[Optional[str]]] = None
        self.sensor_paths: Optional[List[Optional[str]]] = None
        self.sensor_spec = sensor_spec

        if image_paths is not None:
            self.image_paths = [p if p and os.path.exists(p) else None for p in image_paths]
            self.modalities.append("image")
            self._img_transform = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            C, H, W = 3, image_size[0], image_size[1]
            self._img_zero = torch.zeros(C, H, W, dtype=torch.float32)
        else:
            self._img_transform = None
            self._img_zero = None

        if sensor_paths is not None and sensor_spec is not None and sensor_spec.channels > 0:
            self.sensor_paths = [p if p and os.path.exists(p) else None for p in sensor_paths]
            self.modalities.append("sensor")
            self._sensor_zero = torch.zeros(
                sensor_spec.channels, sensor_spec.target_len, dtype=torch.float32
            )
        else:
            self._sensor_zero = None

    # ------------------------------------------------------------------
    def __len__(self) -> int:  # type: ignore[override]
        return len(self.ids)

    # ------------------------------------------------------------------
    def _load_image(self, path: str) -> torch.Tensor:
        with Image.open(path) as im:
            im = im.convert("RGB")
        if self._img_transform is not None:
            return self._img_transform(im)
        return transforms.ToTensor()(im)

    # ------------------------------------------------------------------
    def _load_sensor(self, path: str) -> torch.Tensor:
        assert self.sensor_spec is not None
        df = pd.read_csv(path)
        cols = [c for c in self.sensor_spec.columns if c in df.columns]
        if not cols:
            return self._sensor_zero.clone() if self._sensor_zero is not None else torch.zeros(0)
        arr = df[cols].to_numpy(dtype=np.float32).T  # [C, L]
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = _resample_sequence(arr, self.sensor_spec.target_len)
        return torch.from_numpy(arr)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):  # type: ignore[override]
        modalities: Dict[str, torch.Tensor] = {
            "tabular": self.tabular[idx],
        }

        if self.image_paths is not None:
            path = self.image_paths[idx]
            if path is None:
                modalities["image"] = self._img_zero.clone() if self._img_zero is not None else torch.zeros(3, 224, 224)
            else:
                try:
                    modalities["image"] = self._load_image(path)
                except Exception:
                    modalities["image"] = self._img_zero.clone() if self._img_zero is not None else torch.zeros(3, 224, 224)

        if self.sensor_paths is not None and self.sensor_spec is not None:
            path = self.sensor_paths[idx]
            if path is None:
                modalities["sensor"] = self._sensor_zero.clone() if self._sensor_zero is not None else torch.zeros(self.sensor_spec.channels, self.sensor_spec.target_len)
            else:
                try:
                    modalities["sensor"] = self._load_sensor(path)
                except Exception:
                    modalities["sensor"] = self._sensor_zero.clone() if self._sensor_zero is not None else torch.zeros(self.sensor_spec.channels, self.sensor_spec.target_len)

        sample = {
            "modalities": modalities,
            "y": self.labels[idx],
            "m": self.masks[idx],
            "modality_mask": self.modality_mask[idx],
        }
        return sample

    # ------------------------------------------------------------------
    def collate_fn(self, batch: Sequence[Dict[str, torch.Tensor]]):
        keys = batch[0]["modalities"].keys()
        collated_modalities = {
            k: torch.stack([sample["modalities"][k] for sample in batch]) for k in keys
        }
        y = torch.stack([sample["y"] for sample in batch])
        m = torch.stack([sample["m"] for sample in batch])
        mask = torch.stack([sample["modality_mask"] for sample in batch])
        return collated_modalities, y, m, mask


def build_image_paths_for_ids(
    ids: Sequence[str],
    *,
    manifest: pd.DataFrame,
    id_col: str,
    path_col: str,
    root: str,
) -> List[Optional[str]]:
    lookup = _resolve_asset_paths(manifest, id_col=id_col, path_col=path_col, root=root)
    return [lookup.get(str(i)) for i in ids]


def build_sensor_paths_for_ids(
    ids: Sequence[str],
    *,
    manifest: pd.DataFrame,
    id_col: str,
    path_col: str,
    root: str,
) -> Tuple[List[Optional[str]], Optional[SensorSpec]]:
    lookup = _resolve_asset_paths(manifest, id_col=id_col, path_col=path_col, root=root)
    paths = [lookup.get(str(i)) for i in ids]
    existing = [p for p in paths if p]
    if not existing:
        return paths, None
    cols, length = _find_sensor_columns(existing[0])
    if not cols:
        return paths, None
    target_len = min(512, length) if length > 0 else 256
    return paths, SensorSpec(columns=cols, target_len=target_len)


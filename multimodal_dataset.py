import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional


class MultiModalDataset(Dataset):
    """Dataset that stores multiple modalities and their availability masks.

    Parameters
    ----------
    modalities: Dict[str, np.ndarray]
        Mapping from modality name to array of shape [N, ...]. All modalities
        must share the same number of samples ``N``.
    labels: np.ndarray
        Event labels in one-hot encoding of shape [N, T].
    masks: np.ndarray
        Risk-set masks of shape [N, T].
    modality_available: Optional[np.ndarray]
        Boolean array [N, M] indicating which modalities are present for each
        sample. If ``None`` we assume every modality is available.
    modality_dropout: float, optional
        Probability of randomly dropping a modality during training to improve
        robustness.
    train: bool
        Whether this dataset will be used for training (enables dropout).
    """

    def __init__(
        self,
        modalities: Dict[str, np.ndarray],
        labels: np.ndarray,
        masks: np.ndarray,
        modality_available: Optional[np.ndarray] = None,
        modality_dropout: float = 0.0,
        train: bool = True,
    ) -> None:
        self.modalities = {k: torch.from_numpy(v).float() for k, v in modalities.items()}
        self.labels = torch.from_numpy(labels).float()
        self.masks = torch.from_numpy(masks).float()
        self.modality_names: List[str] = list(modalities.keys())
        self.train = train
        self.modality_dropout = float(modality_dropout)

        N = self.labels.shape[0]
        M = len(self.modality_names)
        if modality_available is None:
            modality_available = np.ones((N, M), dtype=np.float32)
        self.modality_available = torch.from_numpy(modality_available).float()

    def __len__(self) -> int:  # type: ignore[override]
        return self.labels.shape[0]

    def __getitem__(self, idx: int):  # type: ignore[override]
        x = {k: v[idx] for k, v in self.modalities.items()}
        mod_mask = self.modality_available[idx].clone()

        if self.train and self.modality_dropout > 0.0:
            drop = torch.rand_like(mod_mask) < self.modality_dropout
            mod_mask = mod_mask * (~drop).float()

        return x, self.labels[idx], self.masks[idx], mod_mask

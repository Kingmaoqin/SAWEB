"""TEXGI-specific counterfactual generation.

This module performs small, gradient-based feature updates on top of the
trained TEXGISA/MySA tabular model to lower per-patient cumulative hazard.
It targets a user-requested survival extension and surfaces the top feature
changes with estimated hazard reductions.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from algorithm.CF import _target_cumhaz
from model import MultiTaskModel


@dataclass
class TexgiCFResult:
    table: pd.DataFrame
    summary: Dict[str, Any]


def _to_feature_frame(features: Iterable, feature_names: Sequence[str]) -> pd.DataFrame:
    if isinstance(features, pd.DataFrame):
        return features.reset_index(drop=True)[list(feature_names)]
    arr = np.asarray(features, dtype=float)
    return pd.DataFrame(arr, columns=list(feature_names))


def _stats_frame(stats: Optional[pd.DataFrame], feature_names: Sequence[str]) -> pd.DataFrame:
    if stats is None:
        return pd.DataFrame(index=feature_names)
    if isinstance(stats, pd.DataFrame):
        if "feature" in stats.columns:
            return stats.set_index("feature").reindex(feature_names)
        return stats.reindex(feature_names)
    if isinstance(stats, dict):
        df = pd.DataFrame(stats)
        if "feature" in df.columns:
            df = df.set_index("feature")
        return df.reindex(feature_names)
    return pd.DataFrame(index=feature_names)


def _load_model(spec: Dict[str, Any]) -> MultiTaskModel:
    model = MultiTaskModel(
        input_dim=int(spec["input_dim"]),
        num_bins=int(spec["num_bins"]),
        hidden=int(spec.get("hidden", 256)),
        depth=int(spec.get("depth", 2)),
        dropout=float(spec.get("dropout", 0.2)),
    )
    state_path = spec.get("state_path")
    if state_path:
        state = torch.load(state_path, map_location="cpu")
        model.load_state_dict(state)
    model.eval()
    return model


def _optimize_single(
    model: MultiTaskModel,
    base: torch.Tensor,
    bounds_min: torch.Tensor,
    bounds_max: torch.Tensor,
    target_ch: float,
    steps: int = 160,
    lr: float = 0.05,
    l2_weight: float = 0.01,
    frozen_mask: Optional[torch.Tensor] = None,
):
    base = base.clone().detach()
    with torch.no_grad():
        base.clamp_(bounds_min, bounds_max)
    x = base.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)

    best = x.clone().detach()
    with torch.no_grad():
        haz0 = model(x).sum().item()
    best_ch = haz0

    for _ in range(max(1, steps)):
        optimizer.zero_grad()
        hazards = model(x)
        ch = hazards.sum()
        loss_hazard = torch.relu(ch - target_ch)
        loss_l2 = l2_weight * torch.mean((x - base) ** 2)
        loss = loss_hazard + loss_l2
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            x.clamp_(bounds_min, bounds_max)
            if frozen_mask is not None and frozen_mask.any():
                x[..., frozen_mask] = base[..., frozen_mask]

            ch_post = model(x).sum()
            if ch_post.item() < best_ch:
                best_ch = ch_post.item()
                best = x.clone().detach()
            if ch_post.item() <= target_ch * 1.02:
                break

    with torch.no_grad():
        final_haz = model(best).detach().sum().item()
    return best, haz0, final_haz


def generate_texgi_counterfactuals(
    model_spec: Dict[str, Any],
    features: Iterable,
    *,
    hazards: Optional[Iterable[Iterable[float]]] = None,
    feature_stats: Optional[pd.DataFrame | Dict[str, Any]] = None,
    patient_indices: Optional[Sequence[int]] = None,
    desired_extension: float = 1.0,
    steps: int = 160,
    lr: float = 0.05,
    top_k: int = 3,
    immutable_features: Optional[Sequence[str]] = None,
) -> TexgiCFResult:
    """Generate per-patient counterfactuals with TEXGI gradients.

    The function reloads the trained MultiTaskModel, performs constrained
    gradient updates on the selected patient feature vectors, and reports the
    top feature changes that reduce cumulative hazard toward the target level
    implied by ``desired_extension``.
    """

    feature_names = list(model_spec.get("feature_names", []))
    feat_df = _to_feature_frame(features, feature_names)
    idx_subset = list(patient_indices) if patient_indices is not None else list(range(len(feat_df)))
    idx_subset = [i for i in idx_subset if 0 <= i < len(feat_df)]
    if not idx_subset:
        return TexgiCFResult(pd.DataFrame(), {"error": "No valid patients provided."})

    stats_df = _stats_frame(feature_stats, feature_names)
    bounds_min = torch.tensor(stats_df.get("min", pd.Series(0, index=feature_names)).to_numpy(), dtype=torch.float32)
    bounds_max = torch.tensor(stats_df.get("max", pd.Series(0, index=feature_names)).to_numpy(), dtype=torch.float32)

    immutable_features = set(immutable_features or [])
    frozen_mask = torch.tensor([name in immutable_features for name in feature_names], dtype=torch.bool)

    model = _load_model(model_spec)
    num_bins = int(model_spec.get("num_bins", model.hazard_layer.out_features))

    hazards_arr: Optional[np.ndarray] = None
    if hazards is not None:
        haz_np = torch.as_tensor(hazards, dtype=torch.float32).cpu().numpy()
        hazards_arr = haz_np if haz_np.ndim == 2 else haz_np[:, None]

    suggestions = []
    for idx in idx_subset:
        base_vec = torch.tensor(feat_df.iloc[idx].to_numpy(dtype=float), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            base_haz = model(base_vec).sum().item() if hazards_arr is None else float(hazards_arr[idx].sum())

        target_ch = _target_cumhaz(base_haz, horizon=num_bins, desired_extension=desired_extension)
        best_vec, base_ch, achieved_ch = _optimize_single(
            model,
            base_vec,
            bounds_min,
            bounds_max,
            target_ch,
            steps=steps,
            lr=lr,
            frozen_mask=frozen_mask,
        )

        delta = (best_vec - base_vec).squeeze(0).detach().cpu().numpy()
        base_arr = base_vec.squeeze(0).detach().cpu().numpy()
        best_arr = best_vec.squeeze(0).detach().cpu().numpy()
        order = np.argsort(np.abs(delta))[::-1]

        for rank, feat_idx in enumerate(order[: max(1, top_k)], start=1):
            if frozen_mask[feat_idx]:
                continue
            if np.isclose(delta[feat_idx], 0.0):
                continue
            feat_name = feature_names[feat_idx]
            suggestions.append(
                {
                    "sample_id": idx,
                    "suggestion_rank": rank,
                    "feature": feat_name,
                    "current_value": float(base_arr[feat_idx]),
                    "suggested_value": float(best_arr[feat_idx]),
                    "delta": float(delta[feat_idx]),
                    "current_cumhaz": float(base_ch),
                    "target_cumhaz": float(target_ch),
                    "achieved_cumhaz": float(achieved_ch),
                    "estimated_extension": float(num_bins * max(base_ch / max(achieved_ch, 1e-6) - 1.0, 0.0)),
                }
            )

    table = pd.DataFrame(suggestions)
    summary = {
        "desired_extension": desired_extension,
        "mean_current_cumhaz": float(table["current_cumhaz"].mean()) if not table.empty else 0.0,
        "mean_target_cumhaz": float(table["target_cumhaz"].mean()) if not table.empty else 0.0,
        "mean_achieved_cumhaz": float(table["achieved_cumhaz"].mean()) if not table.empty else 0.0,
    }
    return TexgiCFResult(table=table, summary=summary)


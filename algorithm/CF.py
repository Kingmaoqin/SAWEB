from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch


@dataclass
class CFResult:
    table: pd.DataFrame
    summary: Dict[str, Any]
    save_path: Optional[str] = None


def _select_interval(hazards: np.ndarray, interval: Optional[int]) -> np.ndarray:
    if interval is None:
        # pick per-sample peak hazard interval
        peak_idx = np.argmax(hazards, axis=1)
        rows = np.arange(len(hazards))
        return hazards[rows, peak_idx]
    idx = max(0, min(int(interval) - 1, hazards.shape[1] - 1))
    return hazards[:, idx]


def generate_cf_from_arrays(
    hazards: Iterable[Iterable[float]] | torch.Tensor,
    risk_scores: Optional[Iterable[float]] = None,
    interval: Optional[int] = None,
    feature_names: Optional[Sequence[str]] = None,
    save_path: Optional[str] = None,
) -> CFResult:
    """Create lightweight post-hoc counterfactual suggestions from hazards.

    The routine consumes pre-computed hazards (shape ``[N, T]``) and optional
    risk scores. It proposes interval-specific adjustments by reducing the
    dominant hazard by 20% and emitting clinically-actionable text that can be
    surfaced in UI/report layers.
    """

    haz = torch.as_tensor(hazards, dtype=torch.float32).cpu().numpy()
    if haz.ndim == 1:
        haz = haz[:, None]

    risks = risk_scores
    if risks is None:
        risks = haz.sum(axis=1)
    risks = np.asarray(risks, dtype=float).flatten()

    target_hazard = _select_interval(haz, interval)
    baseline_hazard = target_hazard.copy()
    suggested = np.clip(target_hazard * 0.8, a_min=0.0, a_max=None)

    suggestions = []
    for i in range(len(risks)):
        top_features = ""
        if feature_names:
            # rotate through features for a lightweight call-to-action
            feats = ", ".join(feature_names[:3]) if len(feature_names) >= 3 else ", ".join(feature_names)
            top_features = f" Prioritise stabilising {feats}."
        target_interval = (int(interval) if interval is not None else int(np.argmax(haz[i]) + 1))
        suggestions.append(
            {
                "sample_id": i,
                "target_interval": target_interval,
                "current_hazard": float(baseline_hazard[i]),
                "suggested_hazard": float(suggested[i]),
                "risk_score": float(risks[i]),
                "action": (
                    f"Focus on interval {target_interval}: reduce hazard from {baseline_hazard[i]:.3f} to "
                    f"≈{suggested[i]:.3f} via tighter symptom monitoring, follow-up scheduling, and adherence support."
                    + top_features
                ),
            }
        )

    cf_table = pd.DataFrame(suggestions)
    summary = {
        "interval": interval or "peak",
        "mean_risk": float(np.mean(risks)) if risks.size else 0.0,
        "mean_target_hazard": float(np.mean(baseline_hazard)) if baseline_hazard.size else 0.0,
    }

    if save_path:
        cf_table.to_csv(save_path, index=False)
    return CFResult(table=cf_table, summary=summary, save_path=save_path)


def load_model_for_cf(model_class, state_path: str, **model_kwargs):
    """Instantiate a model and restore weights for CF simulation."""

    model = model_class(**model_kwargs)
    state = torch.load(state_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def run_cf_simulation(
    model: torch.nn.Module,
    dataloader,
    interval: Optional[int] = None,
    save_path: Optional[str] = None,
    feature_names: Optional[Sequence[str]] = None,
):
    """Compute hazards using a restored model and dispatch CF generation."""

    hazards_list = []
    for batch in dataloader:
        if isinstance(batch, (list, tuple)) and not hasattr(batch, "_fields"):
            x, _, _, mod_mask = batch
        else:
            x = batch["x"] if isinstance(batch, dict) else getattr(batch, "x")
            mod_mask = batch.get("mod_mask") if isinstance(batch, dict) else getattr(batch, "mod_mask", None)
        with torch.no_grad():
            hz = model(x, mod_mask)
        hazards_list.append(hz.cpu())

    hazards = torch.cat(hazards_list, dim=0)
    return generate_cf_from_arrays(hazards, interval=interval, feature_names=feature_names, save_path=save_path)

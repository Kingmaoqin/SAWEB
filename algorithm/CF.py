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
        return hazards[rows, peak_idx], peak_idx
    idx = max(0, min(int(interval) - 1, hazards.shape[1] - 1))
    return hazards[:, idx], np.full(hazards.shape[0], idx, dtype=int)


def _target_cumhaz(cumhaz: float, horizon: int, desired_extension: float) -> float:
    """Map a desired survival extension (in intervals/time units) to a target cumulative hazard."""

    if desired_extension <= 0 or horizon <= 0:
        return cumhaz
    scale = horizon / float(horizon + desired_extension)
    return max(cumhaz * scale, 0.0)


def _binary_scale_search(
    haz_row: np.ndarray, target_ch: float, interval: Optional[int] = None
) -> np.ndarray:
    """Deterministically shrink hazards to hit the target cumulative hazard."""

    haz_adj = haz_row.copy()
    base_ch = float(haz_adj.sum())
    if base_ch <= 0:
        return haz_adj

    if target_ch >= base_ch:
        # Already safe enough; keep as-is
        return haz_adj

    if interval is None:
        scale = max(target_ch / base_ch, 0.0)
        return np.clip(haz_adj * scale, a_min=0.0, a_max=None)

    idx = max(0, min(int(interval), haz_adj.shape[0] - 1))
    other_ch = float(base_ch - haz_adj[idx])
    if haz_adj[idx] <= 0:
        return haz_adj

    # Solve for scale on the chosen interval to meet the target cumulative hazard
    target_for_idx = max(target_ch - other_ch, 0.0)
    if target_for_idx <= 0:
        haz_adj[idx] = 0.0
        return haz_adj

    lo, hi = 0.0, 1.0
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        cand = other_ch + haz_adj[idx] * mid
        if cand > target_ch:
            hi = mid
        else:
            lo = mid
    haz_adj[idx] = haz_adj[idx] * lo
    return haz_adj


def _genetic_search(
    haz_row: np.ndarray, target_ch: float, n_gen: int = 18, pop_size: int = 32
) -> np.ndarray:
    """Fallback genetic search when deterministic scaling cannot satisfy constraints."""

    rng = np.random.default_rng()
    T = haz_row.shape[0]
    pop = rng.uniform(0.0, 1.0, size=(pop_size, T))

    def fitness(scales: np.ndarray) -> np.ndarray:
        adjusted = haz_row[None, :] * scales
        cum = adjusted.sum(axis=1)
        # Penalize exceeding the target and deviation from it
        penalty = np.maximum(cum - target_ch, 0.0)
        return np.abs(cum - target_ch) + 5.0 * penalty

    for _ in range(n_gen):
        fit = fitness(pop)
        order = np.argsort(fit)
        pop = pop[order]
        elites = pop[: max(2, pop_size // 5)]
        children = []
        while len(children) + elites.shape[0] < pop_size:
            p1, p2 = rng.choice(elites, size=2, replace=True)
            cross = rng.uniform(0.2, 0.8)
            child = cross * p1 + (1 - cross) * p2
            mutation = rng.normal(0, 0.05, size=T)
            child = np.clip(child + mutation, 0.0, 1.0)
            children.append(child)
        pop = np.vstack([elites, np.asarray(children)])

    best_idx = int(np.argmin(fitness(pop)))
    return haz_row * pop[best_idx]


def generate_cf_from_arrays(
    hazards: Iterable[Iterable[float]] | torch.Tensor,
    risk_scores: Optional[Iterable[float]] = None,
    interval: Optional[int] = None,
    feature_names: Optional[Sequence[str]] = None,
    save_path: Optional[str] = None,
    patient_indices: Optional[Sequence[int]] = None,
    desired_extension: float = 1.0,
    use_genetic_backup: bool = True,
) -> CFResult:
    """Create post-hoc counterfactual suggestions from hazards.

    A deterministic scaling-based search shrinks hazards to achieve a user-defined
    survival extension. When the constraint cannot be satisfied, a lightweight
    genetic search explores interval-wise scaling factors as a backup.
    """

    haz = torch.as_tensor(hazards, dtype=torch.float32).cpu().numpy()
    if haz.ndim == 1:
        haz = haz[:, None]

    all_indices = list(range(haz.shape[0]))
    idx_subset = list(patient_indices) if patient_indices is not None else all_indices
    idx_subset = [i for i in idx_subset if 0 <= i < haz.shape[0]]
    if not idx_subset:
        idx_subset = [0] if haz.shape[0] else []

    risks = risk_scores
    if risks is None:
        risks = haz.sum(axis=1)
    risks = np.asarray(risks, dtype=float).flatten()

    if risks.shape[0] < haz.shape[0]:
        # pad risks to align with hazards if a smaller vector was passed
        pad = np.repeat(risks[-1] if risks.size else 0.0, haz.shape[0] - risks.shape[0])
        risks = np.concatenate([risks, pad])

    target_vals, interval_idx = _select_interval(haz, interval)
    horizon = haz.shape[1]

    suggestions = []
    for i in idx_subset:
        base_row = haz[i].copy()
        base_risk = float(risks[i]) if i < risks.size else float(base_row.sum())
        base_ch = float(base_row.sum())

        target_ch = _target_cumhaz(base_ch, horizon=horizon, desired_extension=desired_extension)
        scaled_row = _binary_scale_search(base_row, target_ch, interval=interval_idx[i])

        if use_genetic_backup and float(scaled_row.sum()) > target_ch * 1.01:
            scaled_row = _genetic_search(base_row, target_ch)

        achieved_ch = float(scaled_row.sum())
        achieved_extension = horizon * max(base_ch / max(achieved_ch, 1e-6) - 1.0, 0.0)
        hazard_delta = float(base_row.sum() - scaled_row.sum())

        top_features = ""
        if feature_names:
            feats = ", ".join(feature_names[:3]) if len(feature_names) >= 3 else ", ".join(feature_names)
            top_features = f" 优先关注: {feats}."

        t_int = int(interval_idx[i] + 1)
        suggestions.append(
            {
                "sample_id": i,
                "target_interval": t_int,
                "current_cumhaz": base_ch,
                "target_cumhaz": target_ch,
                "achieved_cumhaz": achieved_ch,
                "hazard_reduction": hazard_delta,
                "estimated_extension": achieved_extension,
                "desired_extension": desired_extension,
                "risk_score": base_risk,
                "action": (
                    f"期望延长生存约 {desired_extension:.1f} 个时间单位；"
                    f"在区间 {t_int} 处收紧危险度，可将累计风险由 {base_ch:.3f} 下调到 {achieved_ch:.3f}。"
                    "可尝试加强随访频率、监测症状和依从性管理，若仍不足则考虑多因素联合干预。"
                    + top_features
                ),
            }
        )

    cf_table = pd.DataFrame(suggestions)
    summary = {
        "interval": interval or "peak",
        "mean_risk": float(np.mean([s.get("risk_score", 0.0) for s in suggestions])) if suggestions else 0.0,
        "mean_target_hazard": float(np.mean(target_vals)) if len(target_vals) else 0.0,
        "mean_achieved_extension": float(np.mean([s.get("estimated_extension", 0.0) for s in suggestions])) if suggestions else 0.0,
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
    patient_indices: Optional[Sequence[int]] = None,
    desired_extension: float = 1.0,
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
    return generate_cf_from_arrays(
        hazards,
        interval=interval,
        feature_names=feature_names,
        save_path=save_path,
        patient_indices=patient_indices,
        desired_extension=desired_extension,
    )


# models/mysa.py
# =============================================================================
# MySA (TEXGI + Expert Priors) — Full, non-simplified integration
# -----------------------------------------------------------------------------
# - Uses your MultiTaskModel, discretization & loaders from data.py
# - Adversarial extreme baseline via a learned generator with real input
# - Generalized Pareto distributed "extreme code" to steer generator
# - Time-dependent Integrated Gradients per time-bin (TEXGI)
# - Expert penalty supports:
#     * directional constraints (sign in {-1,0,+1})
#     * minimum magnitude constraints per feature
#     * relational constraints w.r.t. global mean importance (>=mean or <=mean)
# - Two tunable loss weights exposed to UI:
#     * lambda_smooth: temporal smoothness on hazards
#     * lambda_expert: expert prior penalty on attributions
#
# Public API:
#   run_mysa(data: pd.DataFrame, config: dict) -> dict
#   run_texgisa(...)  # alias for compatibility with existing UI
# =============================================================================

from typing import Dict, List, Optional, Tuple, Any, Sequence
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---- robust import guard (optional hardening) ----
import os, sys, importlib.util, pathlib
_CUR = pathlib.Path(__file__).resolve()
_ROOT = _CUR.parent.parent  # /.../DHAIWEB_SA
for p in (str(_ROOT), str(pathlib.Path.cwd())):
    if p not in sys.path:
        sys.path.insert(0, p)

# Try normal imports first
try:
    from data import make_intervals, build_supervision, infer_feature_cols, create_dataloaders
    from model import MultiTaskModel
    from metrics import cindex_fast
except ModuleNotFoundError:
    # Fallback: import by absolute file path
    def _load_module(name, path):
        spec = importlib.util.spec_from_file_location(name, str(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    data_mod    = _load_module("data",    _ROOT / "data.py")
    model_mod   = _load_module("model",   _ROOT / "model.py")
    metrics_mod = _load_module("metrics", _ROOT / "metrics.py")
    make_intervals    = data_mod.make_intervals
    build_supervision = data_mod.build_supervision
    infer_feature_cols= data_mod.infer_feature_cols
    create_dataloaders= data_mod.create_dataloaders
    MultiTaskModel    = model_mod.MultiTaskModel
    cindex_fast       = metrics_mod.cindex_fast
# --------------------------------------------------


from data import make_intervals, build_supervision, infer_feature_cols, create_dataloaders
from model import MultiTaskModel
from metrics import cindex_fast
from multimodal_assets import (
    AssetBackedMultiModalDataset,
    build_image_paths_for_ids,
    build_sensor_paths_for_ids,
    SensorSpec,
)
from utils.identifiers import canonicalize_series

__all__ = ["run_mysa", "run_texgisa"]

# -------------------------- Core loss components -------------------------------

def masked_bce_nll(hazards: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """
    hazards: [B, T] in (0,1)
    labels : [B, T] in {0,1}
    masks  : [B, T] in {0,1} -> 1 means valid timestep for the sample
    """
    eps = 1e-6
    hazards = hazards.clamp(eps, 1 - eps)
    bce = -(labels * torch.log(hazards) + (1.0 - labels) * torch.log(1.0 - hazards))
    masked = bce * masks
    denom = masks.sum().clamp_min(1.0)
    return masked.sum() / denom


def attribution_temporal_l1(phi_tbd: torch.Tensor) -> torch.Tensor:
    """L1 temporal smoothness penalty on TEXGI attributions.

    Implements Eq.(smooth) from the paper: sum_{i,b,d} |phi_{t+1,b,d} - phi_{t,b,d}|.
    Returns the raw (unweighted) penalty value.
    """
    if phi_tbd.dim() != 3:
        raise ValueError("phi tensor must have shape [T, B, D] for temporal smoothing")
    if phi_tbd.size(0) <= 1:
        return phi_tbd.new_zeros(())
    diff = phi_tbd[1:] - phi_tbd[:-1]
    return diff.abs().sum()


# -------------------- Adversarial Extreme Baseline Generator ------------------

class TabularGeneratorWithRealInput(nn.Module):
    """
    G(x, z, e) -> x_adv
    - x: real standardized features [B, D]
    - z: latent noise [B, Z]
    - e: extreme code sampled from a Generalized Pareto dist [B, E]
    The generator outputs a baseline lying near the data manifold but
    shifted toward an extreme direction encoded by e.
    """
    def __init__(self, input_dim: int, latent_dim: int = 16, extreme_dim: int = 1, hidden: int = 256, depth: int = 3):
        super().__init__()
        D = input_dim + latent_dim + extreme_dim
        layers = []
        h = hidden
        layers.append(nn.Linear(D, h))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 1):
            layers.append(nn.Linear(h, h))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(h, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, z: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, z, e], dim=1))


class TensorDatasetWithMask(Dataset):
    """Tensor dataset that optionally yields modality availability masks."""

    def __init__(self, X: np.ndarray, y: np.ndarray, m: np.ndarray, mask: Optional[np.ndarray] = None):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.m = torch.from_numpy(m.astype(np.float32))
        self.mask = None if mask is None else torch.from_numpy(mask.astype(np.float32))

    def __len__(self) -> int:  # type: ignore[override]
        return self.X.shape[0]

    def __getitem__(self, idx: int):  # type: ignore[override]
        if self.mask is None:
            return self.X[idx], self.y[idx], self.m[idx]
        return self.X[idx], self.y[idx], self.m[idx], self.mask[idx]


def create_tensor_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    m_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    m_val: np.ndarray,
    *,
    batch_size: int = 128,
    num_workers: int = 0,
    mask_train: Optional[np.ndarray] = None,
    mask_val: Optional[np.ndarray] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create dataloaders for either single- or multi-modal tensors."""

    train_ds = TensorDatasetWithMask(X_train, y_train, m_train, mask_train)
    val_ds = TensorDatasetWithMask(X_val, y_val, m_val, mask_val)

    pin_memory = torch.cuda.is_available()
    use_persistent = bool(num_workers and num_workers > 0)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
    )
    return train_loader, val_loader


def _build_encoder(input_dim: int, hidden: int, depth: int, dropout: float) -> nn.Sequential:
    layers: List[nn.Module] = []
    d = input_dim
    for _ in range(max(1, depth)):
        layers.append(nn.Linear(d, hidden))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = hidden
    return nn.Sequential(*layers)


class TabularAssetEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, depth: int, hidden: int, dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        d = input_dim
        for _ in range(max(1, depth - 1)):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden
        layers.append(nn.Linear(d, embed_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiModalMySAModel(nn.Module):
    """Gating-based fusion model supporting both flat and raw multimodal inputs."""

    def __init__(
        self,
        modality_slices: Optional[List[Tuple[str, slice]]],
        num_bins: int,
        hidden: int = 256,
        depth: int = 2,
        dropout: float = 0.2,
        raw_modalities: Optional[List[Tuple[str, Dict[str, Any]]]] = None,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.mode = "flat"
        self.hazard_layer = nn.Linear(hidden, num_bins)

        if raw_modalities:
            self.mode = "raw"
            self.raw_order: List[str] = []
            self.raw_slices: Dict[str, slice] = {}
            self.raw_encoders = nn.ModuleDict()
            self.projectors = nn.ModuleDict()
            self.gates = nn.ModuleDict()

            start = 0
            for name, info in raw_modalities:
                raw_dim = int(info.get("raw_dim", 0))
                if raw_dim <= 0:
                    raise ValueError(f"Modality '{name}' has invalid raw_dim {raw_dim}")
                sl = slice(start, start + raw_dim)
                self.raw_order.append(name)
                self.raw_slices[name] = sl
                start += raw_dim

                if name == "tabular":
                    input_dim = int(info.get("input_dim", raw_dim))
                    encoder = TabularAssetEncoder(input_dim, raw_dim, depth, hidden, dropout)
                elif name == "image":
                    from .multimodal import ImageEncoder as _ImageEncoder

                    backbone = info.get("backbone", "resnet18")
                    pretrained = bool(info.get("pretrained", True))
                    img_backbone = _ImageEncoder(backbone=backbone, pretrained=pretrained)
                    encoder = nn.Sequential(
                        img_backbone,
                        nn.Linear(img_backbone.out_dim, raw_dim),
                        nn.ReLU(),
                    )
                elif name == "sensor":
                    from .multimodal import SensorEncoder as _SensorEncoder

                    in_channels = int(info.get("in_channels", 1))
                    backbone = info.get("backbone", "cnn")
                    sensor_backbone = _SensorEncoder(input_channels=in_channels, backbone=backbone)
                    encoder = nn.Sequential(
                        sensor_backbone,
                        nn.Linear(sensor_backbone.out_dim, raw_dim),
                        nn.ReLU(),
                    )
                else:
                    raise ValueError(f"Unsupported modality '{name}' in raw configuration")

                self.raw_encoders[name] = encoder
                proj_layers: List[nn.Module] = [nn.Linear(raw_dim, hidden), nn.ReLU()]
                if dropout > 0:
                    proj_layers.append(nn.Dropout(dropout))
                self.projectors[name] = nn.Sequential(*proj_layers)
                self.gates[name] = nn.Linear(hidden, 1)

        elif modality_slices:
            self.modality_slices = modality_slices
            encoders = {}
            gates = {}
            for name, sl in modality_slices:
                in_dim = sl.stop - sl.start
                if in_dim <= 0:
                    raise ValueError(f"Modality '{name}' has non-positive feature dimension: {in_dim}")
                encoders[name] = _build_encoder(in_dim, hidden, depth, dropout)
                gates[name] = nn.Linear(hidden, 1)
            self.encoders = nn.ModuleDict(encoders)
            self.gates = nn.ModuleDict(gates)
        else:
            raise ValueError("Either modality_slices or raw_modalities must be provided.")

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def encode_modalities(
        self,
        x: Dict[str, torch.Tensor],
        modality_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.mode != "raw":
            raise RuntimeError("encode_modalities is only available in raw mode.")
        feats: List[torch.Tensor] = []
        for idx, name in enumerate(self.raw_order):
            feat = self.raw_encoders[name](x[name])
            sl = self.raw_slices[name]
            if modality_mask is not None:
                mask = modality_mask[:, idx].unsqueeze(1)
                feat = feat * mask
            feats.append(feat)
        return torch.cat(feats, dim=1)

    # ------------------------------------------------------------------
    def forward_from_embeddings(
        self,
        embeddings: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.mode == "flat":
            return self._forward_flat(embeddings, modality_mask)

        feats: List[torch.Tensor] = []
        gate_logits: List[torch.Tensor] = []
        for idx, name in enumerate(self.raw_order):
            sl = self.raw_slices[name]
            seg = embeddings[:, sl]
            if modality_mask is not None:
                seg = seg * modality_mask[:, idx].unsqueeze(1)
            h = self.projectors[name](seg)
            feats.append(h)
            gate_logits.append(self.gates[name](h))

        logits = torch.cat(gate_logits, dim=1)
        if modality_mask is not None:
            logits = logits.masked_fill(modality_mask == 0, -1e4)

        attn = torch.softmax(logits, dim=1)
        fused = torch.zeros(embeddings.size(0), self.hidden, device=embeddings.device)
        for idx, _ in enumerate(self.raw_order):
            fused = fused + feats[idx] * attn[:, idx].unsqueeze(1)

        return torch.sigmoid(self.hazard_layer(fused))

    # ------------------------------------------------------------------
    def _forward_flat(
        self,
        x: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = x.size(0)
        feats: List[torch.Tensor] = []
        gate_logits: List[torch.Tensor] = []
        for idx, (name, sl) in enumerate(self.modality_slices):
            h = self.encoders[name](x[:, sl])
            if modality_mask is not None:
                mask = modality_mask[:, idx].unsqueeze(1)
                h = h * mask
            feats.append(h)
            gate_logits.append(self.gates[name](h))
        logits = torch.cat(gate_logits, dim=1)
        if modality_mask is not None:
            logits = logits.masked_fill(modality_mask == 0, -1e4)
        attn = torch.softmax(logits, dim=1)
        fused = torch.zeros(B, self.hidden, device=x.device)
        for idx, _ in enumerate(self.modality_slices):
            fused = fused + feats[idx] * attn[:, idx].unsqueeze(1)
        return torch.sigmoid(self.hazard_layer(fused))

    # ------------------------------------------------------------------
    def forward(
        self,
        x: Any,
        modality_mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Any:
        if self.mode == "flat":
            if not torch.is_tensor(x):
                raise TypeError("Flat mode expects a tensor input")
            hazards = self._forward_flat(x, modality_mask)
            if return_embeddings:
                return hazards, x
            return hazards

        # raw mode
        if isinstance(x, dict):
            embeddings = self.encode_modalities(x, modality_mask)
            hazards = self.forward_from_embeddings(embeddings, modality_mask)
            if return_embeddings:
                return hazards, embeddings
            return hazards

        if torch.is_tensor(x):
            hazards = self.forward_from_embeddings(x, modality_mask)
            if return_embeddings:
                return hazards, x
            return hazards

        raise TypeError("Unsupported input type for MultiModalMySAModel")


@torch.no_grad()
def _standardize_fit(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mu = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True).clamp_min(1e-6)
    return mu, std


def _standardize_apply(X: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (X - mu) / std


def _destandardize(Xs: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return Xs * std + mu


def sample_extreme_code(batch_size: int, extreme_dim: int = 1, device: str = "cpu",
                        xi: float = 0.3, beta: float = 1.0) -> torch.Tensor:
    """
    Sample from Generalized Pareto Distribution (GPD) using inverse CDF for u~Uniform(0,1):
      GPD(u) = beta/xi * ( (1 - u)^(-xi) - 1 )
    We map it monotonically to (0, +∞). For stability, cap u away from 1.
    """
    u = torch.rand(batch_size, extreme_dim, device=device).clamp(1e-6, 1 - 1e-6)
    e = beta / xi * ((1 - u) ** (-xi) - 1.0)
    return e


def train_adv_generator(
    model: nn.Module,
    X_tr: torch.Tensor,
    mu: torch.Tensor,
    std: torch.Tensor,
    epochs: int = 200,
    batch_size: int = 256,
    latent_dim: int = 16,
    extreme_dim: int = 1,
    lr: float = 1e-3,
    alpha_dist: float = 1.0,
    device: Optional[str] = None,
    modality_mask: Optional[torch.Tensor] = None,
) -> Tuple[TabularGeneratorWithRealInput, Dict[str, torch.Tensor]]:
    """
    Train a generator G to produce "adversarial extreme baselines".
    Objective (max-risk with proximity):
       maximize  Risk( model( x_adv ) )  - alpha_dist * ||x_adv - x||_2^2
    where Risk = sum_t hazard_t (earlier event -> larger risk).
    Notes:
      - We optimize the negative of above because we use Adam on G to MINIMIZE.
      - x, x_adv are standardized here; caller will de-standardize when needed.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()  # freeze model during G training

    G = TabularGeneratorWithRealInput(X_tr.shape[1], latent_dim, extreme_dim).to(device)
    opt = torch.optim.Adam(G.parameters(), lr=lr)

    N = X_tr.shape[0]
    idx = torch.randperm(N)
    X_std = _standardize_apply(X_tr.to(device), mu.to(device), std.to(device))
    if modality_mask is not None:
        modality_mask = modality_mask.to(device)

    steps_per_epoch = max(1, N // batch_size)
    for ep in range(1, epochs + 1):
        ep_loss = 0.0
        for i in range(steps_per_epoch):
            sl = i * batch_size
            sr = min(N, sl + batch_size)
            xb = X_std[idx[sl:sr]]  # standardized input

            z = torch.randn(xb.size(0), latent_dim, device=device)
            e = sample_extreme_code(xb.size(0), extreme_dim=extreme_dim, device=device)

            x_adv = G(xb, z, e)                          # standardized baseline
            x_adv_real = _destandardize(x_adv, mu.to(device), std.to(device))  # back to real space

            # Risk proxy: sum of hazards
            mask_batch = None
            if modality_mask is not None:
                mask_batch = modality_mask[idx[sl:sr]]

            with torch.no_grad():
                if mask_batch is not None:
                    hazards = model(x_adv_real, modality_mask=mask_batch)  # [B,T]
                else:
                    hazards = model(x_adv_real)               # [B,T]
                risk = hazards.sum(dim=1)                 # larger -> earlier event

            # Proximity in standardized space
            dist = (x_adv - xb).pow(2).sum(dim=1)

            # We minimize: -risk + alpha * dist
            loss = (-risk + alpha_dist * dist).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
            opt.step()

            ep_loss += loss.item()

        if ep % 50 == 0 or ep == 1:
            print(f"[Gen] Ep{ep:03d}  loss={ep_loss / steps_per_epoch:.4f}")

    ref_stats = {"mu": mu.to(device), "std": std.to(device)}
    try:
        torch.save(G.state_dict(), "mysa_G.pt")
    except Exception:
        pass

    return G, ref_stats


# ---------------------- Time-dependent IG (TEXGI per t) -----------------------

def integrated_gradients_time(
    f: nn.Module,
    X: torch.Tensor,           # [B, D] (real space, requires_grad can be True)
    X_baseline: torch.Tensor,  # [B, D] (real space)
    hazard_index: int,         # target time-bin
    M: int = 20,
    forward_kwargs: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Compute IG for one time index t along the straight path from baseline to input.
    Returns: IG attributions [B, D] for this t.
    """
    assert X.shape == X_baseline.shape
    device = X.device
    alphas = torch.linspace(0.0, 1.0, steps=M + 1, device=device)[1:]  # exclude 0
    Xdiff = (X - X_baseline)

    atts_terms: List[torch.Tensor] = []
    kwargs = forward_kwargs or {}

    for a in alphas:
        Xpath = X_baseline + a * Xdiff
        Xpath.requires_grad_(True)
        hazards = f(Xpath, **kwargs) if kwargs else f(Xpath)  # [B, T]
        out = hazards[:, hazard_index]          # focus on bin t
        grads = torch.autograd.grad(
            out.sum(),
            Xpath,
            retain_graph=False,
            create_graph=False,
        )[0]
        atts_terms.append(grads)

    atts = torch.stack(atts_terms, dim=0).mean(dim=0) * Xdiff
    return atts


def texgi_time_series(
    f: nn.Module,
    X: torch.Tensor,                 # [B, D] real space
    G: Optional[TabularGeneratorWithRealInput],
    ref_stats: Dict[str, torch.Tensor],
    M: int = 20,
    latent_dim: int = 16,
    extreme_dim: int = 1,
    t_sample: Optional[int] = None,
    forward_kwargs: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Compute TEXGI per time-bin, returning phi with shape [T, B, D].
    Baseline is produced by adversarial generator conditioned on (x, z, e).
    """
    if not torch.is_tensor(X):
        raise TypeError(
            "texgi_time_series expects a torch.Tensor as input features; "
            f"received {type(X).__name__}."
        )
    device = X.device
    # prepare baseline
    if G is not None:
        with torch.no_grad():
            mu, std = ref_stats["mu"], ref_stats["std"]
            Xstd = _standardize_apply(X, mu, std)
            z = torch.randn(X.size(0), latent_dim, device=device)
            e = sample_extreme_code(X.size(0), extreme_dim=extreme_dim, device=device)
            Xb_std = G(Xstd, z, e)
            X_baseline = _destandardize(Xb_std, mu, std)
    else:
        # fallback: use high-quantile statistic as last resort (should rarely happen)
        q_hi = torch.quantile(X, 0.98, dim=0, keepdim=True)
        X_baseline = q_hi.repeat(X.size(0), 1)

    with torch.no_grad():
        if forward_kwargs:
            T = f(X[:1], **forward_kwargs).shape[1]
        else:
            T = f(X[:1]).shape[1]

    # Choose which time bins participate in this pass (random or evenly spaced)
    if t_sample is not None and t_sample > 0 and t_sample < T:
        # Evenly spaced sampling tends to be more stable
        import math
        step = max(1, math.floor(T / t_sample))
        t_indices = list(range(0, T, step))[:t_sample]
    else:
        t_indices = list(range(T))

    phi_list = []
    for t in t_indices:
        ig_t = integrated_gradients_time(
            f,
            X,
            X_baseline,
            hazard_index=t,
            M=M,
            forward_kwargs=forward_kwargs,
        )
        phi_list.append(ig_t)

    # Resulting shape [T', B, D] (if subsampled, T' < T)
    phi = torch.stack(phi_list, dim=0)
    return phi



# -------------------- Expert prior penalty (rich constraints) -----------------

def _aggregate_importance(phi_tbd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    phi_tbd: [T, B, D]
    Returns:
      imp_abs: per-feature global importance = mean_{t,b} |phi| -> [D]
      imp_dir: per-feature directional mean = mean_{t,b} phi     -> [D]
    """
    imp_abs = phi_tbd.abs().mean(dim=(0, 1))  # [D]
    imp_dir = phi_tbd.mean(dim=(0, 1))        # [D]
    return imp_abs, imp_dir


def _resolve_important_feature_indices(
    expert_config: Optional[Dict[str, Any]], feat2idx: Dict[str, int]
) -> List[int]:
    """Extract the set I (important features) from the expert configuration."""
    if not expert_config:
        return []

    names: List[str] = []
    for key in ("important_features", "important", "I"):
        val = expert_config.get(key)
        if isinstance(val, (list, tuple, set)):
            names.extend(str(v) for v in val)

    # Backward compatibility: treat old rule entries with relation >=mean as important.
    for rule in expert_config.get("rules", []) if isinstance(expert_config.get("rules"), list) else []:
        if not isinstance(rule, dict):
            continue
        fname = rule.get("feature")
        if not fname:
            continue
        relation = str(rule.get("relation", "")).lower()
        if relation in {">=mean", ">=avg", "important"} or bool(rule.get("important", False)):
            names.append(str(fname))

    idx = {feat2idx[name] for name in names if name in feat2idx}
    return sorted(idx)


def expert_penalty(phi_tbd: torch.Tensor, important_idx: Sequence[int]) -> torch.Tensor:
    """Expert prior penalty Ω_expert(Φ) following Eq.(18) of the paper."""
    if phi_tbd.dim() != 3:
        raise ValueError("phi tensor must have shape [T, B, D] for expert penalty")

    device = phi_tbd.device
    # ||Φ_l||_2 across all time bins and batch elements
    norms = torch.sqrt(phi_tbd.pow(2).sum(dim=(0, 1)) + 1e-12)  # [D]
    if norms.numel() == 0:
        return torch.zeros((), device=device)

    bar_s = norms.mean()
    total = torch.zeros((), device=device)

    important_mask = torch.zeros_like(norms, dtype=torch.bool)
    if important_idx:
        important_tensor = torch.tensor(important_idx, device=device, dtype=torch.long)
        important_mask[important_tensor] = True
        total = total + F.relu(bar_s - norms[important_tensor]).sum()

    non_important = ~important_mask
    if non_important.any():
        total = total + norms[non_important].sum()

    return total


# ------------------------------- Trainer --------------------------------------

class MySATrainer:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        device: Optional[str] = None,
        lambda_smooth: float = 0.0,
        lambda_expert: float = 0.0,
        expert_rules: Optional[Dict[str, Any]] = None,
        feat2idx: Optional[Dict[str, int]] = None,
        # TEXGI settings:
        ig_steps: int = 20,
        latent_dim: int = 16,
        extreme_dim: int = 1,
        ig_batch_samples: int = 64,                 # Max samples per batch participating in TEXGI
        ig_time_subsample: Optional[int] = None,    # Max time bins per batch for TEXGI (None = all bins)
        # Generator settings:
        gen_epochs: int = 200,
        gen_batch: int = 256,
        gen_lr: float = 1e-3,
        gen_alpha_dist: float = 1.0,
        ref_stats: Optional[Dict[str, torch.Tensor]] = None,
        X_train_ref: Optional[torch.Tensor] = None,  # real-space training features for generator fit
        modality_mask_ref: Optional[torch.Tensor] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode="min", factor=0.5, patience=5, verbose=False)

        self.lambda_smooth = float(lambda_smooth)
        self.lambda_expert = float(lambda_expert)
        self.expert_rules = expert_rules or {}
        self.feat2idx = feat2idx or {}
        self.important_idx = _resolve_important_feature_indices(self.expert_rules, self.feat2idx)

        self.ig_steps = int(ig_steps)
        self.latent_dim = int(latent_dim)
        self.extreme_dim = int(extreme_dim)

        self.G: Optional[TabularGeneratorWithRealInput] = None
        self.ref_stats = ref_stats
        self.gen_epochs = int(gen_epochs)
        self.gen_batch = int(gen_batch)
        self.gen_lr = float(gen_lr)
        self.gen_alpha_dist = float(gen_alpha_dist)
        self.X_train_ref = X_train_ref  # [N,D] float32 tensor
        self.modality_mask_ref = modality_mask_ref
        self.ig_batch_samples = int(ig_batch_samples)
        self.ig_time_subsample = (int(ig_time_subsample) if ig_time_subsample
                                  else None)

    def _move_to_device(self, obj: Any):
        """Recursively move tensors/nested containers onto the configured device."""
        if torch.is_tensor(obj):
            return obj.to(self.device, non_blocking=True)
        if isinstance(obj, dict):
            return {k: self._move_to_device(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(*(self._move_to_device(v) for v in obj))
        return obj

    def _model_forward(
        self,
        X: Any,
        modality_mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> Any:
        kwargs: Dict[str, Any] = {}
        if modality_mask is not None:
            kwargs["modality_mask"] = modality_mask
        if return_embeddings:
            kwargs["return_embeddings"] = True
        try:
            return self.model(X, **kwargs)
        except TypeError:
            kwargs.pop("modality_mask", None)
            return self.model(X, **kwargs)

    @staticmethod
    def _unpack_batch(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(batch, dict):
            X = batch.get("modalities") or batch.get("x")
            y = batch.get("y")
            m = batch.get("m")
            mask = batch.get("modality_mask")
            if X is None or y is None or m is None:
                raise ValueError("Batch dict must contain modalities/y/m")
            return X, y, m, mask
        if isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                X, y, m = batch
                return X, y, m, None
            if len(batch) == 4:
                X, y, m, mask = batch
                return X, y, m, mask
        raise ValueError("Unexpected batch structure for MySA trainer.")

    def fit_generator_if_needed(self):
        if self.lambda_expert <= 0:
            return  # no expert penalty, generator not required
        if self.G is not None:
            return
        # If a trained generator already exists on disk, load it to accelerate repeat runs
        import os, torch
        ckpt_path = "mysa_G.pt"
        if os.path.exists(ckpt_path):
            try:
                self.G = TabularGeneratorWithRealInput(self.X_train_ref.shape[1],
                                                       self.latent_dim, self.extreme_dim).to(self.device)
                self.G.load_state_dict(torch.load(ckpt_path, map_location=self.device))
                return
            except Exception:
                self.G = None  # Fall back to retraining if loading fails

        assert self.X_train_ref is not None, "X_train_ref must be provided to fit generator."
        Xtr = self.X_train_ref.to(self.device)
        mu, std = _standardize_fit(Xtr)
        self.G, self.ref_stats = train_adv_generator(
            self.model, Xtr, mu, std,
            epochs=self.gen_epochs, batch_size=self.gen_batch,
            latent_dim=self.latent_dim, extreme_dim=self.extreme_dim,
            lr=self.gen_lr, alpha_dist=self.gen_alpha_dist, device=self.device,
            modality_mask=self.modality_mask_ref
        )

    def step(self, batch) -> Tuple[float, float, float]:
        X, y, m, modality_mask = self._unpack_batch(batch)
        X = self._move_to_device(X)
        y = y.to(self.device, non_blocking=True)
        m = m.to(self.device, non_blocking=True)
        mask_t = modality_mask.to(self.device, non_blocking=True) if modality_mask is not None else None

        self.model.train()
        out = self._model_forward(X, mask_t, return_embeddings=True)
        if isinstance(out, tuple):
            hazards, embeddings = out
        else:
            hazards, embeddings = out, X if torch.is_tensor(X) else None
        hazards = hazards  # [B, T]
        loss_main = masked_bce_nll(hazards, y, m)
        loss_smooth = hazards.new_tensor(0.0)
        loss_expert = hazards.new_tensor(0.0)
        need_phi = (self.lambda_smooth > 0) or (self.lambda_expert > 0)
        if need_phi:
            # Generator only required when expert penalty is active
            if self.lambda_expert > 0:
                self.fit_generator_if_needed()
            with torch.enable_grad():
                if torch.is_tensor(X):
                    B = X.shape[0]
                elif embeddings is not None and torch.is_tensor(embeddings):
                    B = embeddings.shape[0]
                else:
                    raise ValueError(
                        "TEXGI requires tensor inputs or embeddings; received type "
                        f"{type(X).__name__}."
                    )

                sub = min(B, self.ig_batch_samples)
                idx = torch.randperm(B, device=hazards.device)[:sub]

                use_embeddings = (
                    embeddings is not None
                    and torch.is_tensor(embeddings)
                    and hasattr(self.model, "forward_from_embeddings")
                )

                if use_embeddings:
                    X_source = embeddings
                else:
                    if not torch.is_tensor(X):
                        raise ValueError(
                            "Model does not expose forward_from_embeddings; TEXGI "
                            "must operate on tensor inputs."
                        )
                    X_source = X

                Xsub = X_source[idx].detach().clone().requires_grad_(True)

                mask_sub = None
                if mask_t is not None:
                    mask_sub = mask_t[idx].detach()

                if use_embeddings:
                    forward_fn = lambda z, modality_mask=None: self.model.forward_from_embeddings(  # noqa: E731
                        z,
                        modality_mask=modality_mask,
                    )
                    forward_kwargs = {"modality_mask": mask_sub} if mask_sub is not None else None
                else:
                    forward_fn = self.model
                    forward_kwargs = None

                phi = texgi_time_series(
                    forward_fn,
                    Xsub,
                    self.G,
                    self.ref_stats if (self.G is not None and self.ref_stats is not None) else {},
                    M=self.ig_steps,
                    latent_dim=self.latent_dim,
                    extreme_dim=self.extreme_dim,
                    t_sample=self.ig_time_subsample,
                    forward_kwargs=forward_kwargs,
                )

                if self.lambda_smooth > 0:
                    omega_smooth = attribution_temporal_l1(phi)
                    loss_smooth = self.lambda_smooth * omega_smooth

                if self.lambda_expert > 0:
                    omega_expert = expert_penalty(phi, self.important_idx)
                    loss_expert = self.lambda_expert * omega_expert

        loss = loss_main + loss_smooth + loss_expert

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.opt.step()
        return (
            loss_main.item(),
            loss_smooth.item(),
            loss_expert.item(),
        )

    @torch.no_grad()
    def evaluate_cindex(self, loader, durations, events) -> float:
        self.model.eval()
        all_risk = []
        for batch in loader:
            X, _, _, modality_mask = self._unpack_batch(batch)
            X = self._move_to_device(X)
            mask_t = modality_mask.to(self.device, non_blocking=True) if modality_mask is not None else None
            hazards = self._model_forward(X, mask_t)
            risk = torch.sum(hazards, dim=1)  # simple risk proxy; larger -> earlier event
            all_risk.append(risk.cpu())
        risks = torch.cat(all_risk, dim=0)
        return cindex_fast(durations, events, risks).item()


# ------------------------------ Utilities -------------------------------------


def _prepare_tabular_inputs(data: pd.DataFrame, config: Dict) -> Dict[str, Any]:
    df = data.copy().reset_index(drop=True)
    required_cols = {"duration", "event"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing columns for MySA tabular run: {missing}")

    n_bins = int(config.get("n_bins", 30))
    df = make_intervals(df, duration_col="duration", event_col="event", n_bins=n_bins, method="quantile")
    num_bins = int(df["interval_number"].max())

    feat_cols = config.get("feature_cols")
    if not feat_cols:
        feat_cols = infer_feature_cols(df, exclude=["duration", "event"])

    cleaned_cols: List[str] = []
    dropped_cols: List[str] = []
    for col in list(feat_cols):
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() == 0:
            dropped_cols.append(col)
            df.drop(columns=[col], inplace=True)
            continue
        fill_value = float(series.median()) if series.notna().any() else 0.0
        df[col] = series.fillna(fill_value).astype(np.float32)
        cleaned_cols.append(col)

    if not cleaned_cols:
        raise ValueError("No numeric feature columns available after preprocessing.")

    feat_cols = cleaned_cols
    feat2idx = {n: i for i, n in enumerate(feat_cols)}

    val_ratio = float(config.get("val_ratio", 0.2))
    N = len(df)
    idx = np.arange(N)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    split = int(N * (1.0 - val_ratio))
    tr, va = idx[:split], idx[split:]

    durations = df["duration"].to_numpy(np.float32)
    events = df["event"].to_numpy(np.int32)
    intervals = df["interval_number"].to_numpy(np.int32)
    X_all = df[feat_cols].to_numpy(np.float32)

    X_tr = X_all[tr]
    X_va = X_all[va]
    y_tr, m_tr = build_supervision(intervals[tr], events[tr], num_bins)
    y_va, m_va = build_supervision(intervals[va], events[va], num_bins)

    train_loader, val_loader = create_tensor_dataloaders(
        X_tr,
        y_tr,
        m_tr,
        X_va,
        y_va,
        m_va,
        batch_size=int(config.get("batch_size", 128)),
        num_workers=int(config.get("num_workers", 0)),
    )

    model = MultiTaskModel(
        input_dim=X_tr.shape[1],
        num_bins=num_bins,
        hidden=int(config.get("hidden", 256)),
        depth=int(config.get("depth", 2)),
        dropout=float(config.get("dropout", 0.2)),
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "model": model,
        "feature_names": feat_cols,
        "feat2idx": feat2idx,
        "num_bins": num_bins,
        "dur_va": torch.tensor(durations[va]),
        "evt_va": torch.tensor(events[va]),
        "X_train": X_tr,
        "X_val": X_va,
        "mask_train": None,
        "mask_val": None,
        "dropped_feature_columns": dropped_cols,
    }


def _align_modality_features(
    base_ids: pd.Index,
    df: pd.DataFrame,
    id_col: str,
    feature_cols: List[str],
    prefix: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if not feature_cols:
        return np.zeros((len(base_ids), 0), dtype=np.float32), np.zeros((len(base_ids),), dtype=np.float32), []

    sub = df[[id_col] + feature_cols].copy()
    sub[id_col] = canonicalize_series(sub[id_col]).fillna("")
    for col in feature_cols:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.set_index(id_col)
    aligned = sub.reindex(base_ids)
    mask = (~aligned.isna()).any(axis=1).astype(np.float32).to_numpy()
    aligned = aligned.fillna(0.0)
    arr = aligned.to_numpy(np.float32)

    if prefix:
        names = [f"{prefix}{c}" for c in feature_cols]
    else:
        names = list(feature_cols)
    return arr, mask, names


def _collect_embeddings(
    model: "MultiModalMySAModel",
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect encoder embeddings and modality masks from a dataloader."""

    feats: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            X, _, _, mod_mask = batch
            if isinstance(X, dict):
                X_dev = {k: v.to(device, non_blocking=True) for k, v in X.items()}
            else:
                X_dev = X.to(device, non_blocking=True)
            mask_dev = mod_mask.to(device, non_blocking=True)
            _, emb = model(X_dev, modality_mask=mask_dev, return_embeddings=True)
            feats.append(emb.cpu())
            masks.append(mod_mask.cpu())
    if not feats:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 0), dtype=np.float32)
    emb_np = torch.cat(feats, dim=0).numpy()
    mask_np = torch.cat(masks, dim=0).numpy()
    return emb_np.astype(np.float32, copy=False), mask_np.astype(np.float32, copy=False)


def _prepare_multimodal_inputs(data: pd.DataFrame, config: Dict, multimodal_cfg: Dict[str, Any]) -> Dict[str, Any]:
    image_info = multimodal_cfg.get("image") or {}
    sensor_info = multimodal_cfg.get("sensor") or {}
    image_raw = image_info.get("raw_assets") if isinstance(image_info, dict) else None
    sensor_raw = sensor_info.get("raw_assets") if isinstance(sensor_info, dict) else None

    if image_raw or sensor_raw:
        return _prepare_multimodal_inputs_raw(
            data,
            config,
            multimodal_cfg,
            image_raw,
            sensor_raw,
            image_info,
            sensor_info,
        )

    tab_info = multimodal_cfg.get("tabular", {}) or {}
    tab_df = tab_info.get("data")
    if tab_df is None:
        tab_df = data.copy()
    else:
        tab_df = tab_df.copy()

    id_col = tab_info.get("id_col") or multimodal_cfg.get("id_col") or config.get("id_col")
    if not id_col:
        raise ValueError("`id_col` must be provided for multimodal MySA training.")
    if id_col not in tab_df.columns:
        raise ValueError(f"Tabular modality must contain id column '{id_col}'.")

    tab_df[id_col] = canonicalize_series(tab_df[id_col]).fillna("")

    required_cols = {"duration", "event"}
    if not required_cols.issubset(tab_df.columns):
        missing = required_cols - set(tab_df.columns)
        raise ValueError(f"Tabular modality missing required columns: {missing}")

    tab_feat_cols = tab_info.get("feature_cols") or config.get("feature_cols") or []
    tab_feat_cols = [c for c in tab_feat_cols if c not in {id_col, "duration", "event"}]
    if not tab_feat_cols:
        tab_feat_cols = infer_feature_cols(tab_df, exclude=[id_col, "duration", "event"])

    n_bins = int(config.get("n_bins", 30))
    df = make_intervals(tab_df, duration_col="duration", event_col="event", n_bins=n_bins, method="quantile")
    if id_col in df.columns:
        df[id_col] = canonicalize_series(df[id_col]).fillna("")
    num_bins = int(df["interval_number"].max())

    base_ids = df[id_col].astype(str)
    id_index = pd.Index(base_ids)

    feature_arrays: List[np.ndarray] = []
    feature_names: List[str] = []
    modality_masks: List[np.ndarray] = []
    modality_slices: List[Tuple[str, slice]] = []

    start = 0

    tab_arr, tab_mask, tab_names = _align_modality_features(id_index, df, id_col, tab_feat_cols)
    if tab_arr.shape[1] > 0:
        feature_arrays.append(tab_arr)
        feature_names.extend(tab_names)
        modality_masks.append(np.ones_like(tab_mask) if tab_mask.size == 0 else tab_mask)
        modality_slices.append(("tabular", slice(start, start + tab_arr.shape[1])))
        start += tab_arr.shape[1]

    def _process_optional(name: str) -> None:
        nonlocal start
        info = multimodal_cfg.get(name)
        if not info or info.get("data") is None:
            return
        df_mod = info["data"].copy()
        mod_id = info.get("id_col") or id_col
        if mod_id not in df_mod.columns:
            raise ValueError(f"Modality '{name}' must contain id column '{mod_id}'.")
        df_mod[mod_id] = canonicalize_series(df_mod[mod_id]).fillna("")
        feat_cols = info.get("feature_cols")
        if not feat_cols:
            feat_cols = [c for c in df_mod.columns if c not in {mod_id, "duration", "event"}]
        arr, mask, names = _align_modality_features(id_index, df_mod, mod_id, feat_cols, prefix=f"{name}:")
        if arr.shape[1] == 0:
            return
        feature_arrays.append(arr)
        feature_names.extend(names)
        modality_masks.append(mask)
        modality_slices.append((name, slice(start, start + arr.shape[1])))
        start += arr.shape[1]

    _process_optional("image")
    _process_optional("sensor")

    if not feature_arrays:
        raise ValueError("No usable modality features found for multimodal MySA.")

    X_all = np.concatenate(feature_arrays, axis=1)
    mask_matrix = np.stack(modality_masks, axis=1) if modality_masks else None

    durations = df["duration"].to_numpy(np.float32)
    events = df["event"].to_numpy(np.int32)
    intervals = df["interval_number"].to_numpy(np.int32)

    N = len(df)
    idx = np.arange(N)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    split = int(N * (1.0 - float(config.get("val_ratio", 0.2))))
    tr, va = idx[:split], idx[split:]

    X_tr = X_all[tr]
    X_va = X_all[va]
    mask_tr = mask_matrix[tr] if mask_matrix is not None else None
    mask_va = mask_matrix[va] if mask_matrix is not None else None

    y_tr, m_tr = build_supervision(intervals[tr], events[tr], num_bins)
    y_va, m_va = build_supervision(intervals[va], events[va], num_bins)

    train_loader, val_loader = create_tensor_dataloaders(
        X_tr,
        y_tr,
        m_tr,
        X_va,
        y_va,
        m_va,
        batch_size=int(config.get("batch_size", 128)),
        num_workers=int(config.get("num_workers", 0)),
        mask_train=mask_tr,
        mask_val=mask_va,
    )

    model = MultiModalMySAModel(
        modality_slices,
        num_bins=num_bins,
        hidden=int(config.get("hidden", 256)),
        depth=int(config.get("depth", 2)),
        dropout=float(config.get("dropout", 0.2)),
    )

    feat2idx = {n: i for i, n in enumerate(feature_names)}

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "model": model,
        "feature_names": feature_names,
        "feat2idx": feat2idx,
        "num_bins": num_bins,
        "dur_va": torch.tensor(durations[va]),
        "evt_va": torch.tensor(events[va]),
        "X_train": X_tr,
        "X_val": X_va,
        "mask_train": mask_tr,
        "mask_val": mask_va,
    }


def _prepare_multimodal_inputs_raw(
    data: pd.DataFrame,
    config: Dict,
    multimodal_cfg: Dict[str, Any],
    image_raw: Optional[Dict[str, Any]],
    sensor_raw: Optional[Dict[str, Any]],
    image_info: Dict[str, Any],
    sensor_info: Dict[str, Any],
) -> Dict[str, Any]:
    tab_info = multimodal_cfg.get("tabular", {}) or {}
    tab_df = tab_info.get("data")
    if tab_df is None:
        tab_df = data.copy()
    else:
        tab_df = tab_df.copy()

    id_col = tab_info.get("id_col") or multimodal_cfg.get("id_col") or config.get("id_col")
    if not id_col:
        raise ValueError("`id_col` must be provided for multimodal MySA training.")
    if id_col not in tab_df.columns:
        raise ValueError(f"Tabular modality must contain id column '{id_col}'.")

    tab_df[id_col] = canonicalize_series(tab_df[id_col]).fillna("")

    required_cols = {"duration", "event"}
    if not required_cols.issubset(tab_df.columns):
        missing = required_cols - set(tab_df.columns)
        raise ValueError(f"Tabular modality missing required columns: {missing}")

    tab_feat_cols = tab_info.get("feature_cols") or config.get("feature_cols") or []
    tab_feat_cols = [c for c in tab_feat_cols if c not in {id_col, "duration", "event"}]
    if not tab_feat_cols:
        tab_feat_cols = infer_feature_cols(tab_df, exclude=[id_col, "duration", "event"])

    n_bins = int(config.get("n_bins", 30))
    df = make_intervals(tab_df, duration_col="duration", event_col="event", n_bins=n_bins, method="quantile")
    if id_col in df.columns:
        df[id_col] = canonicalize_series(df[id_col]).fillna("")
    num_bins = int(df["interval_number"].max())

    base_ids = df[id_col].astype(str)
    id_list = base_ids.tolist()
    # base_ids index not required beyond id list

    # Tabular matrix
    tab_numeric = df[tab_feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    tab_arr = tab_numeric.to_numpy(np.float32)

    # Prepare modality masks and asset paths
    modality_order: List[str] = ["tabular"]
    modality_masks = [np.ones(len(df), dtype=np.float32)]
    image_paths: Optional[List[Optional[str]]] = None
    sensor_paths: Optional[List[Optional[str]]] = None
    sensor_spec: Optional[SensorSpec] = None

    image_embed_dim = int(config.get("image_embed_dim", 256))
    sensor_embed_dim = int(config.get("sensor_embed_dim", 128))

    if image_raw:
        manifest = image_raw.get("manifest")
        root = image_raw.get("root")
        path_col = image_raw.get("path_col", "image")
        id_override = image_raw.get("id_col") or image_info.get("id_col") or id_col
        if manifest is not None and root:
            manifest = manifest.copy()
            manifest[id_override] = canonicalize_series(manifest[id_override]).fillna("")
            image_paths = build_image_paths_for_ids(
                id_list,
                manifest=manifest,
                id_col=id_override,
                path_col=path_col,
                root=root,
            )
            mask = np.array([1.0 if p else 0.0 for p in image_paths], dtype=np.float32)
            modality_order.append("image")
            modality_masks.append(mask)
        else:
            image_paths = None

    if sensor_raw:
        manifest = sensor_raw.get("manifest")
        root = sensor_raw.get("root")
        path_col = sensor_raw.get("path_col", "file")
        id_override = sensor_raw.get("id_col") or sensor_info.get("id_col") or id_col
        if manifest is not None and root:
            manifest = manifest.copy()
            manifest[id_override] = canonicalize_series(manifest[id_override]).fillna("")
            sensor_paths, sensor_spec = build_sensor_paths_for_ids(
                id_list,
                manifest=manifest,
                id_col=id_override,
                path_col=path_col,
                root=root,
            )
            if sensor_spec is not None:
                mask = np.array([1.0 if p else 0.0 for p in sensor_paths], dtype=np.float32)
                modality_order.append("sensor")
                modality_masks.append(mask)
            else:
                sensor_paths = None

    if len(modality_order) == 1:
        raise ValueError("Raw multimodal pipeline requires at least one additional modality.")

    mask_matrix = np.stack(modality_masks, axis=1)

    durations = df["duration"].to_numpy(np.float32)
    events = df["event"].to_numpy(np.int32)
    intervals = df["interval_number"].to_numpy(np.int32)

    N = len(df)
    idx = np.arange(N)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    split = int(N * (1.0 - float(config.get("val_ratio", 0.2))))
    tr, va = idx[:split], idx[split:]

    y_tr, m_tr = build_supervision(intervals[tr], events[tr], num_bins)
    y_va, m_va = build_supervision(intervals[va], events[va], num_bins)

    tab_tr = tab_arr[tr]
    tab_va = tab_arr[va]

    mask_tr = mask_matrix[tr]
    mask_va = mask_matrix[va]

    id_tr = [id_list[i] for i in tr]
    id_va = [id_list[i] for i in va]

    image_tr = [image_paths[i] if image_paths else None for i in tr]
    image_va = [image_paths[i] if image_paths else None for i in va]

    sensor_tr = [sensor_paths[i] if sensor_paths else None for i in tr] if sensor_paths else None
    sensor_va = [sensor_paths[i] if sensor_paths else None for i in va] if sensor_paths else None

    image_size = config.get("image_size", 224)
    if isinstance(image_size, (int, float)):
        image_size = (int(image_size), int(image_size))
    elif isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        image_size = (int(image_size[0]), int(image_size[1]))
    else:
        image_size = (224, 224)

    train_ds = AssetBackedMultiModalDataset(
        ids=id_tr,
        tabular=tab_tr,
        labels=y_tr,
        masks=m_tr,
        modality_mask=mask_tr,
        tabular_names=tab_feat_cols,
        image_paths=image_tr if image_paths is not None else None,
        image_size=image_size,
        sensor_paths=sensor_tr if sensor_paths is not None else None,
        sensor_spec=sensor_spec,
    )
    val_ds = AssetBackedMultiModalDataset(
        ids=id_va,
        tabular=tab_va,
        labels=y_va,
        masks=m_va,
        modality_mask=mask_va,
        tabular_names=tab_feat_cols,
        image_paths=image_va if image_paths is not None else None,
        image_size=image_size,
        sensor_paths=sensor_va if sensor_paths is not None else None,
        sensor_spec=sensor_spec,
    )

    batch_size = int(config.get("batch_size", 128))
    num_workers = int(config.get("num_workers", 0))
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=train_ds.collate_fn,
    )
    train_eval_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=train_ds.collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=val_ds.collate_fn,
    )

    # Configure model modalities
    feature_names: List[str] = []
    raw_modalities: List[Tuple[str, Dict[str, Any]]] = []

    tab_names = [f"tabular:{c}" for c in tab_feat_cols]
    feature_names.extend(tab_names)
    raw_modalities.append(
        (
            "tabular",
            {
                "input_dim": len(tab_feat_cols),
                "raw_dim": len(tab_feat_cols),
                "feature_names": tab_names,
            },
        )
    )

    if image_paths is not None:
        img_names = [f"image:feat_{i:03d}" for i in range(image_embed_dim)]
        feature_names.extend(img_names)
        raw_modalities.append(
            (
                "image",
                {
                    "raw_dim": image_embed_dim,
                    "feature_names": img_names,
                    "backbone": config.get("image_backbone", "resnet18"),
                    "pretrained": bool(config.get("image_pretrained", True)),
                },
            )
        )

    if sensor_paths is not None and sensor_spec is not None:
        sens_names = [f"sensor:feat_{i:03d}" for i in range(sensor_embed_dim)]
        feature_names.extend(sens_names)
        raw_modalities.append(
            (
                "sensor",
                {
                    "raw_dim": sensor_embed_dim,
                    "feature_names": sens_names,
                    "backbone": config.get("sensor_backbone", "cnn"),
                    "in_channels": sensor_spec.channels,
                    "target_len": sensor_spec.target_len,
                },
            )
        )

    model = MultiModalMySAModel(
        None,
        num_bins=num_bins,
        hidden=int(config.get("hidden", 256)),
        depth=int(config.get("depth", 2)),
        dropout=float(config.get("dropout", 0.2)),
        raw_modalities=raw_modalities,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    X_tr, mask_tr_np = _collect_embeddings(model, train_eval_loader, device)
    X_va, mask_va_np = _collect_embeddings(model, val_loader, device)

    feat2idx = {n: i for i, n in enumerate(feature_names)}

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "model": model,
        "feature_names": feature_names,
        "feat2idx": feat2idx,
        "num_bins": num_bins,
        "dur_va": torch.tensor(durations[va]),
        "evt_va": torch.tensor(events[va]),
        "X_train": X_tr,
        "X_val": X_va,
        "mask_train": mask_tr_np,
        "mask_val": mask_va_np,
    }


def hazards_to_survival(h: np.ndarray) -> np.ndarray:
    """Convert hazards [N, T] to survival curves [N, T] via S_t = Π_k≤t (1 - h_k)."""
    h = np.asarray(h, dtype=np.float32)
    N, T = h.shape
    S = np.ones_like(h)
    S[:, 0] = 1.0 - h[:, 0]
    for t in range(1, T):
        S[:, t] = S[:, t - 1] * (1.0 - h[:, t])
    return S


def topk_fi_table(phi_tbd: torch.Tensor, feature_names: List[str], k: int = 10) -> pd.DataFrame:
    """Aggregate TEXGI over (T,B) with abs, return sorted table."""
    imp_abs, _ = _aggregate_importance(phi_tbd)  # [D], [D]
    imp = imp_abs.detach().cpu().numpy()
    df = pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values("importance", ascending=False)
    return df.reset_index(drop=True).head(k)


# ------------------------------- Public API -----------------------------------


def _run_mysa_core(prepared: Dict[str, Any], config: Dict) -> Dict:
    train_loader = prepared["train_loader"]
    val_loader = prepared["val_loader"]
    model = prepared["model"]
    feature_names: List[str] = prepared["feature_names"]
    feat2idx = prepared["feat2idx"]
    dur_va: torch.Tensor = prepared["dur_va"]
    evt_va: torch.Tensor = prepared["evt_va"]
    X_tr = prepared["X_train"]
    X_va = prepared["X_val"]
    mask_tr = prepared.get("mask_train")
    mask_va = prepared.get("mask_val")

    expert_rules = config.get("expert_rules")
    legacy_feats = config.get("expert_features") or []
    if expert_rules is None and legacy_feats:
        expert_rules = {"rules": [{"feature": f, "relation": ">=mean", "weight": 1.0} for f in legacy_feats]}

    trainer = MySATrainer(
        model,
        lr=float(config.get("lr", 1e-3)),
        device=config.get("device", None),
        lambda_smooth=float(config.get("lambda_smooth", 0.0)),
        lambda_expert=float(config.get("lambda_expert", 0.0)),
        expert_rules=expert_rules,
        feat2idx=feat2idx,
        ig_steps=int(config.get("ig_steps", 20)),
        latent_dim=int(config.get("latent_dim", 16)),
        extreme_dim=int(config.get("extreme_dim", 1)),
        ig_batch_samples=int(config.get("ig_batch_samples", 64)),
        ig_time_subsample=config.get("ig_time_subsample"),
        gen_epochs=int(config.get("gen_epochs", 200)),
        gen_batch=int(config.get("gen_batch", 256)),
        gen_lr=float(config.get("gen_lr", 1e-3)),
        gen_alpha_dist=float(config.get("gen_alpha_dist", 1.0)),
        ref_stats=None,
        X_train_ref=torch.tensor(X_tr, dtype=torch.float32),
        modality_mask_ref=(torch.tensor(mask_tr, dtype=torch.float32) if mask_tr is not None else None),
    )

    capture_curves = bool(config.get("capture_training_curves", False))

    curve_history: List[Dict[str, Any]] = []
    time_bins = list(range(1, int(prepared["num_bins"]) + 1))
    max_curve_samples = max(1, int(config.get("curve_samples", 5)))

    def _infer_batch_size(batch_x: Any) -> int:
        if batch_x is None:
            return 0
        if torch.is_tensor(batch_x):
            return int(batch_x.shape[0])
        if isinstance(batch_x, dict):
            for v in batch_x.values():
                return _infer_batch_size(v)
            return 0
        if isinstance(batch_x, (list, tuple)) and batch_x:
            return _infer_batch_size(batch_x[0])
        return 0

    def _slice_first_k(obj: Any, k: int):
        if obj is None:
            return None
        if torch.is_tensor(obj):
            return obj[:k].detach().cpu()
        if isinstance(obj, dict):
            return {key: _slice_first_k(val, k) for key, val in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(_slice_first_k(val, k) for val in obj)
        return obj

    monitor_features = None
    monitor_mask = None
    sample_labels: List[str] = []
    sample_meta: List[Dict[str, Any]] = []

    if capture_curves:
        try:
            first_val_batch = next(iter(val_loader))
        except StopIteration:
            first_val_batch = None

        if first_val_batch is not None:
            X0, _, _, mask0 = trainer._unpack_batch(first_val_batch)
            batch_size = _infer_batch_size(X0)
            take = min(batch_size, max_curve_samples)
            monitor_features = _slice_first_k(X0, take)
            monitor_mask = _slice_first_k(mask0, take) if mask0 is not None else None

            dur_arr = prepared["dur_va"].detach().cpu().numpy() if torch.is_tensor(prepared["dur_va"]) else np.asarray(prepared["dur_va"])
            evt_arr = prepared["evt_va"].detach().cpu().numpy() if torch.is_tensor(prepared["evt_va"]) else np.asarray(prepared["evt_va"])

            for idx in range(take):
                label = f"Val #{idx + 1}"
                duration_val = float(dur_arr[idx]) if idx < len(dur_arr) else None
                event_val = int(evt_arr[idx]) if idx < len(evt_arr) else None
                if duration_val is not None and event_val is not None:
                    label = f"Val #{idx + 1} (T={duration_val:.0f}, E={event_val})"
                sample_labels.append(label)
                sample_meta.append({"duration": duration_val, "event": event_val})

    def _capture_epoch_curves(epoch_idx: int, val_c: float, losses: Dict[str, float]):
        if not capture_curves or monitor_features is None:
            return
        with torch.no_grad():
            trainer.model.eval()
            X_dev = trainer._move_to_device(monitor_features)
            mask_dev = trainer._move_to_device(monitor_mask) if monitor_mask is not None else None
            out = trainer._model_forward(X_dev, mask_dev)
            hazards_tensor: torch.Tensor
            if isinstance(out, tuple):
                hazards_tensor = out[0]
            else:
                hazards_tensor = out
            hazards_np = hazards_tensor.detach().cpu().numpy()
            hazards_np = hazards_np[: len(sample_labels)]
            survival_np = hazards_to_survival(hazards_np)
        curve_history.append(
            {
                "epoch": int(epoch_idx),
                "val_cindex": float(val_c),
                "train_nll": float(losses.get("nll", 0.0)),
                "train_smooth": float(losses.get("smooth", 0.0)),
                "train_expert": float(losses.get("expert", 0.0)),
                "hazards": hazards_np.tolist(),
                "survival": survival_np.tolist(),
            }
        )

    best = {"val_cindex": 0.0, "epoch": -1}
    epochs = int(config.get("epochs", 200))
    for ep in range(1, epochs + 1):
        lm = ls = le = 0.0
        steps = 0
        for batch in train_loader:
            lmm, lss, lee = trainer.step(batch)
            lm += lmm
            ls += lss
            le += lee
            steps += 1
        steps = max(steps, 1)
        lm /= steps
        ls /= steps
        le /= steps

        val_c = trainer.evaluate_cindex(val_loader, dur_va, evt_va)
        _capture_epoch_curves(ep, val_c, {"nll": lm, "smooth": ls, "expert": le})
        trainer.sched.step(1.0 - val_c)

        if val_c > best["val_cindex"]:
            best = {"val_cindex": val_c, "epoch": ep}
            torch.save(model.state_dict(), "model.pt")

        print(
            f"[MySA] Ep{ep:03d}  NLL={lm:.4f}  Smooth={ls:.4f}  Expert={le:.4f}  | Val C-index={val_c:.4f}"
        )

    device = trainer.device
    model.eval()
    all_h = []
    with torch.no_grad():
        for batch in val_loader:
            Xb, _, _, maskb = trainer._unpack_batch(batch)
            Xb = trainer._move_to_device(Xb)
            maskb_t = maskb.to(device, non_blocking=True) if maskb is not None else None
            hb = trainer._model_forward(Xb, maskb_t).detach().cpu().numpy()
            all_h.append(hb)
    H_va = np.concatenate(all_h, axis=0)
    S_va = hazards_to_survival(H_va)
    surv_df = pd.DataFrame(S_va.T)
    surv_df.index.name = "time_bin"

    if capture_curves and curve_history:
        curve_payload = {
            "time_bins": time_bins,
            "sample_ids": sample_labels,
            "sample_metadata": sample_meta,
            "entries": curve_history,
        }
    else:
        curve_payload = None

    Xva_t = torch.tensor(X_va, device=device)
    mask_va_t = torch.tensor(mask_va, device=device) if mask_va is not None else None
    Ns = min(len(Xva_t), int(config.get("fi_samples", 256)))
    Xsub = Xva_t[:Ns]
    forward_kwargs = {"modality_mask": mask_va_t[:Ns]} if mask_va_t is not None else None
    phi_val = texgi_time_series(
        model.to(device),
        Xsub,
        trainer.G,
        trainer.ref_stats,
        M=trainer.ig_steps,
        latent_dim=trainer.latent_dim,
        extreme_dim=trainer.extreme_dim,
        forward_kwargs=forward_kwargs,
    )

    imp_abs, imp_dir = _aggregate_importance(phi_val)
    fi_global = pd.DataFrame({
        "feature": feature_names,
        "importance": imp_abs.detach().cpu().numpy(),
        "directional_mean": imp_dir.detach().cpu().numpy(),
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    torch.save({"phi_val": phi_val.detach().cpu(), "feature_names": feature_names}, "mysa_phi_val.pt")

    return {
        "C-index (Validation)": float(best["val_cindex"]),
        "Num Bins": int(prepared["num_bins"]),
        "Val Samples": int(len(X_va)),
        "Surv_Test": surv_df,
        "training_curve_history": curve_payload,
        "Feature Importance": fi_global,
        "Expert Rules Used": expert_rules or {},
        "Best Epoch": int(best["epoch"]),
        "Phi_Val_Path": "mysa_phi_val.pt",
    }


def run_mysa(data: pd.DataFrame, config: Dict) -> Dict:
    """
    Train & evaluate MySA (full) model.
    Supports tabular and multimodal inputs (with `config["multimodal_sources"]`).
    """
    multimodal_cfg = config.get("multimodal_sources")
    if multimodal_cfg:
        prepared = _prepare_multimodal_inputs(data, config, multimodal_cfg)
    else:
        prepared = _prepare_tabular_inputs(data, config)
    return _run_mysa_core(prepared, config)


def run_texgisa(data: pd.DataFrame, config: Dict) -> Dict:
    """Alias for compatibility with existing UI expecting 'texgisa' entry point."""
    return run_mysa(data, config)

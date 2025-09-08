
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

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

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


def smooth_l1_temporal(hazards: torch.Tensor, lambda_smooth: float) -> torch.Tensor:
    """
    Total-variation-like temporal smoothing on hazards.
    hazards: [B, T]
    """
    if lambda_smooth <= 0:
        return hazards.new_tensor(0.0)
    diff = hazards[:, 1:] - hazards[:, :-1]
    return lambda_smooth * F.smooth_l1_loss(diff, torch.zeros_like(diff))


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
            with torch.no_grad():
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
) -> torch.Tensor:
    """
    Compute IG for one time index t along the straight path from baseline to input.
    Returns: IG attributions [B, D] for this t.
    """
    assert X.shape == X_baseline.shape
    device = X.device
    alphas = torch.linspace(0.0, 1.0, steps=M + 1, device=device)[1:]  # exclude 0
    Xdiff = (X - X_baseline)

    atts = torch.zeros_like(X)
    for a in alphas:
        Xpath = X_baseline + a * Xdiff
        Xpath.requires_grad_(True)
        hazards = f(Xpath)                      # [B, T]
        out = hazards[:, hazard_index]          # focus on bin t
        grads = torch.autograd.grad(out.sum(), Xpath, retain_graph=False, create_graph=False)[0]
        atts += grads
    atts = atts * Xdiff / float(len(alphas))
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
) -> torch.Tensor:
    """
    Compute TEXGI per time-bin, returning phi with shape [T, B, D].
    Baseline is produced by adversarial generator conditioned on (x, z, e).
    """
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
        T = f(X[:1]).shape[1]

    # 选择本次要参与的时间 bin（随机/等间隔均可）
    if t_sample is not None and t_sample > 0 and t_sample < T:
        # 等间隔抽样更稳定
        import math
        step = max(1, math.floor(T / t_sample))
        t_indices = list(range(0, T, step))[:t_sample]
    else:
        t_indices = list(range(T))

    phi_list = []
    for t in t_indices:
        ig_t = integrated_gradients_time(f, X, X_baseline, hazard_index=t, M=M)
        phi_list.append(ig_t)

    # [T', B, D] （如果抽样，T' < T）
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


def expert_penalty(
    phi_tbd: torch.Tensor,                  # [T,B,D]
    expert_config: Dict[str, Any],          # rules per feature name
    feat2idx: Dict[str, int],
) -> torch.Tensor:
    """
    Support rules:
      - relation: '>=mean' or '<=mean'  (relative to global mean importance of all features)
      - sign: -1, 0, +1                 (directional expectation)
      - min_mag: float                   (minimum |directional mean|)
      - weight: float                    (per feature weight in penalty aggregation)
    expert_config example:
      {
        "rules": [
           {"feature":"AGE", "relation":">=mean", "sign":+1, "min_mag":0.01, "weight":1.0},
           {"feature":"BUN", "relation":"<=mean", "sign":0, "weight":0.5}
        ]
      }
    """
    device = phi_tbd.device
    imp_abs, imp_dir = _aggregate_importance(phi_tbd)  # [D], [D]
    mean_abs = imp_abs.mean()

    total = torch.zeros((), device=device)
    rules = expert_config.get("rules", []) if expert_config else []
    for rule in rules:
        fname = rule.get("feature")
        if fname not in feat2idx:
            continue
        j = feat2idx[fname]
        w = float(rule.get("weight", 1.0))

        # relation constraint
        rel = rule.get("relation", None)
        if rel == ">=mean":
            total = total + w * F.relu(mean_abs - imp_abs[j])
        elif rel == "<=mean":
            total = total + w * F.relu(imp_abs[j] - mean_abs)

        # directional expectation
        sign = int(rule.get("sign", 0))
        if sign != 0:
            total = total + w * F.relu(-sign * imp_dir[j])

        # minimum magnitude on directional mean
        min_mag = float(rule.get("min_mag", 0.0))
        if min_mag > 0.0:
            total = total + w * F.relu(min_mag - imp_dir[j].abs())

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
        ig_batch_samples: int = 64,                 # 每个 batch 里参与 TEXGI 的样本上限
        ig_time_subsample: Optional[int] = None,    # 每个 batch 里参与 TEXGI 的时间 bin 数上限（None=全时间）
        # Generator settings:
        gen_epochs: int = 200,
        gen_batch: int = 256,
        gen_lr: float = 1e-3,
        gen_alpha_dist: float = 1.0,
        ref_stats: Optional[Dict[str, torch.Tensor]] = None,
        X_train_ref: Optional[torch.Tensor] = None,  # real-space training features for generator fit
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode="min", factor=0.5, patience=5, verbose=False)

        self.lambda_smooth = float(lambda_smooth)
        self.lambda_expert = float(lambda_expert)
        self.expert_rules = expert_rules or {}
        self.feat2idx = feat2idx or {}

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
        self.ig_batch_samples = int(ig_batch_samples)
        self.ig_time_subsample = (int(ig_time_subsample) if ig_time_subsample
                                  else None)
        

    def fit_generator_if_needed(self):
        if self.lambda_expert <= 0:
            return  # no expert penalty, generator not required
        if self.G is not None:
            return
        # 若磁盘上已有已训好的生成器，直接加载（加速多次实验）
        import os, torch
        ckpt_path = "mysa_G.pt"
        if os.path.exists(ckpt_path):
            try:
                self.G = TabularGeneratorWithRealInput(self.X_train_ref.shape[1],
                                                       self.latent_dim, self.extreme_dim).to(self.device)
                self.G.load_state_dict(torch.load(ckpt_path, map_location=self.device))
                return
            except Exception:
                self.G = None  # 加载失败则正常重训

        assert self.X_train_ref is not None, "X_train_ref must be provided to fit generator."
        Xtr = self.X_train_ref.to(self.device)
        mu, std = _standardize_fit(Xtr)
        self.G, self.ref_stats = train_adv_generator(
            self.model, Xtr, mu, std,
            epochs=self.gen_epochs, batch_size=self.gen_batch,
            latent_dim=self.latent_dim, extreme_dim=self.extreme_dim,
            lr=self.gen_lr, alpha_dist=self.gen_alpha_dist, device=self.device
        )

    def step(self, batch) -> Tuple[float, float, float]:
        X, y, m = batch
        X = X.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        m = m.to(self.device, non_blocking=True)

        self.model.train()
        hazards = self.model(X)  # [B, T]
        loss_main = masked_bce_nll(hazards, y, m)
        loss_smooth = smooth_l1_temporal(hazards, self.lambda_smooth)
        loss = loss_main + loss_smooth

        loss_expert = X.new_tensor(0.0)
        if self.lambda_expert > 0 and self.expert_rules and self.feat2idx:
            # Ensure generator fitted
            self.fit_generator_if_needed()
            with torch.enable_grad():
                # Subsample for IG to control cost
                B = X.shape[0]
                sub = min(B, self.ig_batch_samples)
                idx = torch.randperm(B, device=X.device)[:sub]
                Xsub = X[idx].detach().clone().requires_grad_(True)
                phi = texgi_time_series(
                    self.model, Xsub, self.G, self.ref_stats,
                    M=self.ig_steps, latent_dim=self.latent_dim, extreme_dim=self.extreme_dim,
                    t_sample=self.ig_time_subsample
                )

                loss_Omega = expert_penalty(phi, self.expert_rules, self.feat2idx)
                loss_expert = self.lambda_expert * loss_Omega
                loss = loss + loss_expert

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.opt.step()
        return loss_main.item(), loss_smooth.item(), loss_expert.item()

    @torch.no_grad()
    def evaluate_cindex(self, loader, durations, events) -> float:
        self.model.eval()
        all_risk = []
        for X, y, m in loader:
            X = X.to(self.device, non_blocking=True)
            hazards = self.model(X)
            risk = torch.sum(hazards, dim=1)  # simple risk proxy; larger -> earlier event
            all_risk.append(risk.cpu())
        risks = torch.cat(all_risk, dim=0)
        return cindex_fast(durations, events, risks).item()


# ------------------------------ Utilities -------------------------------------

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

def run_mysa(data: pd.DataFrame, config: Dict) -> Dict:
    """
    Train & evaluate MySA (full) model.
    Required columns in `data`: 'duration', 'event' + feature columns.
    config keys (examples):
      - n_bins (30), val_ratio (0.2), batch_size (128), epochs (200)
      - lr (1e-3), hidden (256), depth (2), dropout (0.2)
      - lambda_smooth (0.01), lambda_expert (0.01)
      - expert_features (list[str])           # backward compat (treated as relation >=mean, weight=1.0)
      - expert_rules (dict)                    # rich rules; see expert_penalty docstring
      - feature_cols (list[str])               # else infer from data
      - ig_steps (20), latent_dim (16), extreme_dim (1)
      - gen_epochs (200), gen_batch (256), gen_lr (1e-3), gen_alpha_dist (1.0)
      - num_workers (0)
    Returns: dict with metrics + survival curves + FI tables + per-time FI tensor path.
    """
    df = data.copy().reset_index(drop=True)
    required_cols = {"duration", "event"}
    assert required_cols.issubset(df.columns), f"Missing columns: {required_cols - set(df.columns)}"

    # Time discretization
    n_bins = int(config.get("n_bins", 30))
    df = make_intervals(df, duration_col="duration", event_col="event", n_bins=n_bins, method="quantile")
    num_bins = int(df["interval_number"].max())

    # Features
    feat_cols = config.get("feature_cols")
    if not feat_cols:
        feat_cols = infer_feature_cols(df, exclude=["duration", "event"])
    feat2idx = {n: i for i, n in enumerate(feat_cols)}

    # Split
    val_ratio = float(config.get("val_ratio", 0.2))
    N = len(df)
    idx = np.arange(N)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    split = int(N * (1.0 - val_ratio))
    tr, va = idx[:split], idx[split:]

    durations = df["duration"].to_numpy(np.float32)
    events    = df["event"].to_numpy(np.int32)
    intervals = df["interval_number"].to_numpy(np.int32)
    X_all     = df[feat_cols].to_numpy(np.float32)

    X_tr = X_all[tr]; X_va = X_all[va]
    dur_va = torch.tensor(durations[va])
    evt_va = torch.tensor(events[va])

    # Supervision
    y_tr, m_tr = build_supervision(intervals[tr], events[tr], num_bins)
    y_va, m_va = build_supervision(intervals[va], events[va], num_bins)

    # Dataloaders
    train_loader, val_loader = create_dataloaders(
        X_tr, y_tr, m_tr, X_va, y_va, m_va,
        batch_size=int(config.get("batch_size", 128)),
        num_workers=int(config.get("num_workers", 0)),
    )

    # Model
    model = MultiTaskModel(
        input_dim=X_tr.shape[1],
        num_bins=num_bins,
        hidden=int(config.get("hidden", 256)),
        depth=int(config.get("depth", 2)),
        dropout=float(config.get("dropout", 0.2)),
    )

    # Expert rules: support legacy "expert_features"
    expert_rules = config.get("expert_rules")
    legacy_feats = config.get("expert_features") or []
    if expert_rules is None and legacy_feats:
        expert_rules = {"rules": [{"feature": f, "relation": ">=mean", "weight": 1.0} for f in legacy_feats]}

    # Prepare generator reference tensors
    X_train_ref = torch.tensor(X_tr)

    # Trainer
    trainer = MySATrainer(
        model,
        lr=float(config.get("lr", 1e-3)),
        device=config.get("device", None),
        lambda_smooth=float(config.get("lambda_smooth", 0.0)),
        lambda_expert=float(config.get("lambda_expert", 0.0)),
        expert_rules=expert_rules,
        feat2idx=feat2idx,
        # TEXGI
        ig_steps=int(config.get("ig_steps", 20)),
        latent_dim=int(config.get("latent_dim", 16)),
        extreme_dim=int(config.get("extreme_dim", 1)),
        # Generator
        gen_epochs=int(config.get("gen_epochs", 200)),
        gen_batch=int(config.get("gen_batch", 256)),
        gen_lr=float(config.get("gen_lr", 1e-3)),
        gen_alpha_dist=float(config.get("gen_alpha_dist", 1.0)),
        ref_stats=None,
        X_train_ref=X_train_ref,
    )

    best = {"val_cindex": 0.0, "epoch": -1}
    epochs = int(config.get("epochs", 200))
    for ep in range(1, epochs + 1):
        lm, ls, le, steps = 0.0, 0.0, 0.0, 0
        for batch in train_loader:
            lmm, lss, lee = trainer.step(batch)
            lm += lmm; ls += lss; le += lee; steps += 1
        lm /= max(steps, 1); ls /= max(steps, 1); le /= max(steps, 1)

        # Validation
        val_c = trainer.evaluate_cindex(val_loader, dur_va, evt_va)
        trainer.sched.step(1.0 - val_c)

        if val_c > best["val_cindex"]:
            best = {"val_cindex": val_c, "epoch": ep}
            torch.save(model.state_dict(), "model.pt")

        print(f"[MySA] Ep{ep:03d}  NLL={lm:.4f}  Smooth={ls:.4f}  Expert={le:.4f}  | Val C-index={val_c:.4f}")

    # Final pass on validation for hazards & TEXGI
    device = trainer.device
    model.eval()
    all_h = []
    Xva_t = torch.tensor(X_va, device=device)
    with torch.no_grad():
        for Xb, yb, mb in val_loader:
            hb = model(Xb.to(device)).cpu().numpy()
            all_h.append(hb)
    H_va = np.concatenate(all_h, axis=0)  # [Nv, T]
    S_va = hazards_to_survival(H_va)       # [Nv, T]
    surv_df = pd.DataFrame(S_va.T)         # time-major
    surv_df.index.name = "time_bin"

    # TEXGI on a validation subset for FI tables
    Ns = min(len(Xva_t), int(config.get("fi_samples", 256)))
    Xsub = Xva_t[:Ns]
    phi_val = texgi_time_series(
        model.to(device), Xsub, trainer.G, trainer.ref_stats,
        M=int(config.get("ig_steps", 20)),
        latent_dim=int(config.get("latent_dim", 16)),
        extreme_dim=int(config.get("extreme_dim", 1))
    )  # [T, Ns, D]

    # Global FI and per-time FI
    imp_abs, imp_dir = _aggregate_importance(phi_val)  # [D],[D]
    fi_global = pd.DataFrame({
        "feature": feat_cols,
        "importance": imp_abs.detach().cpu().numpy(),
        "directional_mean": imp_dir.detach().cpu().numpy()
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # Optionally return the raw tensor path to avoid huge JSON
    # Save phi_val as torch file for later drill-down in UI
    torch.save({"phi_val": phi_val.detach().cpu(), "feature_names": feat_cols}, "mysa_phi_val.pt")

    return {
        "C-index (Validation)": float(best["val_cindex"]),
        "Num Bins": int(num_bins),
        "Val Samples": int(len(X_va)),
        "Surv_Test": surv_df,
        "Feature Importance": fi_global,          # table (abs importance + directional mean)
        "Expert Rules Used": expert_rules or {},
        "Best Epoch": int(best["epoch"]),
        "Phi_Val_Path": "mysa_phi_val.pt",        # raw time-dependent attributions
    }


def run_texgisa(data: pd.DataFrame, config: Dict) -> Dict:
    """Alias for compatibility with existing UI expecting 'texgisa' entry point."""
    return run_mysa(data, config)

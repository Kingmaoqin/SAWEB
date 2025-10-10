import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import make_intervals, build_supervision, infer_feature_cols, MultiTaskDataset
from model import MultiTaskModel
from trainer import masked_bce_nll


def integrated_brier_score(hazards: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Compute Integrated Brier Score over discrete time bins.

    Parameters
    ----------
    hazards : Tensor [N, T]
        Predicted hazard probabilities per time bin.
    labels : Tensor [N, T]
        One-hot event indicators.
    masks : Tensor [N, T]
        Risk-set masks indicating valid time bins for each sample.
    """
    # survival probability per bin
    surv_prob = torch.cumprod(1 - hazards, dim=1)

    # ground truth survival indicator: 1 until event occurs, else 0
    gt_surv = torch.ones_like(hazards)
    event_bins = labels.argmax(dim=1)
    has_event = labels.sum(dim=1) > 0
    for i in torch.nonzero(has_event).view(-1):
        k = event_bins[i]
        gt_surv[i, k:] = 0.0

    brier = (surv_prob - gt_surv) ** 2
    return (brier * masks).sum() / masks.sum().clamp_min(1.0)


def integrated_nll(hazards: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Integrated negative log-likelihood using discrete-time hazards."""
    eps = 1e-6
    hazards = hazards.clamp(eps, 1 - eps)
    nll = -(labels * torch.log(hazards) + (1 - labels) * torch.log(1 - hazards))
    return (nll * masks).sum() / masks.sum().clamp_min(1.0)


def run_gabs(path: str = "gabs.csv", n_bins: int = 30, epochs: int = 10, batch_size: int = 32):
    """Train MultiTaskModel on the GABS dataset and report IBS and INLL on the test set."""
    df = pd.read_csv(path)
    df = make_intervals(df, "duration", "event", n_bins=n_bins)
    feature_cols = infer_feature_cols(df, exclude=["duration", "event"])

    X = df[feature_cols].to_numpy(dtype=np.float32)
    intervals = df["interval_number"].to_numpy()
    events = df["event"].to_numpy()
    labels, masks = build_supervision(intervals, events, n_bins)

    N = len(df)
    rng = np.random.default_rng(42)
    idx = np.arange(N)
    rng.shuffle(idx)
    split = int(0.8 * N)
    train_idx, test_idx = idx[:split], idx[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    m_train, m_test = masks[train_idx], masks[test_idx]

    train_loader = DataLoader(MultiTaskDataset(X_train, y_train, m_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(MultiTaskDataset(X_test, y_test, m_test), batch_size=batch_size, shuffle=False)

    model = MultiTaskModel(input_dim=X.shape[1], num_bins=n_bins)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(epochs):
        for xb, yb, mb in train_loader:
            opt.zero_grad()
            hazards = model(xb)
            loss = masked_bce_nll(hazards, yb, mb)
            loss.backward()
            opt.step()

    model.eval()
    all_h = []
    with torch.no_grad():
        for xb, _, _ in test_loader:
            all_h.append(model(xb))
    hazards = torch.cat(all_h, dim=0)
    y_test_t = torch.from_numpy(y_test)
    m_test_t = torch.from_numpy(m_test)

    ibs = integrated_brier_score(hazards, y_test_t, m_test_t)
    inll = integrated_nll(hazards, y_test_t, m_test_t)
    print(f"Integrated Brier Score: {ibs.item():.4f}")
    print(f"Integrated NLL: {inll.item():.4f}")


if __name__ == "__main__":
    run_gabs()
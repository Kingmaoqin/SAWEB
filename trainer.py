import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from typing import Optional, Dict, Any

from metrics import cindex_fast

__all__ = ["masked_bce_nll", "smooth_l1_temporal", "Trainer"]

def masked_bce_nll(hazards, labels, masks):
    """
    hazards: [B, T] in (0,1)
    labels : [B, T] one-hot at event bin or zeros if censored
    masks  : [B, T] risk-set mask (1 up to interval, else 0)
    BCE per bin equals discrete-time NLL when applied with correct labels/masks.
    """
    eps = 1e-6
    hazards = hazards.clamp(eps, 1 - eps)
    bce = -(labels * torch.log(hazards) + (1 - labels) * torch.log(1 - hazards))
    # Masked average
    denom = masks.sum().clamp_min(1.0)
    return (bce * masks).sum() / denom


def smooth_l1_temporal(hazards, weight=0.0):
    """
    L1 smoothing across adjacent time bins to stabilize training.
    """
    if weight <= 0.0:
        return hazards.new_tensor(0.0)
    diff = torch.abs(hazards[:, 1:] - hazards[:, :-1])
    return weight * diff.mean()


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        device: Optional[str] = None,
        lambda_smooth: float = 0.0,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.opt = Adam(self.model.parameters(), lr=lr)
        self.sched = ReduceLROnPlateau(self.opt, mode="min", factor=0.5, patience=5, verbose=True, min_lr=1e-6)
        self.scaler = GradScaler(enabled=True)
        self.lambda_smooth = float(lambda_smooth)

    def _move_to_device(self, obj: Any):
        if torch.is_tensor(obj):
            return obj.to(self.device, non_blocking=True)
        if isinstance(obj, dict):
            return {k: self._move_to_device(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(*(self._move_to_device(v) for v in obj))
        return obj

    def _get(self, batch: Any, name: str):
        if isinstance(batch, dict):
            return batch[name]
        if hasattr(batch, name):
            return getattr(batch, name)
        if isinstance(batch, (list, tuple)):
            idx_map = {"y": 1, "m": 2}
            if name in idx_map and len(batch) > idx_map[name]:
                return batch[idx_map[name]]
        raise KeyError(f"Field '{name}' not found in batch")

    def step(self, batch):
        if isinstance(batch, (list, tuple)) and not hasattr(batch, "_fields"):
            X, y, m, mod_mask = batch
        else:
            X       = self._get(batch, "x")
            y       = self._get(batch, "y")
            m       = self._get(batch, "m")
            mod_mask = self._get(batch, "mod_mask")

        X        = self._move_to_device(X)
        y        = y.to(self.device, non_blocking=True)
        m        = m.to(self.device, non_blocking=True)
        mod_mask = mod_mask.to(self.device, non_blocking=True)

        with autocast(enabled=True):
            hazards     = self.model(X, mod_mask)           # [B, T]
            loss_main   = masked_bce_nll(hazards, y, m)
            loss_smooth = smooth_l1_temporal(hazards, self.lambda_smooth)
            loss        = loss_main + loss_smooth

        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.scaler.step(self.opt)
        self.scaler.update()
        return loss_main.detach().item(), loss_smooth.detach().item()


    @torch.no_grad()
    def evaluate(self, loader, durations, events):
        self.model.eval()
        all_risk = []
        for batch in loader:
            if isinstance(batch, (list, tuple)) and not hasattr(batch, "_fields"):
                X, _, _, mod_mask = batch
            else:
                X       = self._get(batch, "x")
                mod_mask = self._get(batch, "mod_mask")

            X        = self._move_to_device(X)
            mod_mask = mod_mask.to(self.device, non_blocking=True)
            hazards  = self.model(X, mod_mask)               # [B, T]
            risk     = hazards.sum(dim=1)                    # [B]
            all_risk.append(risk.cpu())

        risks = torch.cat(all_risk, dim=0)
        c = cindex_fast(durations, events, risks)
        return c.item()

    def fit(self, train_loader, val_loader, val_durations, val_events, epochs=50):
        best = {"val_cindex": 0.0, "epoch": -1}
        for ep in range(1, epochs + 1):
            self.model.train()
            loss_m, loss_s, steps = 0.0, 0.0, 0
            for batch in train_loader:
                lm, ls = self.step(batch)
                loss_m += lm; loss_s += ls; steps += 1
            loss_m /= max(steps, 1); loss_s /= max(steps, 1)

            # Validation
            val_c = self.evaluate(val_loader, val_durations, val_events)
            val_loss_proxy = 1.0 - val_c  # for plateau scheduler
            self.sched.step(val_loss_proxy)

            print(f"Epoch {ep:03d} | train_nll={loss_m:.4f} + smooth={loss_s:.4f} | val_cindex={val_c:.4f}")

            if val_c > best["val_cindex"]:
                best = {"val_cindex": val_c, "epoch": ep}
                torch.save(self.model.state_dict(), "model.pt")
        print(f"Best val C-index={best['val_cindex']:.4f} @ epoch {best['epoch']}")

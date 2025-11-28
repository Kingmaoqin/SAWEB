import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MultiTaskModel"]

class MultiTaskModel(nn.Module):
    """
    Shared MLP trunk -> per-time-bin hazards via linear head over shared features.
    Returns hazards in shape [B, T].
    """

    def __init__(self, input_dim: int, num_bins: int, hidden: int = 256, depth: int = 2, dropout: float = 0.2):
        super().__init__()
        layers = []
        d = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))     # << 加这行
            d = hidden
        self.trunk = nn.Sequential(*layers)
        self.hazard_layer = nn.Linear(d, num_bins)
        self.num_bins = num_bins
        # Xavier init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, mod_mask=None, return_embeddings: bool = False):
        """Run the shared trunk and hazard head.

        Parameters
        ----------
        x: torch.Tensor
            Input features of shape ``[B, D]``.
        mod_mask: torch.Tensor, optional
            Unused placeholder to maintain compatibility with callers that pass
            modality masks; present to provide a unified risk interface for
            single- and multi-modal models.
        return_embeddings: bool, default False
            When ``True`` also return the hidden representation produced by the
            shared trunk so downstream routines can reuse it for attribution
            without re-running the model.
        """

        h = self.trunk(x)
        logits = self.hazard_layer(h)
        hazards = torch.sigmoid(logits)
        if return_embeddings:
            return hazards, h
        return hazards

    @torch.no_grad()
    def predict_risk(self, x, mod_mask=None, interval: int | None = None):
        """Return per-sample risk scores.

        By default, risk is the sum of hazards across all intervals; when an
        ``interval`` is specified (1-indexed) the method returns the hazard for
        that interval to support interval-specific risk inspection.
        """

        hazards = self.forward(x, mod_mask)
        if interval is not None:
            idx = max(0, min(int(interval) - 1, hazards.shape[1] - 1))
            return hazards[:, idx]
        return hazards.sum(dim=1)

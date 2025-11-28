import torch
import torch.nn as nn
from typing import Dict, Optional


class MultiModalSurvivalModel(nn.Module):
    """Simple multimodal survival model with gating-based fusion.

    Each modality has a dedicated encoder producing a representation of the
    same dimensionality. During ``forward`` the model accepts a dictionary of
    modality inputs and a ``modality_masks`` tensor indicating which modalities
    are present for each sample. Missing modalities are zeroed-out and excluded
    from the fusion weights. The fusion is performed via a lightweight gating
    mechanism similar to a Gated Multimodal Unit.
    """

    def __init__(self, encoders: Dict[str, nn.Module], hidden_dim: int, num_bins: int):
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.modality_names = list(encoders.keys())
        self.hidden_dim = hidden_dim

        # gating layers produce a single logit per modality
        self.gates = nn.ModuleDict({name: nn.Linear(hidden_dim, 1) for name in self.modality_names})
        # final hazard head
        self.hazard_layer = nn.Linear(hidden_dim, num_bins)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, inputs: Dict[str, torch.Tensor], modality_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with modality masking.

        Parameters
        ----------
        inputs: Dict[str, Tensor]
            Mapping from modality name to its input tensor.
        modality_masks: Tensor of shape [B, M], optional
            1 indicates the modality is available for a given sample. Missing
            modalities are zeroed-out and ignored in fusion.
        """
        B = next(iter(inputs.values())).shape[0]
        M = len(self.modality_names)
        if modality_masks is None:
            modality_masks = inputs[next(iter(inputs))].new_ones(B, M)

        feats = []
        gate_logits = []
        for i, name in enumerate(self.modality_names):
            x = inputs.get(name)
            h = self.encoders[name](x)
            mask = modality_masks[:, i].unsqueeze(1)
            h = h * mask  # zero-out missing modality
            feats.append(h)
            gate_logits.append(self.gates[name](h))

        gate_logits = torch.cat(gate_logits, dim=1)
        # mask out missing modalities by setting their logits to a large negative value
        gate_logits = gate_logits.masked_fill(modality_masks == 0, -1e4)
        attn = torch.softmax(gate_logits, dim=1)  # [B, M]

        fused = torch.zeros(B, self.hidden_dim, device=attn.device)
        for i in range(M):
            fused = fused + feats[i] * attn[:, i].unsqueeze(1)

        hazards = torch.sigmoid(self.hazard_layer(fused))
        return hazards

    @torch.no_grad()
    def predict_risk(self, inputs: Dict[str, torch.Tensor], modality_masks: Optional[torch.Tensor] = None, interval: int | None = None) -> torch.Tensor:
        """Return per-sample risk scores or interval hazards for CF/UX layers."""

        hazards = self.forward(inputs, modality_masks)
        if interval is not None:
            idx = max(0, min(int(interval) - 1, hazards.shape[1] - 1))
            return hazards[:, idx]
        return hazards.sum(dim=1)

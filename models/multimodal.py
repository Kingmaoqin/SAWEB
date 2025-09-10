"""Multi-modal survival model.

This module implements :class:`MultiModalSurvivalModel` which fuses image,
sensor and tabular modalities and predicts per-time-bin hazards.  The design is
kept intentionally light‑weight so it can act as a reference implementation that
can be swapped out with more sophisticated encoders later on.

The model consists of three encoders:

* :class:`ImageEncoder` – wraps a torchvision backbone (ResNet50 or ViT) and
  outputs a single feature vector per image.
* :class:`SensorEncoder` – encodes 1D sensor sequences via either a simple CNN
  stack or a Transformer encoder.
* :class:`TabularEncoder` – an MLP for dense tabular features.

The resulting embeddings are concatenated and projected to a shared hidden
space before being passed to the existing :class:`~model.MultiTaskModel` hazard
head which outputs hazards of shape ``[B, T]``.

In addition the model exposes :func:`load_pretrained_backbones` allowing users
to load pre–trained weights for the image and sensor encoders.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from model import MultiTaskModel


class ImageEncoder(nn.Module):
    """Encode images using a torchvision backbone.

    Parameters
    ----------
    backbone: str
        Either ``"resnet50"`` or ``"vit"`` (ViT-B/16).
    pretrained: bool
        If ``True`` use ImageNet weights.
    """

    def __init__(self, backbone: str = "resnet50", pretrained: bool = True):
        super().__init__()
        name = backbone.lower()
        if name == "resnet50":
            from torchvision.models import resnet50, ResNet50_Weights

            weights = ResNet50_Weights.DEFAULT if pretrained else None
            model = resnet50(weights=weights)
            self.out_dim = model.fc.in_features
            model.fc = nn.Identity()
            self.model = model
        elif name in {"vit", "vit_b_16"}:
            from torchvision.models import vit_b_16, ViT_B_16_Weights

            weights = ViT_B_16_Weights.DEFAULT if pretrained else None
            model = vit_b_16(weights=weights)
            self.out_dim = model.heads.head.in_features
            model.heads.head = nn.Identity()
            self.model = model
        else:
            raise ValueError(f"Unknown image backbone: {backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SensorEncoder(nn.Module):
    """Encode 1D sensor sequences using a small CNN or Transformer."""

    def __init__(
        self,
        input_channels: int,
        backbone: str = "cnn",
        hidden: int = 64,
    ):
        super().__init__()
        name = backbone.lower()
        if name == "cnn":
            self.model = nn.Sequential(
                nn.Conv1d(input_channels, hidden, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.out_dim = hidden
        elif name == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=input_channels, nhead=4
            )
            self.model = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.out_dim = input_channels
        else:
            raise ValueError(f"Unknown sensor backbone: {backbone}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.model, nn.Sequential):
            x = self.model(x)  # [B, hidden, 1]
            return x.squeeze(-1)
        # Transformer expects [L, B, C]
        x = x.permute(2, 0, 1)
        x = self.model(x)  # [L, B, C]
        return x.mean(0)


class TabularEncoder(nn.Module):
    """Simple MLP for tabular features."""

    def __init__(self, input_dim: int, hidden: int = 128, depth: int = 2):
        super().__init__()
        layers = []
        d = input_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers.append(nn.Linear(d, hidden))
        self.model = nn.Sequential(*layers)
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class MultiModalSurvivalModel(nn.Module):
    """Fuse multiple modalities and predict hazards."""

    def __init__(
        self,
        num_bins: int,
        *,
        image_backbone: str = "resnet50",
        sensor_backbone: str = "cnn",
        sensor_channels: int = 1,
        tabular_dim: int = 0,
        fusion_hidden: int = 256,
    ):
        super().__init__()

        self.image_encoder = ImageEncoder(image_backbone) if image_backbone else None
        self.sensor_encoder = (
            SensorEncoder(sensor_channels, sensor_backbone)
            if sensor_backbone
            else None
        )
        self.tabular_encoder = (
            TabularEncoder(tabular_dim) if tabular_dim > 0 else None
        )

        fused_dim = 0
        for enc in (self.image_encoder, self.sensor_encoder, self.tabular_encoder):
            if enc is not None:
                fused_dim += enc.out_dim

        self.fusion = nn.Sequential(nn.Linear(fused_dim, fusion_hidden), nn.ReLU())
        self.hazard_head = MultiTaskModel(input_dim=fusion_hidden, num_bins=num_bins)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        feats = []
        if self.image_encoder is not None and "image" in x:
            feats.append(self.image_encoder(x["image"]))
        if self.sensor_encoder is not None and "sensor" in x:
            feats.append(self.sensor_encoder(x["sensor"]))
        if self.tabular_encoder is not None and "tabular" in x:
            feats.append(self.tabular_encoder(x["tabular"]))

        if not feats:
            raise ValueError("No modalities provided for fusion.")

        fused = torch.cat(feats, dim=1)
        fused = self.fusion(fused)
        return self.hazard_head(fused)

    # ------------------------------------------------------------------
    def load_pretrained_backbones(
        self,
        *,
        image_state_dict: Optional[str] = None,
        sensor_state_dict: Optional[str] = None,
        strict: bool = False,
    ) -> None:
        """Load weights for backbone encoders.

        Parameters
        ----------
        image_state_dict: str, optional
            Path to a state dict for :class:`ImageEncoder`.
        sensor_state_dict: str, optional
            Path to a state dict for :class:`SensorEncoder`.
        strict: bool
            Whether to enforce that the keys match exactly.
        """

        if image_state_dict and self.image_encoder is not None:
            state = torch.load(image_state_dict, map_location="cpu")
            self.image_encoder.load_state_dict(state, strict=strict)

        if sensor_state_dict and self.sensor_encoder is not None:
            state = torch.load(sensor_state_dict, map_location="cpu")
            self.sensor_encoder.load_state_dict(state, strict=strict)


__all__ = ["MultiModalSurvivalModel"]


import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "MultiTaskModel",
    "TabularEncoder",
    "SensorEncoder",
    "ImageEncoder",
    "MultiModalModel",
    "load_model",
]

class MultiTaskModel(nn.Module):
    """
    Shared MLP trunk -> per-time-bin hazards via linear head over shared features.
    Returns hazards in shape [B, T].
    """
    def __init__(self, input_dim: int, num_bins: int, hidden: int = 256, depth: int = 2, dropout: float = 0.2):
        super().__init__()
        self.config = {
            "input_dim": input_dim,
            "num_bins": num_bins,
            "hidden": hidden,
            "depth": depth,
            "dropout": dropout,
        }
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

    def forward(self, x):
        h = self.trunk(x)
        logits = self.hazard_layer(h)
        return torch.sigmoid(logits)


class TabularEncoder(nn.Module):
    """Simple MLP encoder for tabular features."""

    def __init__(self, input_dim: int, hidden: int = 128, depth: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        d = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden
        self.net = nn.Sequential(*layers)
        self.output_dim = hidden

    def forward(self, x):
        return self.net(x)


class SensorEncoder(nn.Module):
    """1D convolutional encoder for sensor time-series."""

    def __init__(self, in_channels: int, hidden: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_dim = hidden

    def forward(self, x):  # x: [B, C, L]
        h = self.conv(x)
        h = self.pool(h)
        return h.squeeze(-1)


class ImageEncoder(nn.Module):
    """Small CNN encoder for images."""

    def __init__(self, in_channels: int, hidden: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(32, hidden)
        self.output_dim = hidden

    def forward(self, x):  # x: [B, C, H, W]
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)


class MultiModalModel(nn.Module):
    """Fuses tabular, sensor, and image modalities into hazard predictions."""

    def __init__(
        self,
        tabular_dim: int,
        sensor_channels: int,
        image_channels: int,
        num_bins: int,
        hidden: int = 128,
    ):
        super().__init__()
        self.tabular = TabularEncoder(tabular_dim, hidden)
        self.sensor = SensorEncoder(sensor_channels, hidden)
        self.image = ImageEncoder(image_channels, hidden)
        fusion_dim = self.tabular.output_dim + self.sensor.output_dim + self.image.output_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_bins),
        )
        self.num_bins = num_bins
        self.config = {
            "tabular_dim": tabular_dim,
            "sensor_channels": sensor_channels,
            "image_channels": image_channels,
            "num_bins": num_bins,
            "hidden": hidden,
        }
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, inputs: dict):
        t = self.tabular(inputs["tabular"])
        s = self.sensor(inputs["sensor"])
        i = self.image(inputs["image"])
        h = torch.cat([t, s, i], dim=1)
        logits = self.head(h)
        return torch.sigmoid(logits)


def load_model(path: str, map_location=None):
    """Load model (MultiTaskModel or MultiModalModel) from disk."""

    payload = torch.load(path, map_location=map_location)
    if isinstance(payload, dict) and "state_dict" in payload:
        cls = payload.get("class")
        config = payload.get("config", {})
        state = payload["state_dict"]
    else:  # backwards compatibility: plain state_dict
        cls = "MultiTaskModel"
        config = payload.get("config", {}) if isinstance(payload, dict) else {}
        state = payload if isinstance(payload, dict) else payload

    if cls == "MultiTaskModel":
        model = MultiTaskModel(**config)
    elif cls == "MultiModalModel":
        model = MultiModalModel(**config)
    else:
        raise ValueError(f"Unknown model class: {cls}")
    model.load_state_dict(state)
    return model

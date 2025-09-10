import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from multimodal_dataset import MultiModalDataset
from models.multimodal_survival import MultiModalSurvivalModel
from trainer import Trainer

# ---- 1. 生成合成数据 ----
N, T = 128, 5  # 样本数与离散时间区间
modalities = {
    "image": np.random.randn(N, 3, 64, 64).astype("float32"),
    "sensor": np.random.randn(N, 100, 6).astype("float32"),
    "tabular": np.random.randn(N, 20).astype("float32"),
}
labels = np.zeros((N, T), dtype="float32")        # one-hot 事件标签 (示例)
masks  = np.tril(np.ones((N, T), dtype="float32"))  # 风险集掩码

dataset = MultiModalDataset(modalities, labels, masks, train=True)
loader  = DataLoader(dataset, batch_size=16, shuffle=True)

# ---- 2. 定义各模态编码器 ----
encoders = {
    "image": nn.Sequential(nn.Flatten(), nn.Linear(3 * 64 * 64, 64), nn.ReLU()),
    "sensor": nn.Sequential(nn.Flatten(), nn.Linear(100 * 6, 64), nn.ReLU()),
    "tabular": nn.Sequential(nn.Linear(20, 64), nn.ReLU()),
}

model = MultiModalSurvivalModel(encoders, hidden_dim=64, num_bins=T)

# ---- 3. 训练 ----
trainer = Trainer(model, lr=1e-3)
dummy_duration = np.ones(N)    # 示例：真实数据需替换为生存时间
dummy_event    = np.ones(N)    # 示例：真实数据需替换为事件指示
trainer.fit(loader, loader, dummy_duration, dummy_event, epochs=5)

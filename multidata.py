import numpy as np
import os

# ---- 1. 定义数据参数 ----
N, T = 128, 5  # 样本数与离散时间区间
output_dir = "synthetic_survival_data"

print("开始生成合成数据...")

# ---- 2. 生成多模态特征数据 ----
# 创建一个字典来存储不同模态的数据
modalities = {
    # 图像数据: N个样本, 3个通道 (RGB), 64x64 像素
    "image": np.random.randn(N, 3, 64, 64).astype("float32"),
    
    # 传感器/时间序列数据: N个样本, 100个时间步, 6个特征
    "sensor": np.random.randn(N, 100, 6).astype("float32"),
    
    # 表格数据: N个样本, 20个特征
    "tabular": np.random.randn(N, 20).astype("float32"),
}
print(f"已生成 {len(modalities)} 种模态的数据。")

# ---- 3. 生成生存分析标签 ----
# 随机生成每个样本的持续时间 (1 到 T)
durations = np.random.randint(1, T + 1, size=N)

# 随机生成事件是否发生 (70% 的概率发生事件)
events = np.random.binomial(1, 0.7, size=N)

# 根据时间和事件，创建适用于离散时间模型的标签 (labels) 和掩码 (masks)
labels = np.zeros((N, T), dtype="float32")
masks  = np.zeros((N, T), dtype="float32")

for i in range(N):
    # 将持续时间转换为从0开始的索引
    time_bin_index = durations[i] - 1
    
    # 创建风险集掩码：在事件发生时间点之前（包括该时间点），样本都处于风险中
    masks[i, :time_bin_index + 1] = 1
    
    # 如果事件实际发生了，并且在我们的观察窗口T内
    if events[i] == 1 and time_bin_index < T:
        # 在对应的时间区间将标签置为1
        labels[i, time_bin_index] = 1

print("已生成生存时间和事件标签。")

# ---- 4. 保存所有数据到本地文件夹 ----
# 创建目标文件夹，如果已存在则不报错
os.makedirs(output_dir, exist_ok=True)

print(f"\n正在将数据保存到 '{output_dir}' 文件夹...")

# 循环保存每个模态的数据
for name, data in modalities.items():
    file_path = os.path.join(output_dir, f"{name}.npy")
    np.save(file_path, data)
    print(f"- 已保存 {name} 数据, 维度: {data.shape}")

# 保存原始的持续时间和事件数据
np.save(os.path.join(output_dir, "durations.npy"), durations)
print(f"- 已保存 durations 数据, 维度: {durations.shape}")
np.save(os.path.join(output_dir, "events.npy"), events)
print(f"- 已保存 events 数据, 维度: {events.shape}")

# 保存为模型训练准备好的标签和掩码
np.save(os.path.join(output_dir, "labels.npy"), labels)
print(f"- 已保存 labels 数据, 维度: {labels.shape}")
np.save(os.path.join(output_dir, "masks.npy"), masks)
print(f"- 已保存 masks 数据, 维度: {masks.shape}")

print("\n所有数据均已成功保存！")

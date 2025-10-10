# image_pipeline.py
# ------------------------------------------------------------
# Build a tabular dataset for survival analysis from images.
# Input: a manifest CSV (columns: image, duration, event, [optional extra numeric cols])
#        and an image root directory (unzipped folder).
# Output: pandas.DataFrame with columns:
#         ['duration', 'event', 'img_feat_0000', 'img_feat_0001', ...] (+ any extra numeric cols)
# ------------------------------------------------------------
from __future__ import annotations
import os, io, math, zipfile, tempfile
from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

# 尝试导入 torchvision（若未安装会给出清晰错误提示）
try:
    import torchvision
    from torchvision.models import resnet50, ResNet50_Weights
    from torchvision.transforms import v2 as T
except Exception as e:
    raise RuntimeError(
        "torchvision 未安装或版本不兼容。请先安装：\n"
        "  pip install --upgrade torchvision\n"
        "并确保 PyTorch 版本与其匹配。"
    ) from e


def _resolve_device(device: Optional[str] = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_backbone_and_transform(
    model_name: str = "resnet50",
    device: Optional[str] = None
):
    """
    目前提供 ResNet-50（ImageNet 预训练）。后续可以扩展 ViT/timm。
    返回: (feature_extractor_model, transform, feature_dim)
    """
    dev = _resolve_device(device)
    name = (model_name or "resnet50").lower().strip()

    if name != "resnet50":
        raise ValueError("当前版本仅支持 model_name='resnet50'。等你确认可运行后，我再帮你扩展 ViT。")

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()  # 去掉分类头，输出 2048 维嵌入
    model.eval().to(dev)

    # 官方推荐的 transforms（与权重配套）
    tfm = weights.transforms()

    feat_dim = 2048
    return model, tfm, feat_dim, dev


class _ImageTableDataset(Dataset):
    """
    从 manifest 逐行读取图像。允许 image 列给出相对路径（相对 image_root）或者绝对路径。
    缺失/损坏图像会跳过（记录 warning），保持与 manifest 的对齐映射（keep_idx）。
    """
    def __init__(self,
                 manifest: pd.DataFrame,
                 image_root: str,
                 image_col: str = "image",
                 transform=None):
        assert image_col in manifest.columns, f"manifest 缺少列: '{image_col}'"
        self.manifest = manifest.reset_index(drop=False).rename(columns={"index": "_orig_idx"})
        self.image_root = image_root
        self.image_col = image_col
        self.transform = transform

        self.samples: List[Tuple[int, str]] = []  # (orig_row_idx, resolved_path)
        self._build_samples()

    def _build_samples(self):
        miss = 0
        for _, row in self.manifest.iterrows():
            p = str(row[self.image_col]).strip()
            if not p:
                miss += 1
                continue
            # 绝对路径或相对路径均可
            abspath = p if os.path.isabs(p) else os.path.join(self.image_root, p)
            if os.path.exists(abspath):
                self.samples.append((int(row["_orig_idx"]), abspath))
            else:
                miss += 1
        if miss > 0:
            print(f"[image_pipeline] 跳过 {miss} 张缺失/无效图像（路径不存在）。有效图像数 = {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i: int):
        orig_idx, path = self.samples[i]
        with Image.open(path) as im:
            im = im.convert("RGB")
        if self.transform is not None:
            im = self.transform(im)
        return im, orig_idx


@torch.no_grad()
def _extract_features(
    model,
    loader: DataLoader,
    device: str,
    feat_dim: int
) -> Tuple[np.ndarray, List[int]]:
    feats = np.zeros((len(loader.dataset), feat_dim), dtype=np.float32)
    orig_indices: List[int] = []
    k = 0
    for xb, idxb in loader:
        xb = xb.to(device, non_blocking=True)
        fb = model(xb)                   # [B, 2048]
        fb = fb.detach().cpu().numpy()
        bs = fb.shape[0]
        feats[k:k+bs, :] = fb
        orig_indices.extend([int(v) for v in idxb])
        k += bs
    return feats, orig_indices


def images_to_dataframe(
    manifest_df: pd.DataFrame,
    image_root: str,
    *,
    model_name: str = "resnet50",
    image_col: str = "image",
    duration_col: str = "duration",
    event_col: str = "event",
    id_col: Optional[str] = "id",
    batch_size: int = 64,
    num_workers: int = 2,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """
    将图像批量编码为嵌入，并返回适配 MySA 的 DataFrame。
    - manifest_df: 至少包含 [image, duration, event]
      额外的数值列会一并保留（如年龄等临床变量）。
    - image_root: 图像所在根目录（如果 CSV 中是绝对路径，可设为 ""）
    """
    # 基础检查
    for col in (image_col, duration_col, event_col):
        if col not in manifest_df.columns:
            raise KeyError(f"manifest 缺少列: '{col}'. 请在 CSV 中包含: image, duration, event")

    # 加载骨干网络与预处理
    model, tfm, feat_dim, dev = _load_backbone_and_transform(model_name, device)

    # 构建数据集与 DataLoader
    ds = _ImageTableDataset(manifest_df, image_root=image_root, image_col=image_col, transform=tfm)
    if len(ds) == 0:
        raise RuntimeError("没有可用图像。请检查 CSV 路径与 ZIP 解压目录。")

    loader = DataLoader(
        ds, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers),
        pin_memory=(dev == "cuda"), persistent_workers=(int(num_workers) > 0)
    )

    # 提取特征
    feats, keep_orig_idx = _extract_features(model, loader, dev, feat_dim=feat_dim)

    # 对齐回 manifest 行
    kept = manifest_df.reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    kept = kept[kept["_orig_idx"].isin(keep_orig_idx)].copy()
    kept = kept.set_index("_orig_idx").loc[keep_orig_idx].reset_index()

    # 组装 DataFrame
    # 1) 先处理核心列：duration/event
    df = pd.DataFrame({
        "duration": pd.to_numeric(kept[duration_col], errors="coerce").astype(np.float32),
        "event":    pd.to_numeric(kept[event_col], errors="coerce").fillna(0).astype(np.int32).clip(0, 1),
    })

    # 2) 加入图像特征列
    feat_cols = [f"img_feat_{i:04d}" for i in range(feat_dim)]
    feat_df = pd.DataFrame(feats, columns=feat_cols)
    df = pd.concat([df, feat_df], axis=1)

    # 3) 若提供了 ID 列，则保留在最前面方便与其他模态对齐
    if id_col and id_col in kept.columns:
        df.insert(0, id_col, kept[id_col].to_numpy())

    # 4) 保留 manifest 中的额外数值特征
    extra_cols = [c for c in kept.columns if c not in {image_col, duration_col, event_col, "_orig_idx"}]
    for c in extra_cols:
        # 仅保留可数值化的列（非数值会被跳过或变成 NaN→填 0）
        if pd.api.types.is_numeric_dtype(kept[c]):
            df[c] = kept[c].astype(np.float32)
        else:
            try:
                df[c] = pd.to_numeric(kept[c], errors="coerce").astype(np.float32).fillna(0.0)
            except Exception:
                # 非数值列就不并入（避免破坏现有流水线）
                pass

    return df


def unzip_images_to_temp(uploaded_zip_file) -> str:
    """
    辅助函数：把用户上传的 ZIP（Streamlit UploadedFile 或字节流）解压到一个临时目录。
    返回临时目录路径。
    """
    tmp_dir = tempfile.mkdtemp(prefix="sa_imgzip_")
    with zipfile.ZipFile(uploaded_zip_file) as zf:
        zf.extractall(tmp_dir)
    return tmp_dir

def scan_images_recursively(root: str, exts={'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}):
    """递归扫描 root 下的所有图片，返回相对路径列表（不让用户接触绝对路径）。"""
    files = []
    for d, _, fns in os.walk(root):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() in exts:
                abspath = os.path.join(d, fn)
                rel = os.path.relpath(abspath, root)  # 相对 root
                files.append(rel)
    files.sort()
    return files

def build_manifest_from_zip(uploaded_zip_file, default_event: int = 0):
    """
    解压上传的 ZIP，自动生成一个待标注的 manifest DataFrame：
    columns: image(相对路径), duration(空), event(默认0)。
    返回 (manifest_df, image_root)。
    """
    root = unzip_images_to_temp(uploaded_zip_file)
    files = scan_images_recursively(root)
    if not files:
        raise RuntimeError("ZIP 里没有找到图片（支持png/jpg/jpeg/bmp/tif/tiff）。")
    df = pd.DataFrame({
        "image": files,
        "duration": np.nan,       # 让用户在界面里填写或导入 CSV
        "event": default_event,   # 0/1，默认0
    })
    return df, root

def manifest_template_csv_bytes(manifest_df: pd.DataFrame) -> bytes:
    """
    生成一份 CSV 模板（二进制bytes）供“下载模板”按钮使用。
    """
    cols = ['image', 'duration', 'event'] + [c for c in manifest_df.columns
                                             if c not in ('image','duration','event')]
    buf = io.StringIO()
    manifest_df[cols].to_csv(buf, index=False)
    return buf.getvalue().encode('utf-8')
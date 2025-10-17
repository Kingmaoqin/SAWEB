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

# Attempt to import torchvision (provide a clear error if unavailable)
try:
    import torchvision
    from torchvision.models import (
        resnet18,
        resnet34,
        resnet50,
        ResNet18_Weights,
        ResNet34_Weights,
        ResNet50_Weights,
    )
    from torchvision.transforms import v2 as T
except Exception as e:
    raise RuntimeError(
        "torchvision is either missing or incompatible. Install it via:\n"
        "  pip install --upgrade torchvision\n"
        "and ensure the version matches your PyTorch build."
    ) from e

from utils.identifiers import canonicalize_series


def _resolve_device(device: Optional[str] = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_backbone_and_transform(
    model_name: str = "resnet50",
    device: Optional[str] = None
):
    """Load the feature extractor backbone, returning the model, transform, feature dimension."""
    dev = _resolve_device(device)
    name = (model_name or "resnet50").lower().strip()

    if name in {"resnet18", "resnet34", "resnet50"}:
        backbone_map = {
            "resnet18": (resnet18, ResNet18_Weights),
            "resnet34": (resnet34, ResNet34_Weights),
            "resnet50": (resnet50, ResNet50_Weights),
        }
        backbone_fn, weights_cls = backbone_map[name]
        weights = weights_cls.DEFAULT
        model = backbone_fn(weights=weights)
        feat_dim = model.fc.in_features
        model.fc = torch.nn.Identity()  # expose penultimate features
        model.eval().to(dev)
        tfm = weights.transforms()
        return model, tfm, feat_dim, dev

    raise ValueError("Only ResNet backbones (resnet18/resnet34/resnet50) are supported in this version.")


class _ImageTableDataset(Dataset):
    """Dataset that resolves image paths listed in the manifest."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        image_root: str,
        image_col: str = "image",
        transform=None,
    ):
        assert image_col in manifest.columns, f"Manifest is missing column '{image_col}'"
        self.manifest = manifest.reset_index(drop=False).rename(columns={"index": "_orig_idx"})
        self.image_root = image_root
        self.image_col = image_col
        self.transform = transform

        self.samples: List[Tuple[int, str]] = []  # (orig_row_idx, resolved_path)
        self.missing_paths: List[str] = []
        self.ambiguous_paths: List[str] = []
        self._build_samples()

    def _build_samples(self):
        path_lookup = {}
        path_lookup_lower = {}
        basename_lookup = {}
        basename_lookup_lower = {}
        if os.path.isdir(self.image_root):
            for d, _, fns in os.walk(self.image_root):
                for fn in fns:
                    abspath = os.path.join(d, fn)
                    rel = os.path.relpath(abspath, self.image_root).replace("\\", "/")
                    rel_stripped = rel.lstrip("./")
                    lower = rel.lower()
                    lower_stripped = rel_stripped.lower()
                    path_lookup[rel] = abspath
                    path_lookup[rel_stripped] = abspath
                    path_lookup_lower[lower] = abspath
                    path_lookup_lower[lower_stripped] = abspath
                    basename_lookup.setdefault(fn, []).append(abspath)
                    basename_lookup_lower.setdefault(fn.lower(), []).append(abspath)
        miss = 0
        for _, row in self.manifest.iterrows():
            p = str(row[self.image_col]).strip()
            if not p:
                miss += 1
                continue
            rel_norm = p.replace("\\", "/")
            rel_norm_stripped = rel_norm.lstrip("./")
            abspath = None
            if os.path.isabs(rel_norm) and os.path.exists(rel_norm):
                abspath = rel_norm
            else:
                abspath = (
                    path_lookup.get(rel_norm)
                    or path_lookup.get(rel_norm_stripped)
                    or path_lookup_lower.get(rel_norm.lower())
                    or path_lookup_lower.get(rel_norm_stripped.lower())
                )
                if abspath is None:
                    base = os.path.basename(rel_norm)
                    candidates = basename_lookup.get(base, [])
                    if not candidates:
                        candidates = basename_lookup_lower.get(base.lower(), [])
                    if len(candidates) == 1:
                        abspath = candidates[0]
                    elif len(candidates) > 1:
                        self.ambiguous_paths.append(p)
                        miss += 1
                        continue
            if abspath and os.path.exists(abspath):
                self.samples.append((int(row["_orig_idx"]), abspath))
            else:
                self.missing_paths.append(p)
                miss += 1
        if miss > 0:
            print(
                "[image_pipeline] Skipped "
                f"{miss} missing/invalid image entries. Resolved images: {len(self.samples)}"
            )

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
    """Encode images into embeddings and assemble a DataFrame compatible with MySA."""
    # Input validation
    for col in (image_col, duration_col, event_col):
        if col not in manifest_df.columns:
            raise KeyError(f"Manifest is missing column '{col}'. Expected columns: image, duration, event.")

    # Load backbone and preprocessing pipeline
    model, tfm, feat_dim, dev = _load_backbone_and_transform(model_name, device)

    # Build dataset and DataLoader
    ds = _ImageTableDataset(manifest_df, image_root=image_root, image_col=image_col, transform=tfm)
    if len(ds) == 0:
        detail = []
        if ds.missing_paths:
            unique_missing = sorted(set(ds.missing_paths))
            preview = ", ".join(unique_missing[:10])
            if len(unique_missing) > 10:
                preview += "…"
            detail.append(f"Missing files: {preview}")
        if ds.ambiguous_paths:
            unique_ambiguous = sorted(set(ds.ambiguous_paths))
            preview = ", ".join(unique_ambiguous[:10])
            if len(unique_ambiguous) > 10:
                preview += "…"
            detail.append(f"Ambiguous matches (duplicate filenames): {preview}")
        hint = "; ".join(detail)
        raise RuntimeError(
            "No images could be matched against the manifest. Ensure the CSV paths align with the extracted ZIP contents."
            + (f" {hint}" if hint else "")
        )

    loader = DataLoader(
        ds, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers),
        pin_memory=(dev == "cuda"), persistent_workers=(int(num_workers) > 0)
    )

    # Extract features
    feats, keep_orig_idx = _extract_features(model, loader, dev, feat_dim=feat_dim)

    base = manifest_df.reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    base["_orig_idx"] = base["_orig_idx"].astype(int)
    base = base.set_index("_orig_idx")

    # Assemble the output DataFrame aligned to the original manifest order.
    core = pd.DataFrame(index=base.index)
    core["duration"] = pd.to_numeric(base[duration_col], errors="coerce").astype(np.float32)
    core["event"] = (
        pd.to_numeric(base[event_col], errors="coerce")
        .fillna(0)
        .astype(np.int32)
        .clip(0, 1)
    )

    feat_cols = [f"img_feat_{i:04d}" for i in range(feat_dim)]
    feat_df = pd.DataFrame(np.nan, index=base.index, columns=feat_cols, dtype=np.float32)
    matched_rows = len(set(keep_orig_idx))
    if keep_orig_idx:
        matched_df = pd.DataFrame(
            feats.astype(np.float32),
            index=pd.Index(keep_orig_idx),
            columns=feat_cols,
        )
        feat_df.update(matched_df)
    unmatched = len(base) - matched_rows
    if unmatched > 0:
        print(
            f"[image_pipeline] Image embeddings resolved for {matched_rows}/{len(base)} rows. "
            f"{unmatched} rows will retain NaNs until assets are provided."
        )
    df = pd.concat([core, feat_df], axis=1)

    # Preserve identifier column first for downstream joins.
    if id_col and id_col in base.columns:
        ids = canonicalize_series(base[id_col])
        df.insert(0, id_col, ids)

    # Retain additional numeric columns from the manifest (aligned to manifest order).
    extra_cols = [c for c in base.columns if c not in {image_col, duration_col, event_col}]
    for c in extra_cols:
        if c == id_col:
            continue
        series = base[c]
        if pd.api.types.is_numeric_dtype(series):
            df[c] = series.astype(np.float32)
        else:
            try:
                df[c] = pd.to_numeric(series, errors="coerce").astype(np.float32)
            except Exception:
                continue

    # Ensure one row per identifier (if duplicates exist take mean of embeddings).
    if id_col and id_col in df.columns and df[id_col].duplicated().any():
        agg_map = {"duration": "first", "event": "first"}
        for c in feat_cols:
            agg_map[c] = "mean"
        for c in extra_cols:
            if c in df.columns and c not in {id_col, "duration", "event"}:
                agg_map.setdefault(c, "mean")
        df = df.groupby(id_col, as_index=False).agg(agg_map)

    df = df.reset_index(drop=True)
    return df


def unzip_images_to_temp(uploaded_zip_file) -> str:
    """Extract an uploaded ZIP (Streamlit UploadedFile or bytes) to a temporary directory."""
    tmp_dir = tempfile.mkdtemp(prefix="sa_imgzip_")
    with zipfile.ZipFile(uploaded_zip_file) as zf:
        zf.extractall(tmp_dir)
    return tmp_dir

def scan_images_recursively(root: str, exts={'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}):
    """Recursively scan ``root`` for images and return relative paths."""
    files = []
    for d, _, fns in os.walk(root):
        for fn in fns:
            if os.path.splitext(fn)[1].lower() in exts:
                abspath = os.path.join(d, fn)
                rel = os.path.relpath(abspath, root)
                files.append(rel)
    files.sort()
    return files

def build_manifest_from_zip(uploaded_zip_file, default_event: int = 0):
    """Extract the ZIP and build a manifest skeleton (image, duration, event)."""
    root = unzip_images_to_temp(uploaded_zip_file)
    files = scan_images_recursively(root)
    if not files:
        raise RuntimeError("No image files were found in the ZIP (supported: png/jpg/jpeg/bmp/tif/tiff).")
    df = pd.DataFrame({
        "image": files,
        "duration": np.nan,
        "event": default_event,
    })
    return df, root

def manifest_template_csv_bytes(manifest_df: pd.DataFrame) -> bytes:
    """Create a CSV template (bytes) for the "Download template" button."""
    cols = ['image', 'duration', 'event'] + [c for c in manifest_df.columns
                                             if c not in ('image','duration','event')]
    buf = io.StringIO()
    manifest_df[cols].to_csv(buf, index=False)
    return buf.getvalue().encode('utf-8')
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
    from torchvision.models import resnet50, ResNet50_Weights
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

    if name != "resnet50":
        raise ValueError("Only model_name='resnet50' is supported in this version.")

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.fc = torch.nn.Identity()  # remove the classification head to expose 2048-dim embeddings
    model.eval().to(dev)

    # Use the transforms that ship with the selected weights
    tfm = weights.transforms()

    feat_dim = 2048
    return model, tfm, feat_dim, dev


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
        basename_lookup = {}
        if os.path.isdir(self.image_root):
            for d, _, fns in os.walk(self.image_root):
                for fn in fns:
                    abspath = os.path.join(d, fn)
                    rel = os.path.relpath(abspath, self.image_root).replace("\\", "/")
                    path_lookup[rel] = abspath
                    path_lookup[rel.lstrip("./")] = abspath
                    basename_lookup.setdefault(fn, []).append(abspath)
        miss = 0
        for _, row in self.manifest.iterrows():
            p = str(row[self.image_col]).strip()
            if not p:
                miss += 1
                continue
            rel_norm = p.replace("\\", "/")
            abspath = None
            if os.path.isabs(rel_norm) and os.path.exists(rel_norm):
                abspath = rel_norm
            else:
                abspath = path_lookup.get(rel_norm) or path_lookup.get(rel_norm.lstrip("/"))
                if abspath is None:
                    candidates = basename_lookup.get(os.path.basename(rel_norm), [])
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

    # Align features back to the manifest rows
    kept = manifest_df.reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    kept = kept[kept["_orig_idx"].isin(keep_orig_idx)].copy()
    kept = kept.set_index("_orig_idx").loc[keep_orig_idx].reset_index()

    # Assemble the output DataFrame
    # 1) Core duration/event columns
    df = pd.DataFrame({
        "duration": pd.to_numeric(kept[duration_col], errors="coerce").astype(np.float32),
        "event":    pd.to_numeric(kept[event_col], errors="coerce").fillna(0).astype(np.int32).clip(0, 1),
    })

    # 2) Append image embedding columns
    feat_cols = [f"img_feat_{i:04d}" for i in range(feat_dim)]
    feat_df = pd.DataFrame(feats, columns=feat_cols)
    df = pd.concat([df, feat_df], axis=1)

    # 3) Preserve the ID column first when available for cross-modality joins
    if id_col and id_col in kept.columns:
        ids = canonicalize_series(kept[id_col])
        df.insert(0, id_col, ids)

    # 4) Retain additional numeric columns from the manifest
    extra_cols = [c for c in kept.columns if c not in {image_col, duration_col, event_col, "_orig_idx"}]
    for c in extra_cols:
        # Only keep columns that can be coerced to numeric values
        if pd.api.types.is_numeric_dtype(kept[c]):
            df[c] = kept[c].astype(np.float32)
        else:
            try:
                df[c] = pd.to_numeric(kept[c], errors="coerce").astype(np.float32).fillna(0.0)
            except Exception:
                # Skip non-numeric columns to avoid breaking downstream code
                pass

    # 5) Ensure one row per identifier so downstream alignment can reindex safely
    if id_col and id_col in df.columns:
        if df[id_col].duplicated().any():
            # Keep the first duration/event entry (should be identical across duplicates)
            agg_map = {}
            if "duration" in df.columns:
                agg_map["duration"] = "first"
            if "event" in df.columns:
                agg_map["event"] = "first"
            feat_cols = [c for c in df.columns if c not in {id_col, "duration", "event"}]
            agg_map.update({c: "mean" for c in feat_cols})
            df = df.groupby(id_col, as_index=False).agg(agg_map)

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
"""Identifier normalization helpers for multimodal alignment."""
from __future__ import annotations

import math
import os
import re
from typing import Any, Optional

import numpy as np
import pandas as pd


def canonicalize_identifier(value: Any) -> Optional[str]:
    """Return a canonical string identifier or ``None`` when missing."""
    if value is None:
        return None
    # Respect pandas NA / numpy nan sentinels
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, (np.integer, int)):
        return str(int(value))

    if isinstance(value, (np.floating, float)):
        if not math.isfinite(float(value)):
            return None
        fv = float(value)
        if fv.is_integer():
            return str(int(fv))
        return format(fv, "g")

    text = str(value).strip()
    if not text:
        return None

    text = _strip_asset_like_tokens(text)
    lowered = text.lower()
    if "." in text or "e" in lowered:
        try:
            fv = float(text)
        except ValueError:
            return text
        if not math.isfinite(fv):
            return None
        if fv.is_integer():
            return str(int(fv))
        return format(fv, "g")

    if _NUMERIC_STRING_PATTERN.fullmatch(text):
        try:
            return str(int(text))
        except ValueError:
            return text

    return text


_NUMERIC_STRING_PATTERN = re.compile(r"[+-]?[0-9]+")
_ASSET_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".gif",
    ".csv",
    ".tsv",
    ".txt",
    ".json",
    ".npy",
    ".npz",
}


def _strip_asset_like_tokens(text: str) -> str:
    """Normalize identifiers that resemble asset paths.

    When raw manifests are keyed by a path (e.g. ``images/PT_0001.png``), the
    multimodal merge should align on the underlying identifier rather than the
    filename.  This helper removes directory prefixes and well-known asset
    extensions while leaving non-path identifiers untouched.
    """

    if not text:
        return text

    normalized = text.replace("\\", "/")
    if "/" in normalized:
        normalized = normalized.split("/")[-1]

    root, ext = os.path.splitext(normalized)
    if ext.lower() in _ASSET_EXTENSIONS and root:
        return root
    return normalized


def canonicalize_series(series: pd.Series) -> pd.Series:
    """Apply :func:`canonicalize_identifier` element-wise to a series."""
    values = [canonicalize_identifier(v) for v in series]
    return pd.Series(values, index=series.index, dtype="string")

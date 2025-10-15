"""Identifier normalization helpers for multimodal alignment."""
from __future__ import annotations

import math
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

    return text


def canonicalize_series(series: pd.Series) -> pd.Series:
    """Apply :func:`canonicalize_identifier` element-wise to a series."""
    values = [canonicalize_identifier(v) for v in series]
    return pd.Series(values, index=series.index, dtype="string")

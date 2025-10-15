"""Diagnose identifier alignment issues in multimodal manifests.

This utility inspects the tabular, image and sensor artefacts produced for the
multimodal training pipeline.  It mirrors the identifier canonicalisation logic
used by the product so you can quickly spot why a modality fails to align (for
example when every merged image column becomes empty).

Example
-------

.. code-block:: bash

    python analysis/multimodal_alignment_debugger.py \
        --root /home/user/SAWEB1013/synthetic_multimodal \
        --tabular tabular.csv \
        --image-manifest image_manifest.csv \
        --sensor-manifest sensor_manifest.csv

The script prints a textual report and writes optional CSV summaries that list
every identifier together with the resolved asset paths.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.identifiers import canonicalize_identifier, canonicalize_series  # noqa: E402


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected CSV file at {path!s}, but it does not exist.")
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - depends on external files
        raise RuntimeError(f"Failed to load CSV file {path!s}: {exc}") from exc


def _group_raw_ids(df: pd.DataFrame, id_col: str, canonical_col: str) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for canon, rows in df.groupby(canonical_col):
        raw_values = [str(v) for v in rows[id_col].dropna().unique()]
        mapping[str(canon)] = raw_values
    return mapping


def _as_optional_list(values: Iterable[Optional[str]]) -> List[Optional[str]]:
    return [v if v else None for v in values]


@dataclass
class SensorSpec:
    columns: Sequence[str]
    target_len: int

    @property
    def channels(self) -> int:
        return len(self.columns)


def _resolve_asset_paths(
    manifest: pd.DataFrame,
    *,
    id_col: str,
    path_col: str,
    root: Path,
) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    norm_root = root.resolve()
    for _, row in manifest.iterrows():
        key = canonicalize_identifier(row[id_col])
        if not key:
            continue
        raw_path = str(row[path_col]).strip()
        if not raw_path:
            continue
        if os.path.isabs(raw_path):
            abs_path = Path(raw_path)
        else:
            abs_path = (norm_root / raw_path).resolve()
        if abs_path.exists():
            lookup[key] = str(abs_path)
    return lookup


def _find_sensor_columns(path: Path) -> Tuple[List[str], int]:
    df = pd.read_csv(path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    length = int(len(df))
    return numeric_cols, length


def build_image_paths_for_ids(
    ids: Sequence[Optional[str]],
    *,
    manifest: pd.DataFrame,
    id_col: str,
    path_col: str,
    root: Path | str,
) -> List[Optional[str]]:
    lookup = _resolve_asset_paths(manifest, id_col=id_col, path_col=path_col, root=Path(root))
    results: List[Optional[str]] = []
    for i in ids:
        key = canonicalize_identifier(i)
        results.append(lookup.get(key) if key else None)
    return results


def build_sensor_paths_for_ids(
    ids: Sequence[Optional[str]],
    *,
    manifest: pd.DataFrame,
    id_col: str,
    path_col: str,
    root: Path | str,
) -> Tuple[List[Optional[str]], Optional[SensorSpec]]:
    lookup = _resolve_asset_paths(manifest, id_col=id_col, path_col=path_col, root=Path(root))
    results: List[Optional[str]] = []
    for i in ids:
        key = canonicalize_identifier(i)
        results.append(lookup.get(key) if key else None)

    existing = [Path(p) for p in results if p]
    if not existing:
        return results, None

    cols, length = _find_sensor_columns(existing[0])
    if not cols or length <= 0:
        return results, None

    target_len = min(512, length)
    return results, SensorSpec(columns=cols, target_len=target_len)


def analyse(
    *,
    root: Path,
    tabular: Path,
    tab_id: str,
    image_manifest: Optional[Path],
    image_id: str,
    image_path_col: str,
    sensor_manifest: Optional[Path],
    sensor_id: str,
    sensor_path_col: str,
    export_csv: Optional[Path],
) -> None:
    tab_df = _load_csv(tabular)
    tab_df = tab_df.copy()
    tab_df["__canonical_id"] = canonicalize_series(tab_df[tab_id]).fillna("")

    canonical_ids = tab_df["__canonical_id"].replace({None: ""}).astype(str)
    unique_ids = pd.Index(sorted(set(i for i in canonical_ids if i)))

    print("Tabular summary")
    print("--------------")
    print(f"Rows: {len(tab_df):,}")
    print(f"Unique canonical IDs: {len(unique_ids):,}")
    if canonical_ids.eq("").any():
        missing = canonical_ids.eq("").sum()
        print(f"WARNING: {missing} rows have empty canonical IDs after normalisation.")
    print()

    tab_groups = _group_raw_ids(tab_df, tab_id, "__canonical_id")
    debug_rows = pd.DataFrame({"canonical_id": unique_ids})
    debug_rows["tabular_ids"] = (
        debug_rows["canonical_id"].map(tab_groups).apply(lambda v: v or [])
    )

    if image_manifest is not None:
        img_df = _load_csv(image_manifest)
        img_df = img_df.copy()
        img_df["__canonical_id"] = canonicalize_series(img_df[image_id]).fillna("")
        lookup = build_image_paths_for_ids(
            unique_ids.tolist(),
            manifest=img_df,
            id_col=image_id,
            path_col=image_path_col,
            root=str(root),
        )
        img_groups = _group_raw_ids(img_df, image_id, "__canonical_id")
        debug_rows["image_manifest_ids"] = debug_rows["canonical_id"].map(img_groups).apply(lambda v: v or [])
        debug_rows["image_path"] = pd.Series(_as_optional_list(lookup))
        debug_rows["image_path_exists"] = debug_rows["image_path"].apply(lambda p: bool(p) and os.path.exists(p))

        missing = debug_rows[debug_rows["image_path"].isna() | debug_rows["image_path"].eq("")]
        print("Image manifest summary")
        print("-----------------------")
        print(f"Rows: {len(img_df):,}")
        print(f"Unique canonical IDs: {img_df['__canonical_id'].nunique():,}")
        print(f"IDs matched to assets: {len(debug_rows) - len(missing):,}")
        print(f"IDs without assets: {len(missing):,}")
        if not missing.empty:
            print("Examples of IDs without resolved image paths:")
            print(missing.head(10).to_string(index=False))
        print()

    if sensor_manifest is not None:
        sens_df = _load_csv(sensor_manifest)
        sens_df = sens_df.copy()
        sens_df["__canonical_id"] = canonicalize_series(sens_df[sensor_id]).fillna("")
        paths, spec = build_sensor_paths_for_ids(
            unique_ids.tolist(),
            manifest=sens_df,
            id_col=sensor_id,
            path_col=sensor_path_col,
            root=str(root),
        )
        sens_groups = _group_raw_ids(sens_df, sensor_id, "__canonical_id")
        debug_rows["sensor_manifest_ids"] = debug_rows["canonical_id"].map(sens_groups).apply(lambda v: v or [])
        debug_rows["sensor_path"] = pd.Series(_as_optional_list(paths))
        debug_rows["sensor_path_exists"] = debug_rows["sensor_path"].apply(lambda p: bool(p) and os.path.exists(p))

        missing = debug_rows[debug_rows["sensor_path"].isna() | debug_rows["sensor_path"].eq("")]
        print("Sensor manifest summary")
        print("-----------------------")
        print(f"Rows: {len(sens_df):,}")
        print(f"Unique canonical IDs: {sens_df['__canonical_id'].nunique():,}")
        matched = len(debug_rows) - len(missing)
        print(f"IDs matched to assets: {matched:,}")
        print(f"IDs without assets: {len(missing):,}")
        if spec is not None:
            print(
                "Sensor spec: "
                f"{spec.channels} channels, target length {spec.target_len}"
            )
        if not missing.empty:
            print("Examples of IDs without resolved sensor paths:")
            print(missing.head(10).to_string(index=False))
        print()

    if export_csv is not None:
        export_csv.parent.mkdir(parents=True, exist_ok=True)
        debug_rows.to_csv(export_csv, index=False)
        print(f"Detailed alignment table written to {export_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Base directory containing multimodal artefacts.")
    parser.add_argument("--tabular", type=str, default="tabular.csv", help="Relative path of the tabular CSV file.")
    parser.add_argument("--tab-id", type=str, default="id", help="Identifier column in the tabular CSV.")
    parser.add_argument("--image-manifest", type=str, default=None, help="Relative path of the image manifest CSV.")
    parser.add_argument("--image-id", type=str, default="id", help="Identifier column in the image manifest.")
    parser.add_argument("--image-path-col", type=str, default="image", help="Column containing image file paths.")
    parser.add_argument("--sensor-manifest", type=str, default=None, help="Relative path of the sensor manifest CSV.")
    parser.add_argument("--sensor-id", type=str, default="id", help="Identifier column in the sensor manifest.")
    parser.add_argument("--sensor-path-col", type=str, default="file", help="Column containing sensor file paths.")
    parser.add_argument(
        "--export-csv",
        type=Path,
        default=None,
        help="Optional path to save the combined alignment table as CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()

    def _resolve_optional(path_str: Optional[str]) -> Optional[Path]:
        if path_str is None:
            return None
        path = Path(path_str)
        if not path.is_absolute():
            path = root / path
        return path

    tab_path = args.tabular
    tab_path = Path(tab_path)
    if not tab_path.is_absolute():
        tab_path = root / tab_path

    image_manifest = _resolve_optional(args.image_manifest)
    sensor_manifest = _resolve_optional(args.sensor_manifest)
    export_csv = args.export_csv
    if export_csv is not None and not export_csv.is_absolute():
        export_csv = Path.cwd() / export_csv

    analyse(
        root=root,
        tabular=tab_path,
        tab_id=args.tab_id,
        image_manifest=image_manifest,
        image_id=args.image_id,
        image_path_col=args.image_path_col,
        sensor_manifest=sensor_manifest,
        sensor_id=args.sensor_id,
        sensor_path_col=args.sensor_path_col,
        export_csv=export_csv,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()


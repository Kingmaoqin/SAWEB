"""Utilities to keep pycox usable on read-only package paths.

Some deployment environments mount the virtual environment as read-only.
`pycox` tries to create a ``datasets/data`` folder inside its package
directory during import, which raises ``PermissionError`` in those cases.

We pre-create a writable copy of the package (when necessary) and push that
copy ahead of the default site-packages directory on ``sys.path`` so pycox can
initialize without touching the original location.
"""

from importlib.util import find_spec
import shutil
import sys
import tempfile
from pathlib import Path


def ensure_pycox_writable() -> None:
    """Make sure pycox imports even if site-packages is read-only.

    If ``pycox`` resides in a read-only directory, copy it to a writable temp
    directory and point ``sys.path`` there before the first pycox import.
    """

    spec = find_spec("pycox")
    if not spec or not spec.submodule_search_locations:
        return

    pkg_root = Path(spec.submodule_search_locations[0])
    data_dir = pkg_root / "datasets" / "data"

    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        return
    except PermissionError:
        pass

    cache_root = Path(tempfile.gettempdir()) / "pycox_pkg"
    cache_pkg = cache_root / "pycox"

    if not cache_pkg.exists():
        shutil.copytree(pkg_root, cache_pkg, dirs_exist_ok=True)

    cache_data = cache_pkg / "datasets" / "data"
    cache_data.mkdir(parents=True, exist_ok=True)

    cache_parent = str(cache_root)
    if cache_parent not in sys.path:
        sys.path.insert(0, cache_parent)

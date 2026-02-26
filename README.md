# SAWEB
DHAI LAB SA TOOL

## TEXGISA dual-backend switch

TEXGISA now supports dual execution backends without changing existing UI flow:

- `internal` (default): uses in-repo implementation (`models/mysa.py`)
- `package`: uses `texgisa-survival` package backend
- `auto`: tries package backend first, then falls back to internal backend on failure

Set backend via environment variable before launching Streamlit:

```bash
export TEXGISA_BACKEND=internal   # or package / auto
```

You can also pass `texgisa_backend` in the runtime config dictionary.

## TEXGISA package backend installation note

To keep Streamlit Cloud deployments stable, this repo does **not** pin `texgisa-survival` in `requirements.txt`.
The app defaults to `internal` backend, so startup does not depend on the external package.

If you want to enable the package backend in an environment that allows VCS installs, install it manually:

```bash
pip install "git+https://github.com/Kunjoe7/texgisa-survival.git@main"
```

Then set:

```bash
export TEXGISA_BACKEND=package   # or auto
```


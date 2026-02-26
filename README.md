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

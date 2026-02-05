from __future__ import annotations

import os
from pathlib import Path


def load_env() -> None:
    """Minimal .env loader (KEY=VALUE). Skips if OPENAI_API_KEY already set."""
    if os.getenv("OPENAI_API_KEY"):
        return

    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return

    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val

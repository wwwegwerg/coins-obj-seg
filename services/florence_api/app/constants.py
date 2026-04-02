import os
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    return value in {"1", "true", "yes", "on"}


FLORENCE_MODEL_ID = os.getenv("FLORENCE_MODEL_ID", "florence-community/Florence-2-large-ft").strip()
FLORENCE_MODEL_DIR = Path(
    os.getenv("FLORENCE_MODEL_DIR", str(APP_DIR / "models" / "florence-2-large-ft")).strip()
)
PRELOAD_MODELS = _get_bool("PRELOAD_MODELS", True)

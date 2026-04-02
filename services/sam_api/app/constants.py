import os
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    return value in {"1", "true", "yes", "on"}


SAM_MODEL_ID = os.getenv("SAM_MODEL_ID", "facebook/sam2.1-hiera-large").strip()
SAM_MODEL_DIR = Path(
    os.getenv("SAM_MODEL_DIR", str(APP_DIR / "models" / "sam2.1-hiera-large")).strip()
)
PRELOAD_MODELS = _get_bool("PRELOAD_MODELS", True)

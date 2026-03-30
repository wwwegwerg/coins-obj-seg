import os
from pathlib import Path

from dotenv import load_dotenv


DOTENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=DOTENV_PATH, override=False)


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Environment variable {name} is required for sam-api startup.")
    return value


def _require_bool(name: str) -> bool:
    value = _require_env(name).lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(
        f"Environment variable {name} must be a boolean "
        "(one of: 1,true,yes,on,0,false,no,off)."
    )


SAM_MODEL_ID = _require_env("SAM_MODEL_ID")
SAM_MODEL_DIR = Path(_require_env("SAM_MODEL_DIR"))
PRELOAD_MODELS = _require_bool("PRELOAD_MODELS")

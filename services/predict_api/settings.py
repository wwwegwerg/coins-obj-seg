import os
from pathlib import Path

from dotenv import load_dotenv


DOTENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=DOTENV_PATH, override=False)


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Environment variable {name} is required for predict-api startup."
        )
    return value.rstrip("/")


def _require_positive_float(name: str) -> float:
    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        raise RuntimeError(
            f"Environment variable {name} is required for predict-api startup."
        )
    parsed = float(raw_value)
    if parsed <= 0:
        raise RuntimeError(f"Environment variable {name} must be > 0.")
    return parsed


FLORENCE_API_URL = _require_env("FLORENCE_API_URL")
SAM_API_URL = _require_env("SAM_API_URL")
PREDICT_HTTP_TIMEOUT_SECONDS = _require_positive_float("PREDICT_HTTP_TIMEOUT_SECONDS")
PREDICT_READINESS_TIMEOUT_SECONDS = _require_positive_float("PREDICT_READINESS_TIMEOUT_SECONDS")


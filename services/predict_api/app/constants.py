import os


def _get_url(name: str, default: str) -> str:
    return os.getenv(name, default).strip().rstrip("/")


def _get_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)).strip())


FLORENCE_API_URL = _get_url("FLORENCE_API_URL", "http://localhost:8001")
SAM_API_URL = _get_url("SAM_API_URL", "http://localhost:8002")
PREDICT_HTTP_TIMEOUT_SECONDS = _get_float("PREDICT_HTTP_TIMEOUT_SECONDS", 300.0)
PREDICT_READINESS_TIMEOUT_SECONDS = _get_float("PREDICT_READINESS_TIMEOUT_SECONDS", 3.0)

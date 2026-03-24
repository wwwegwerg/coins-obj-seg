import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoProcessor


logger = logging.getLogger(__name__)
load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)

def _env_or_default(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


MODELS_DIR = Path(_env_or_default("MODELS_DIR", "models"))
FLORENCE_MODEL_ID = _env_or_default("FLORENCE_MODEL_ID", "microsoft/Florence-2-base-ft")
FLORENCE_MODEL_DIR = Path(
    _env_or_default("FLORENCE_MODEL_DIR", str(MODELS_DIR / "florence-2-base-ft"))
)

_state_lock = threading.Lock()
_resources: "FlorenceResources | None" = None


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class FlorenceResources:
    device: torch.device
    processor: AutoProcessor
    model: AutoModelForCausalLM


def _ensure_model_downloaded(repo_id: str, local_dir: Path) -> None:
    if (local_dir / "config.json").exists():
        return
    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading model %s -> %s", repo_id, local_dir)
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))


def load_resources() -> FlorenceResources:
    global _resources

    with _state_lock:
        if _resources is not None:
            return _resources

        device = get_device()
        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
        _ensure_model_downloaded(FLORENCE_MODEL_ID, FLORENCE_MODEL_DIR)

        processor = AutoProcessor.from_pretrained(
            str(FLORENCE_MODEL_DIR),
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(FLORENCE_MODEL_DIR),
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            attn_implementation="eager",
        )
        model.to(device)
        model.eval()

        _resources = FlorenceResources(device=device, processor=processor, model=model)
        return _resources


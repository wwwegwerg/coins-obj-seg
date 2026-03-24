import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from transformers import AutoModelForMaskGeneration, AutoProcessor


logger = logging.getLogger(__name__)
load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)

def _env_or_default(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


MODELS_DIR = Path(_env_or_default("MODELS_DIR", "models"))
SAM_MODEL_ID = _env_or_default("SAM_MODEL_ID", "facebook/sam2.1-hiera-tiny")
SAM_MODEL_DIR = Path(_env_or_default("SAM_MODEL_DIR", str(MODELS_DIR / "sam2.1-hiera-tiny")))

_state_lock = threading.Lock()
_resources: "SamResources | None" = None


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class SamResources:
    device: torch.device
    processor: AutoProcessor
    model: AutoModelForMaskGeneration


def _ensure_model_downloaded(repo_id: str, local_dir: Path) -> None:
    if (local_dir / "config.json").exists():
        return
    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading model %s -> %s", repo_id, local_dir)
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))


def load_resources() -> SamResources:
    global _resources

    with _state_lock:
        if _resources is not None:
            return _resources

        device = get_device()
        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
        _ensure_model_downloaded(SAM_MODEL_ID, SAM_MODEL_DIR)

        processor = AutoProcessor.from_pretrained(str(SAM_MODEL_DIR))
        model = AutoModelForMaskGeneration.from_pretrained(
            str(SAM_MODEL_DIR),
            torch_dtype=torch_dtype,
        )
        model.to(device)
        model.eval()

        _resources = SamResources(device=device, processor=processor, model=model)
        return _resources


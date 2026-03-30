import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, Florence2ForConditionalGeneration

from settings import FLORENCE_MODEL_DIR, FLORENCE_MODEL_ID, PRELOAD_MODELS


logger = logging.getLogger(__name__)

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
    model: Florence2ForConditionalGeneration


def _ensure_model_downloaded(repo_id: str, local_dir: Path) -> str:
    if (local_dir / "config.json").exists():
        return "cached"
    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading model %s -> %s", repo_id, local_dir)
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
    return "downloaded_now"


def load_resources() -> FlorenceResources:
    global _resources

    with _state_lock:
        if _resources is not None:
            return _resources

        device = get_device()
        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
        model_source = _ensure_model_downloaded(FLORENCE_MODEL_ID, FLORENCE_MODEL_DIR)

        processor = AutoProcessor.from_pretrained(
            str(FLORENCE_MODEL_DIR),
            use_fast=True,
        )
        model = Florence2ForConditionalGeneration.from_pretrained(
            str(FLORENCE_MODEL_DIR),
            torch_dtype=torch_dtype,
            attn_implementation="eager",
        )
        model.to(device)
        model.eval()

        if device.type == "cuda" and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(device)
            logger.info(
                "Model infra: model_id=%s device=%s preload_models=%s model_source=%s gpu_name=%s gpu_count=%d compute_capability=%d.%d total_vram_mb=%d",
                FLORENCE_MODEL_ID,
                device.type,
                PRELOAD_MODELS,
                model_source,
                props.name,
                torch.cuda.device_count(),
                props.major,
                props.minor,
                int(props.total_memory / (1024 * 1024)),
            )
        else:
            logger.info(
                "Model infra: model_id=%s device=%s preload_models=%s model_source=%s",
                FLORENCE_MODEL_ID,
                device.type,
                PRELOAD_MODELS,
                model_source,
            )

        _resources = FlorenceResources(device=device, processor=processor, model=model)
        return _resources


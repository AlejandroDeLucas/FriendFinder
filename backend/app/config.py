"""Configuration, storage paths, and hardware acceleration detection."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

# Base project directory (FriendFinder/)
BACKEND_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BACKEND_DIR.parent
DATA_DIR = Path(os.environ.get("FRIENDFINDER_DATA_DIR", PROJECT_ROOT / "data")).resolve()

UPLOADS_DIR = DATA_DIR / "uploads"
FRAMES_DIR = DATA_DIR / "frames"
PORTRAITS_DIR = DATA_DIR / "portraits"
CUTOUTS_DIR = DATA_DIR / "cutouts"
CROPS_DIR = DATA_DIR / "crops"
THUMBS_DIR = DATA_DIR / "thumbs"
EXPORTS_DIR = DATA_DIR / "exports"
MODELS_DIR = DATA_DIR / "models"
DB_PATH = DATA_DIR / "friendfinder.db"

ALL_DATA_DIRS = [
    DATA_DIR,
    UPLOADS_DIR,
    FRAMES_DIR,
    PORTRAITS_DIR,
    CUTOUTS_DIR,
    CROPS_DIR,
    THUMBS_DIR,
    EXPORTS_DIR,
    MODELS_DIR,
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def ensure_data_dirs() -> None:
    """Create all required data directories if they do not exist."""
    for directory in ALL_DATA_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def detect_compute_device() -> dict[str, str]:
    """Auto-detect available hardware acceleration (CUDA, MPS, or CPU)."""
    torch_device = "cpu"
    onnx_providers = ["CPUExecutionProvider"]

    try:
        import torch

        if torch.cuda.is_available():
            torch_device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch_device = "mps"
    except Exception:
        pass

    try:
        import onnxruntime as ort

        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            onnx_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif "CoreMLExecutionProvider" in available:
            onnx_providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass

    return {
        "torch_device": torch_device,
        "onnx_provider": onnx_providers[0],
        "onnx_providers": ",".join(onnx_providers),
    }


DEFAULT_SETTINGS: dict[str, str] = {
    "video_sample_fps": "1.0",
    "similarity_threshold": "0.45",
    "min_quality_score": "32.0",
    "max_portraits_per_person": "6",
    "max_fullbody_per_person": "4",
    "min_video_gap_seconds": "1.5",
    "auto_scene_analysis": "false",
    "scene_backend": "heuristic",  # heuristic | blip | ollama
    "ollama_url": "http://localhost:11434",
    "ollama_model": "llava",
    "yolo_model": "yolov8n-seg.pt",
    "insightface_pack": "buffalo_l",
}

SceneBackend = Literal["heuristic", "blip", "ollama"]

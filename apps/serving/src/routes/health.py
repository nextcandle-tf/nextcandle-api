"""Health check routes."""

import torch
from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def root() -> dict:
    """Root endpoint."""
    return {"status": "ok", "service": "nextcandle-serving"}


@router.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_info = None

    if torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB",
            "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1e9:.2f} GB",
        }

    return {
        "status": "healthy",
        "device": device,
        "gpu": gpu_info,
        "torch_version": torch.__version__,
    }

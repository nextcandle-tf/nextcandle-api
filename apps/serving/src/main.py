"""FastAPI ML Serving Application."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.routes import health, pattern


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    # Startup
    device = "cuda" if torch.cuda.is_available() and settings.use_gpu else "cpu"
    print(f"ML Serving starting on device: {device}")
    if device == "cuda":
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    yield

    # Shutdown
    print("ML Serving shutting down...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="NextCandle ML Serving",
    description="ML model serving for cryptocurrency pattern detection",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(health.router, tags=["Health"])
app.include_router(pattern.router, prefix="/api/pattern", tags=["Pattern Detection"])

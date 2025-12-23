"""Pattern detection routes."""

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()


class PatternRequest(BaseModel):
    """Pattern detection request."""

    candles: list[list[float]] = Field(
        ...,
        description="OHLCV candle data: [[open, high, low, close, volume], ...]",
        min_length=3,
        max_length=100,
    )
    top_k: int = Field(default=10, ge=1, le=100, description="Number of similar patterns to return")


class PatternMatch(BaseModel):
    """A matched pattern."""

    similarity: float
    start_index: int
    end_index: int
    timestamp: str | None = None


class PatternResponse(BaseModel):
    """Pattern detection response."""

    success: bool
    matches: list[PatternMatch]
    query_length: int


@router.post("/detect", response_model=PatternResponse)
async def detect_pattern(request: PatternRequest) -> dict[str, Any]:
    """
    Detect similar patterns from historical data.

    This endpoint will be implemented to:
    1. Encode the input candles using the trained PatternEncoder model
    2. Find similar patterns from pre-computed embeddings
    3. Return top-k most similar patterns
    """
    # TODO: Implement actual pattern detection
    # This is a placeholder for the boilerplate
    raise HTTPException(
        status_code=501,
        detail="Pattern detection not yet implemented. Add your model loading and inference logic.",
    )


@router.get("/embeddings/info")
async def embeddings_info() -> dict[str, Any]:
    """Get information about loaded embeddings."""
    # TODO: Return actual embedding info
    return {
        "loaded": False,
        "message": "Embeddings not yet loaded. Implement model loading in lifespan handler.",
    }

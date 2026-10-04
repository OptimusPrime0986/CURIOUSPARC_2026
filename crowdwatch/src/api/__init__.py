"""API package — FastAPI control-room application and real-time pipeline service."""

from src.api.app import app
from src.api.service import PipelineService, get_pipeline

__all__ = [
    "app",
    "PipelineService",
    "get_pipeline",
]

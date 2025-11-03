"""
Tracking Service - Main entry point for Project Argus Multi-Camera Tracking
"""

import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from shared.interfaces.tracking import IMultiCameraTracker, IReIDMatcher, IBehaviorAnalyzer
from shared.models import Track, GlobalTrack, Detection


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Tracking Service...")
    # Initialize tracking services here
    yield
    logger.info("Shutting down Tracking Service...")


# Create FastAPI application
app = FastAPI(
    title="Project Argus Tracking Service",
    description="Multi-camera tracking and re-identification service for border detection system",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "tracking-service"}


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Project Argus Tracking Service", "version": "1.0.0"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8004,
        reload=True,
        log_level="info"
    )
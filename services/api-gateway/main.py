"""
API Gateway - Main entry point for Project Argus API Gateway
"""

import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

# Removed complex imports for development mode
# from shared.interfaces.security import IAuthenticationService, IAuthorizationService
# from shared.models import User


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting API Gateway...")
    # Initialize gateway services, database connections, etc.
    yield
    logger.info("Shutting down API Gateway...")


# Create FastAPI application
app = FastAPI(
    title="Project Argus API Gateway",
    description="Central API Gateway for Project Argus Border Detection System",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
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
    return {"status": "healthy", "service": "api-gateway"}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Project Argus API Gateway", 
        "version": "1.0.0",
        "services": {
            "alert-service": "http://alert-service:8003",
            "tracking-service": "http://tracking-service:8004", 
            "evidence-service": "http://evidence-service:8005"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
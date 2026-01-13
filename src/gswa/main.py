"""GSWA FastAPI Application.

Main entry point for the Gilles-Style Writing Assistant API.
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from gswa.config import get_settings
from gswa.api.routes import router


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting GSWA server...")
    settings = get_settings()
    logger.info(f"LLM server: {settings.vllm_base_url}")
    logger.info(f"External API allowed: {settings.allow_external_api}")

    yield

    # Shutdown
    logger.info("Shutting down GSWA server...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Returns:
        Configured FastAPI application instance
    """
    settings = get_settings()  # This validates security settings

    app = FastAPI(
        title="GSWA - Gilles-Style Writing Assistant",
        description="Local offline scientific paper rewriter",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routes
    app.include_router(router)

    # Serve static files (web UI)
    web_dir = Path(__file__).parent.parent.parent.parent / "web"
    if web_dir.exists():
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")
        logger.info(f"Serving web UI from {web_dir}")
    else:
        logger.warning(f"Web UI directory not found at {web_dir}, skipping static file serving")

    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

"""
ImmunoForge Web Server — FastAPI application for local deployment.

Provides:
  - REST API for pipeline execution and result retrieval
  - Web dashboard for interactive design sessions
  - Target database browser
  - Real-time pipeline progress
"""

import logging
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from immunoforge.core.utils import load_config
from immunoforge.server.api import router as api_router
from immunoforge.server.api_pipeline import router as pipeline_router

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(
    title="ImmunoForge",
    description="De novo immune cell targeting protein design platform",
    version="0.1.0",
)

# Mount static files
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Templates
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Include API routes
app.include_router(api_router, prefix="/api")
app.include_router(pipeline_router, prefix="/api")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the main dashboard."""
    return templates.TemplateResponse(request, "index.html")


@app.get("/targets", response_class=HTMLResponse)
async def targets_page(request: Request):
    """Target database browser page."""
    return templates.TemplateResponse(request, "targets.html")


@app.get("/results", response_class=HTMLResponse)
async def results_page(request: Request):
    """Pipeline results viewer page."""
    return templates.TemplateResponse(request, "results.html")


@app.get("/pipeline", response_class=HTMLResponse)
async def pipeline_page(request: Request):
    """Full pipeline dashboard with step-by-step visualization."""
    return templates.TemplateResponse(request, "pipeline.html")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "ImmunoForge"}


def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the ImmunoForge web server."""
    config = {}
    try:
        config = load_config()
    except FileNotFoundError:
        pass

    server_cfg = config.get("server", {})
    host = server_cfg.get("host", host)
    port = server_cfg.get("port", port)

    logger.info(f"Starting ImmunoForge server on {host}:{port}")
    uvicorn.run(
        "immunoforge.server.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    run_server()

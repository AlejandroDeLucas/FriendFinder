"""FriendFinder FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import (
    CROPS_DIR,
    CUTOUTS_DIR,
    FRAMES_DIR,
    PORTRAITS_DIR,
    PROJECT_ROOT,
    THUMBS_DIR,
    ensure_data_dirs,
)
from .routers import detections, jobs, media, persons, search

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_data_dirs()
    yield


app = FastAPI(
    title="FriendFinder API",
    description="Extracción local de mejores retratos y recortes PNG de personas en fotos y vídeos",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(media.router)
app.include_router(persons.router)
app.include_router(detections.router)
app.include_router(search.router)
app.include_router(jobs.router)

# Ensure asset directories exist before mounting static routes
ensure_data_dirs()
app.mount("/assets/portraits", StaticFiles(directory=str(PORTRAITS_DIR)), name="portraits")
app.mount("/assets/cutouts", StaticFiles(directory=str(CUTOUTS_DIR)), name="cutouts")
app.mount("/assets/crops", StaticFiles(directory=str(CROPS_DIR)), name="crops")
app.mount("/assets/frames", StaticFiles(directory=str(FRAMES_DIR)), name="frames")
app.mount("/assets/thumbs", StaticFiles(directory=str(THUMBS_DIR)), name="thumbs")


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


# Optionally serve built frontend production bundle if present
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
if FRONTEND_DIST.exists() and (FRONTEND_DIST / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")

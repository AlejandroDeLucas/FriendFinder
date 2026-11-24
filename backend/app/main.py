"""FriendFinder FastAPI application entry point (local + Cloudflare Tunnel/Container ready)."""

from __future__ import annotations

import asyncio
import hmac
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .database import db
from .routers import detections, jobs, media, persons, search
from .schemas import PinVerifyRequest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def _expired_source_reaper() -> None:
    """Background loop that auto-deletes expired uploaded source files (default 1h)
    while keeping all generated portraits, PNG cutouts, crops, and person profiles.
    """
    while True:
        try:
            res = db.purge_expired_sources()
            if res["purged_count"] > 0:
                logger.info(
                    "Auto-limpieza 1h: eliminados %d archivo(s) fuente (%d bytes); capturas generadas conservadas.",
                    res["purged_count"],
                    res["freed_bytes"],
                )
        except Exception:
            logger.exception("Error en el limpiador automático de archivos expirados")
        await asyncio.sleep(30)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    config.ensure_data_dirs()
    reaper_task = asyncio.create_task(_expired_source_reaper())
    try:
        yield
    finally:
        reaper_task.cancel()
        try:
            await reaper_task
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="FriendFinder API",
    description="Extracción local de mejores retratos y recortes PNG de personas en fotos y vídeos",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def optional_pin_guard(request: Request, call_next):
    """When FRIENDFINDER_PIN is set (recommended for Cloudflare Tunnel remote access),
    require matching X-Access-Pin header or ?pin= query param on API and asset routes.
    """
    required_pin = config.ACCESS_PIN
    if not required_pin or request.method == "OPTIONS":
        return await call_next(request)

    path = request.url.path
    if path.startswith("/api/health") or path.startswith("/api/auth"):
        return await call_next(request)

    protected_asset_prefixes = (
        "/assets/portraits/",
        "/assets/cutouts/",
        "/assets/crops/",
        "/assets/frames/",
        "/assets/thumbs/",
    )
    if path.startswith("/api/") or path.startswith(protected_asset_prefixes):
        provided = (
            request.headers.get("x-access-pin")
            or request.query_params.get("pin")
            or ""
        ).strip()
        if not hmac.compare_digest(provided, required_pin):
            return JSONResponse(
                status_code=401,
                content={"detail": "PIN de acceso requerido"},
            )

    return await call_next(request)


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/auth/status")
def auth_status() -> dict[str, bool]:
    return {"pin_required": bool(config.ACCESS_PIN)}


@app.post("/api/auth/verify")
def auth_verify(payload: PinVerifyRequest) -> dict[str, bool]:
    required_pin = config.ACCESS_PIN
    if not required_pin:
        return {"valid": True}
    valid = hmac.compare_digest(payload.pin.strip(), required_pin)
    if not valid:
        return JSONResponse(status_code=401, content={"valid": False, "detail": "PIN incorrecto"})  # type: ignore[return-value]
    return {"valid": True}


# Register API routers
app.include_router(media.router)
app.include_router(persons.router)
app.include_router(detections.router)
app.include_router(search.router)
app.include_router(jobs.router)

# Ensure asset directories exist before mounting static routes
config.ensure_data_dirs()
app.mount("/assets/portraits", StaticFiles(directory=str(config.PORTRAITS_DIR)), name="portraits")
app.mount("/assets/cutouts", StaticFiles(directory=str(config.CUTOUTS_DIR)), name="cutouts")
app.mount("/assets/crops", StaticFiles(directory=str(config.CROPS_DIR)), name="crops")
app.mount("/assets/frames", StaticFiles(directory=str(config.FRAMES_DIR)), name="frames")
app.mount("/assets/thumbs", StaticFiles(directory=str(config.THUMBS_DIR)), name="thumbs")

# Serve built React production SPA if present (single-port mode ideal for Cloudflare Tunnel & Docker)
FRONTEND_DIST = config.PROJECT_ROOT / "frontend" / "dist"
if FRONTEND_DIST.exists() and (FRONTEND_DIST / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")

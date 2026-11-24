"""Background job progress, system stats, and local settings endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from ..database import db
from ..schemas import SettingsUpdateRequest
from ..services.ml_models import model_manager

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/jobs")
def list_jobs() -> list[dict[str, Any]]:
    return db.list_jobs(limit=25)


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado")
    return job


@router.get("/settings")
def get_settings() -> dict[str, Any]:
    return {
        "settings": db.get_settings(),
        "device": model_manager.device_info,
    }


@router.patch("/settings")
def update_settings(payload: SettingsUpdateRequest) -> dict[str, Any]:
    updates = payload.model_dump(exclude_none=True)
    updated = db.update_settings(updates)
    return {
        "settings": updated,
        "device": model_manager.device_info,
    }


@router.get("/stats")
def get_stats() -> dict[str, Any]:
    persons = db.list_persons(include_ignored=False)
    media = db.list_media()
    photos = [m for m in media if m["media_type"] == "image"]
    videos = [m for m in media if m["media_type"] == "video"]
    active_jobs = [j for j in db.list_jobs(limit=10) if j["status"] in ("queued", "running")]

    return {
        "persons_count": len(persons),
        "photos_count": len(photos),
        "videos_count": len(videos),
        "kept_captures": sum(p["kept_count"] for p in persons),
        "cutouts_count": sum(p["cutout_count"] for p in persons),
        "discarded_captures": sum(p["discarded_count"] for p in persons),
        "active_jobs": active_jobs,
        "device": model_manager.device_info,
    }

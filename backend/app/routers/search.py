"""Reverse search router: find a person by uploading any reference photo."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..services.search import search_by_image_bytes

router = APIRouter(prefix="/api/search", tags=["search"])


@router.post("/by-image")
async def search_by_image(
    file: UploadFile = File(...),
    threshold: float | None = Form(default=None),
    face_index: int = Form(default=0),
) -> dict[str, Any]:
    """Upload a reference photo to find whether that person appears in saved profiles, photos, or videos."""
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="El archivo está vacío")
    return search_by_image_bytes(
        raw,
        threshold=threshold,
        face_index=face_index,
    )

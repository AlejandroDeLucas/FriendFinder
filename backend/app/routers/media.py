"""Media upload, listing, Range video streaming, and deletion endpoints."""

from __future__ import annotations

import mimetypes
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from ..config import UPLOADS_DIR, ensure_data_dirs
from ..database import db
from ..services.pipeline import classify_media_type, submit_media_batch

router = APIRouter(prefix="/api/media", tags=["media"])


@router.post("/upload")
async def upload_media(
    files: list[UploadFile] = File(...),
    sample_fps: float | None = Form(default=None),
    job_type: str = Form(default="ingest"),
) -> dict[str, Any]:
    """Upload one or more photos and/or videos and start background frame-by-frame analysis."""
    ensure_data_dirs()
    media_ids: list[str] = []
    skipped: list[dict[str, str]] = []

    for upload in files:
        filename = upload.filename or "unnamed"
        media_type = classify_media_type(filename)
        if media_type is None:
            skipped.append({"filename": filename, "reason": "Formato no soportado"})
            continue

        media_id = uuid.uuid4().hex[:12]
        ext = Path(filename).suffix.lower()
        dest_path = UPLOADS_DIR / f"{media_id}{ext}"

        size = 0
        with dest_path.open("wb") as out:
            while chunk := await upload.read(1024 * 1024):
                out.write(chunk)
                size += len(chunk)

        db.create_media(
            media_id=media_id,
            original_filename=filename,
            media_type=media_type,
            file_path=str(dest_path),
            file_size=size,
        )
        media_ids.append(media_id)

    if not media_ids:
        raise HTTPException(
            status_code=400,
            detail="No se subió ningún archivo de imagen o vídeo válido.",
        )

    job_id = submit_media_batch(
        media_ids,
        sample_fps_override=sample_fps,
        job_type=job_type,
    )
    return {
        "job_id": job_id,
        "media_ids": media_ids,
        "uploaded_count": len(media_ids),
        "skipped": skipped,
    }


@router.get("")
def list_media(media_type: str | None = Query(default=None)) -> list[dict[str, Any]]:
    return db.list_media(media_type=media_type)


@router.get("/{media_id}")
def get_media_detail(media_id: str) -> dict[str, Any]:
    media = db.get_media(media_id)
    if not media:
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    detections = db.list_detections_for_media(media_id)
    return {**media, "detections": detections}


@router.get("/{media_id}/stream")
def stream_media(media_id: str, request: Request) -> Any:
    """Serve original photo or stream video with HTTP Range support for instant timestamp seeking."""
    media = db.get_media(media_id)
    if not media:
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    if media.get("source_deleted"):
        raise HTTPException(status_code=410, detail="El archivo fuente fue eliminado")

    file_path = Path(media["file_path"])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Archivo no encontrado en disco")

    content_type, _ = mimetypes.guess_type(str(file_path))
    content_type = content_type or (
        "video/mp4" if media["media_type"] == "video" else "image/jpeg"
    )

    if media["media_type"] == "image":
        return FileResponse(path=file_path, media_type=content_type)

    file_size = file_path.stat().st_size
    range_header = request.headers.get("range")

    if not range_header:
        return FileResponse(
            path=file_path,
            media_type=content_type,
            headers={"Accept-Ranges": "bytes"},
        )

    try:
        byte_range = range_header.strip().lower().replace("bytes=", "")
        start_s, end_s = byte_range.split("-", 1)
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else min(start + 2 * 1024 * 1024 - 1, file_size - 1)
        end = min(end, file_size - 1)
        if start > end or start >= file_size:
            raise HTTPException(status_code=416, detail="Rango inválido")
    except ValueError:
        raise HTTPException(status_code=416, detail="Cabecera Range inválida")

    chunk_len = end - start + 1

    def iter_range():
        with file_path.open("rb") as fh:
            fh.seek(start)
            remaining = chunk_len
            while remaining > 0:
                data = fh.read(min(64 * 1024, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(chunk_len),
    }
    return StreamingResponse(
        iter_range(), status_code=206, media_type=content_type, headers=headers
    )


@router.delete("/{media_id}")
def delete_media(
    media_id: str,
    keep_person_assets: bool = Query(default=True),
) -> dict[str, Any]:
    """Delete a source video/photo. By default preserves all extracted person portraits & cutouts!"""
    media = db.get_media(media_id)
    if not media:
        raise HTTPException(status_code=404, detail="Archivo no encontrado")

    file_path = Path(media["file_path"])
    if file_path.exists():
        try:
            file_path.unlink()
        except OSError:
            pass

    if keep_person_assets:
        db.mark_media_source_deleted(media_id)
        return {
            "deleted": True,
            "mode": "source_only",
            "message": "Archivo original eliminado; las imágenes extraídas de personas se mantienen guardadas.",
        }

    affected_persons = {
        d["person_id"]
        for d in db.list_detections_for_media(media_id)
        if d.get("person_id")
    }
    db.delete_media_completely(media_id)
    for pid in affected_persons:
        db.recompute_person_stats(pid)
    return {
        "deleted": True,
        "mode": "complete",
        "message": "Archivo y sus detecciones eliminados completamente.",
    }

"""Persons gallery, detail, renaming, merging, scene analysis, and ZIP export endpoints."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from ..config import FRAMES_DIR, PORTRAITS_DIR
from ..database import db
from ..schemas import ExportRequest, MergePersonsRequest, PersonUpdateRequest, SceneAnalyzeRequest
from ..services.exporter import build_export_zip
from ..services.pipeline import recluster_all_persons
from ..services.scene_analyzer import scene_analyzer

router = APIRouter(prefix="/api/persons", tags=["persons"])


@router.get("")
def list_persons(include_ignored: bool = Query(default=False)) -> list[dict[str, Any]]:
    return db.list_persons(include_ignored=include_ignored)


@router.post("/recluster")
def recluster_persons() -> dict[str, int]:
    """Re-cluster all stored detections using strict complete-linkage clustering while preserving custom names."""
    return recluster_all_persons(database=db)


@router.get("/{person_id}")
def get_person_detail(person_id: str) -> dict[str, Any]:
    person = db.get_person_summary(person_id)
    if not person:
        raise HTTPException(status_code=404, detail="Persona no encontrada")

    all_dets = db.list_detections_for_person(person_id)
    kept = [d for d in all_dets if d["status"] == "kept"]
    discarded = [d for d in all_dets if d["status"] == "discarded"]

    # Group appearances by media (photos and videos with all timestamps)
    by_media: dict[str, dict[str, Any]] = {}
    ts_map: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for d in all_dets:
        mid = d["media_id"]
        ts_map[mid].append(
            {
                "detection_id": d["id"],
                "timestamp_sec": d["timestamp_sec"],
                "frame_idx": d["frame_idx"],
                "portrait_score": d["portrait_score"],
                "body_score": d["body_score"],
                "status": d["status"],
                "thumb_path": d.get("thumb_path"),
            }
        )
        if mid not in by_media:
            by_media[mid] = {
                "media_id": mid,
                "filename": d.get("media_filename"),
                "media_type": d.get("media_type"),
                "source_deleted": bool(d.get("source_deleted")),
                "appearances_count": 0,
                "best_score": d["overall_score"],
                "timestamps": [],
            }
        else:
            by_media[mid]["best_score"] = max(by_media[mid]["best_score"], d["overall_score"])

    appearances = []
    for mid, item in by_media.items():
        hits = sorted(ts_map[mid], key=lambda x: x["timestamp_sec"])
        item["appearances_count"] = len(hits)
        item["timestamps"] = hits
        appearances.append(item)
    appearances.sort(key=lambda m: m["best_score"], reverse=True)

    return {
        **person,
        "kept_detections": kept,
        "discarded_detections": discarded,
        "media_appearances": appearances,
    }


@router.patch("/{person_id}")
def update_person(person_id: str, payload: PersonUpdateRequest) -> dict[str, Any]:
    existing = db.get_person_summary(person_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Persona no encontrada")
    db.update_person_metadata(
        person_id,
        name=payload.name,
        notes=payload.notes,
        is_ignored=payload.is_ignored,
        cover_detection_id=payload.cover_detection_id,
    )
    updated = db.get_person_summary(person_id)
    assert updated is not None
    return updated


@router.post("/{person_id}/merge")
def merge_persons(person_id: str, payload: MergePersonsRequest) -> dict[str, Any]:
    existing = db.get_person_summary(person_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Persona destino no encontrada")
    merged = db.merge_persons(person_id, payload.source_person_ids)
    assert merged is not None
    return merged


@router.delete("/{person_id}")
def delete_person(person_id: str) -> dict[str, Any]:
    existing = db.get_person_summary(person_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Persona no encontrada")
    db.delete_person(person_id)
    return {"deleted": True, "person_id": person_id}


@router.post("/{person_id}/clear-photos")
def clear_person_photos(person_id: str) -> dict[str, Any]:
    """Delete all saved photos of a person while preserving their profile & facial recognition centroid."""
    existing = db.get_person_summary(person_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Persona no encontrada")
    return db.clear_person_photos(person_id)


@router.post("/{person_id}/analyze-scene")
def analyze_person_scenes(
    person_id: str,
    payload: SceneAnalyzeRequest | None = None,
) -> dict[str, Any]:
    """Run optional local scene/activity analysis on this person's kept captures."""
    person = db.get_person_summary(person_id)
    if not person:
        raise HTTPException(status_code=404, detail="Persona no encontrada")

    settings = db.get_settings()
    backend = (payload.backend if payload and payload.backend else None) or settings.get(
        "scene_backend", "heuristic"
    )
    kept = db.list_detections_for_person(person_id, status="kept")
    updated_count = 0

    for d in kept:
        img_path = None
        if d.get("frame_path") and (FRAMES_DIR / d["frame_path"]).exists():
            img_path = FRAMES_DIR / d["frame_path"]
        elif d.get("portrait_path") and (PORTRAITS_DIR / d["portrait_path"]).exists():
            img_path = PORTRAITS_DIR / d["portrait_path"]
        if not img_path:
            continue

        summary, tags, used_backend = scene_analyzer.analyze_image_file(
            img_path,
            backend=backend,
            ollama_url=settings.get("ollama_url", "http://localhost:11434"),
            ollama_model=settings.get("ollama_model", "llava"),
            fallback_context={
                "is_full_body": d.get("is_full_body", False),
                "frontality_score": d.get("frontality_score", 75.0),
            },
        )
        db.update_detection(
            d["id"],
            scene_summary=summary,
            scene_tags=tags,
            scene_backend=used_backend,
        )
        updated_count += 1

    return {"person_id": person_id, "analyzed_count": updated_count, "backend": backend}


@router.post("/export")
def export_persons_zip(payload: ExportRequest) -> FileResponse:
    """Export a ready-to-use ZIP pack for game avatars/characters."""
    zip_path = build_export_zip(
        person_ids=payload.person_ids,
        include_portraits=payload.include_portraits,
        include_cutouts=payload.include_cutouts,
        include_crops=payload.include_crops,
        include_originals=payload.include_originals,
        only_kept=payload.only_kept,
    )
    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=zip_path.name,
    )

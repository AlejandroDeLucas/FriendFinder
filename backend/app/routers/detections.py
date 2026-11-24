"""Individual detection management: rescue/discard, reassign person, regenerate assets, and scene analysis."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from ..config import FRAMES_DIR, PORTRAITS_DIR, THUMBS_DIR
from ..database import db
from ..schemas import DetectionStatusUpdateRequest, ReassignDetectionRequest, SceneAnalyzeRequest
from ..services.pipeline import regenerate_detection_assets
from ..services.scene_analyzer import scene_analyzer

router = APIRouter(prefix="/api/detections", tags=["detections"])


@router.get("/{det_id}")
def get_detection(det_id: str) -> dict[str, Any]:
    det = db.get_detection(det_id)
    if not det:
        raise HTTPException(status_code=404, detail="Detección no encontrada")
    return det


@router.patch("/{det_id}/status")
def update_detection_status(
    det_id: str, payload: DetectionStatusUpdateRequest
) -> dict[str, Any]:
    """Promote (rescue) a discarded capture to kept (generating all 3 formats) or discard a kept capture."""
    det = db.get_detection(det_id)
    if not det:
        raise HTTPException(status_code=404, detail="Detección no encontrada")

    if payload.status == "kept":
        # Generate full-res portrait, transparent PNG cutout, crop, and frame if missing
        regenerate_detection_assets(det_id)
        updated = db.update_detection(
            det_id,
            status="kept",
            keep_category=payload.keep_category or "manual",
            discard_reason=None,
        )
    else:
        updated = db.update_detection(
            det_id,
            status="discarded",
            discard_reason=payload.discard_reason or "Descartada manualmente",
        )

    if det.get("person_id"):
        db.recompute_person_stats(det["person_id"])
    assert updated is not None
    return updated


@router.post("/{det_id}/reassign")
def reassign_detection(det_id: str, payload: ReassignDetectionRequest) -> dict[str, Any]:
    """Move a detection to a different person or split it into a brand-new person."""
    det = db.get_detection(det_id)
    if not det:
        raise HTTPException(status_code=404, detail="Detección no encontrada")

    old_person_id = det.get("person_id")
    target_person_id = payload.person_id

    if not target_person_id:
        new_name = (payload.new_person_name or "").strip() or f"Persona {db.next_person_number()}"
        created = db.create_person(name=new_name)
        target_person_id = created["id"]
    else:
        if not db.get_person_summary(target_person_id):
            raise HTTPException(status_code=404, detail="Persona destino no encontrada")

    updated = db.update_detection(det_id, person_id=target_person_id)
    if old_person_id:
        db.recompute_person_stats(old_person_id)
    db.recompute_person_stats(target_person_id)
    assert updated is not None
    return updated


@router.post("/{det_id}/analyze-scene")
def analyze_detection_scene(
    det_id: str, payload: SceneAnalyzeRequest | None = None
) -> dict[str, Any]:
    """Run optional local scene/activity analysis on a single detection."""
    det = db.get_detection(det_id)
    if not det:
        raise HTTPException(status_code=404, detail="Detección no encontrada")

    settings = db.get_settings()
    backend = (payload.backend if payload and payload.backend else None) or settings.get(
        "scene_backend", "heuristic"
    )

    img_path = None
    if det.get("frame_path") and (FRAMES_DIR / det["frame_path"]).exists():
        img_path = FRAMES_DIR / det["frame_path"]
    elif det.get("portrait_path") and (PORTRAITS_DIR / det["portrait_path"]).exists():
        img_path = PORTRAITS_DIR / det["portrait_path"]
    elif det.get("thumb_path") and (THUMBS_DIR / det["thumb_path"]).exists():
        img_path = THUMBS_DIR / det["thumb_path"]

    if not img_path:
        raise HTTPException(status_code=400, detail="No hay imagen disponible para analizar")

    summary, tags, used_backend = scene_analyzer.analyze_image_file(
        img_path,
        backend=backend,
        ollama_url=settings.get("ollama_url", "http://localhost:11434"),
        ollama_model=settings.get("ollama_model", "llava"),
        fallback_context={
            "is_full_body": det.get("is_full_body", False),
            "frontality_score": det.get("frontality_score", 75.0),
        },
    )
    updated = db.update_detection(
        det_id,
        scene_summary=summary,
        scene_tags=tags,
        scene_backend=used_backend,
    )
    assert updated is not None
    return updated


@router.delete("/{det_id}")
def delete_detection(det_id: str) -> dict[str, Any]:
    det = db.get_detection(det_id)
    if not det:
        raise HTTPException(status_code=404, detail="Detección no encontrada")
    pid = db.delete_detection(det_id)
    if pid:
        db.recompute_person_stats(pid)
    return {"deleted": True, "detection_id": det_id}

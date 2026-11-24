"""Reverse search by reference photo across saved persons, photos, and non-deleted videos."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import cv2
import numpy as np

from ..database import Database, db as default_db
from .ml_models import model_manager
from .pipeline import cosine_similarity


def search_by_image_bytes(
    image_bytes: bytes,
    *,
    threshold: float | None = None,
    face_index: int = 0,
    database: Database | None = None,
) -> dict[str, Any]:
    """Find a person from an uploaded reference photo across known profiles, photos, and videos."""
    db = database or default_db
    settings = db.get_settings()
    sim_threshold = (
        threshold if threshold is not None else max(0.35, float(settings.get("similarity_threshold", "0.45")) - 0.03)
    )

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return {
            "found_faces": 0,
            "selected_face_index": 0,
            "query_faces": [],
            "matched_persons": [],
            "matched_media": [],
            "message": "No se pudo leer la imagen de consulta.",
        }

    faces = model_manager.detect_faces_only(
        img_bgr, pack=settings.get("insightface_pack", "buffalo_l")
    )
    if not faces:
        return {
            "found_faces": 0,
            "selected_face_index": 0,
            "query_faces": [],
            "matched_persons": [],
            "matched_media": [],
            "message": "No se detectó ningún rostro claro en la imagen de búsqueda.",
        }

    # Sort faces by descending area so face 0 is the main/prominent person
    faces = sorted(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True,
    )
    chosen_idx = max(0, min(len(faces) - 1, face_index))
    query_emb = faces[chosen_idx].embedding

    query_faces_meta = [
        {
            "index": i,
            "bbox": [round(v, 1) for v in f.bbox],
            "det_score": round(f.det_score, 3),
        }
        for i, f in enumerate(faces)
    ]

    # 1. Match against saved Person profiles
    matched_persons: list[dict[str, Any]] = []
    for pid, _name, centroid in db.get_all_person_embeddings():
        sim = cosine_similarity(query_emb, centroid)
        if sim >= sim_threshold:
            summary = db.get_person_summary(pid)
            if summary:
                summary["similarity"] = round(sim * 100.0, 1)
                matched_persons.append(summary)
    matched_persons.sort(key=lambda p: p["similarity"], reverse=True)

    # 2. Match against EVERY indexed detection across photos and videos (including timestamps!)
    all_dets = db.get_all_detections_with_embeddings()
    media_hits: dict[str, dict[str, Any]] = {}
    grouped_timestamps: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for d in all_dets:
        emb = d.pop("embedding_vec", None)
        if emb is None:
            continue
        sim = cosine_similarity(query_emb, emb)
        if sim < sim_threshold:
            continue

        mid = d["media_id"]
        sim_pct = round(sim * 100.0, 1)
        grouped_timestamps[mid].append(
            {
                "detection_id": d["id"],
                "timestamp_sec": d["timestamp_sec"],
                "frame_idx": d["frame_idx"],
                "similarity": sim_pct,
                "portrait_score": d["portrait_score"],
                "person_id": d.get("person_id"),
                "person_name": d.get("person_name"),
                "thumb_path": d.get("thumb_path"),
                "portrait_path": d.get("portrait_path"),
                "status": d.get("status"),
            }
        )
        if mid not in media_hits:
            media_hits[mid] = {
                "media_id": mid,
                "filename": d.get("media_filename"),
                "media_type": d.get("media_type"),
                "source_deleted": bool(d.get("source_deleted")),
                "duration_sec": d.get("duration_sec"),
                "best_similarity": sim_pct,
                "appearances_count": 0,
                "timestamps": [],
            }
        else:
            media_hits[mid]["best_similarity"] = max(media_hits[mid]["best_similarity"], sim_pct)

    matched_media: list[dict[str, Any]] = []
    for mid, item in media_hits.items():
        hits = sorted(grouped_timestamps[mid], key=lambda x: x["timestamp_sec"])
        item["appearances_count"] = len(hits)
        item["timestamps"] = hits
        matched_media.append(item)

    matched_media.sort(key=lambda m: m["best_similarity"], reverse=True)

    return {
        "found_faces": len(faces),
        "selected_face_index": chosen_idx,
        "query_faces": query_faces_meta,
        "matched_persons": matched_persons,
        "matched_media": matched_media,
        "message": (
            f"Encontrado en {len(matched_media)} archivo(s) y {len(matched_persons)} perfil(es)."
            if (matched_media or matched_persons)
            else "Esta persona no aparece en ninguna foto ni vídeo indexado todavía."
        ),
    }

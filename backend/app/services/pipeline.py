"""Core media ingestion pipeline: frame-by-frame video & photo analysis, clustering, quality selection, and smart discarding."""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..config import (
    FRAMES_DIR,
    IMAGE_EXTENSIONS,
    THUMBS_DIR,
    VIDEO_EXTENSIONS,
    ensure_data_dirs,
)
from ..database import Database, db as default_db
from .extractor import extract_all_assets, extract_thumbnail_only
from .ml_models import DetectedBody, DetectedFace, DetectedObject, model_manager
from .quality import QualityMetrics, score_detection
from .scene_analyzer import scene_analyzer

logger = logging.getLogger(__name__)

# Single-worker executor keeps CPU/GPU memory predictable and avoids SQLite/model contention
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ff-pipeline")
_cluster_lock = threading.RLock()


@dataclass
class CandidateRecord:
    det_id: str
    media_id: str
    person_id: str
    person_name: str
    timestamp_sec: float
    frame_idx: int
    face: DetectedFace
    metrics: QualityMetrics
    bodies: list[DetectedBody]
    objects: list[DetectedObject]
    image_bgr: np.ndarray
    shared_frame_name: str | None = None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)


def _make_media_thumb(image_bgr: np.ndarray, media_id: str) -> str:
    ensure_data_dirs()
    h, w = image_bgr.shape[:2]
    scale = min(1.0, 480.0 / max(1, max(h, w)))
    small = (
        cv2.resize(image_bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        if scale < 1.0
        else image_bgr
    )
    out_path = THUMBS_DIR / f"media_{media_id}.jpg"
    cv2.imwrite(str(out_path), small, [int(cv2.IMWRITE_JPEG_QUALITY), 84])
    return out_path.name


def match_or_create_person(
    database: Database,
    embedding: np.ndarray,
    threshold: float,
    in_memory_centroids: dict[str, tuple[str, np.ndarray, int]],
    exclude_person_ids: set[str] | None = None,
) -> tuple[str, str, bool]:
    """Match a face embedding against existing persons or create a new person cluster.

    Returns (person_id, person_name, is_new).
    """
    excluded = exclude_person_ids or set()
    # ArcFace 512-D cosine similarity: same person across different angles/lighting often sits at 0.38–0.55,
    # while different people sit < 0.30. Combined with Cannot-Link per frame, threshold-0.04 avoids over-splitting.
    eff_threshold = max(0.32, min(0.75, threshold - 0.04))

    with _cluster_lock:
        best_pid: str | None = None
        best_name: str = ""
        best_sim = -1.0

        for pid, (name, centroid, _count) in in_memory_centroids.items():
            if pid in excluded:
                continue
            sim = cosine_similarity(embedding, centroid)
            if sim > best_sim:
                best_sim = sim
                best_pid = pid
                best_name = name

        if best_pid is not None and best_sim >= eff_threshold:
            # Update running centroid smoothly
            name, old_c, count = in_memory_centroids[best_pid]
            new_c = old_c * count + embedding
            norm = float(np.linalg.norm(new_c))
            if norm > 0:
                new_c = new_c / norm
            in_memory_centroids[best_pid] = (name, new_c, count + 1)
            return best_pid, best_name, False

        # Create a brand new person
        num = database.next_person_number()
        new_name = f"Persona {num}"
        person = database.create_person(name=new_name, centroid=embedding)
        pid = person["id"]
        in_memory_centroids[pid] = (new_name, embedding.copy(), 1)
        return pid, new_name, True


def evaluate_keep_or_discard(
    database: Database,
    person_id: str,
    person_name: str,
    media_id: str,
    timestamp_sec: float,
    metrics: QualityMetrics,
    settings: dict[str, str],
) -> tuple[str, str | None, str | None]:
    """Decide whether a new detection should be KEPT (top quality) or DISCARDED (redundant / lower quality).

    Returns (status, keep_category, discard_reason).
    """
    min_quality = float(settings.get("min_quality_score", "32.0"))
    max_portraits = int(settings.get("max_portraits_per_person", "6"))
    max_bodies = int(settings.get("max_fullbody_per_person", "4"))
    min_gap = float(settings.get("min_video_gap_seconds", "1.5"))

    existing_kept = database.list_detections_for_person(person_id, status="kept")
    if len(existing_kept) == 0:
        return "kept", "portrait", None

    if metrics.portrait_score < min_quality and metrics.body_score < min_quality + 8.0:
        reasons = []
        if metrics.sharpness_score < 35:
            reasons.append("borrosa")
        if metrics.frontality_score < 40:
            reasons.append("muy de perfil")
        detail = f" ({', '.join(reasons)})" if reasons else ""
        return (
            "discarded",
            None,
            f"Calidad insuficiente ({metrics.overall_score:.0f}/100){detail}",
        )

    # Avoid near-duplicate consecutive video frames from the same video unless meaningfully better
    if min_gap > 0:
        for ex in existing_kept:
            if ex["media_id"] == media_id and abs(float(ex["timestamp_sec"]) - timestamp_sec) < min_gap:
                if (
                    metrics.portrait_score <= float(ex["portrait_score"]) + 4.0
                    and metrics.body_score <= float(ex["body_score"]) + 4.0
                ):
                    return (
                        "discarded",
                        None,
                        f"Frame muy cercano ({timestamp_sec:.1f}s) a una captura mejor de {person_name} ({float(ex['overall_score']):.0f} vs {metrics.overall_score:.0f})",
                    )

    # Check if it qualifies for top-K portraits
    portrait_kept = sorted(existing_kept, key=lambda d: float(d["portrait_score"]), reverse=True)
    qualifies_portrait = (
        len(portrait_kept) < max_portraits
        or metrics.portrait_score > float(portrait_kept[max_portraits - 1]["portrait_score"])
    )

    # Check if it qualifies for top-K full-body / cutout shots
    body_kept = sorted(
        [d for d in existing_kept if d["has_cutout"] or float(d["body_score"]) > 0],
        key=lambda d: float(d["body_score"]),
        reverse=True,
    )
    has_usable_body = metrics.matched_body is not None and metrics.body_score >= min_quality
    qualifies_body = has_usable_body and (
        len(body_kept) < max_bodies
        or metrics.body_score > float(body_kept[max_bodies - 1]["body_score"])
    )

    if qualifies_portrait and qualifies_body:
        return "kept", "both", None
    if qualifies_portrait:
        return "kept", "portrait", None
    if qualifies_body:
        return "kept", "full_body", None

    best_p = float(portrait_kept[0]["portrait_score"]) if portrait_kept else 0.0
    best_b = float(body_kept[0]["body_score"]) if body_kept else 0.0
    return (
        "discarded",
        None,
        f"Descartada: {person_name} ya tiene capturas mejores (retrato {metrics.portrait_score:.0f} vs {best_p:.0f}, cuerpo {metrics.body_score:.0f} vs {best_b:.0f})",
    )


def prune_person_excess_kept(database: Database, person_id: str, person_name: str, settings: dict[str, str]) -> None:
    """Keep the union of top-K portraits and top-K full-body shots; mark older lower-ranked surplus as discarded."""
    max_portraits = int(settings.get("max_portraits_per_person", "6"))
    max_bodies = int(settings.get("max_fullbody_per_person", "4"))

    kept = database.list_detections_for_person(person_id, status="kept")
    if len(kept) <= max(max_portraits, max_bodies):
        return

    top_portraits = {
        d["id"]
        for d in sorted(kept, key=lambda d: float(d["portrait_score"]), reverse=True)[:max_portraits]
    }
    top_bodies = {
        d["id"]
        for d in sorted(
            [d for d in kept if d["has_cutout"] or float(d["body_score"]) > 0],
            key=lambda d: float(d["body_score"]),
            reverse=True,
        )[:max_bodies]
    }
    keep_ids = top_portraits | top_bodies

    best_p = max((float(d["portrait_score"]) for d in kept), default=0.0)
    for d in kept:
        if d["id"] not in keep_ids:
            database.update_detection(
                d["id"],
                status="discarded",
                discard_reason=(
                    f"Sustituida por capturas más claras de {person_name} "
                    f"(retrato {float(d['portrait_score']):.0f} vs mejor {best_p:.0f})"
                ),
            )


def _process_single_frame(
    database: Database,
    media_id: str,
    image_bgr: np.ndarray,
    timestamp_sec: float,
    frame_idx: int,
    settings: dict[str, str],
    in_memory_centroids: dict[str, tuple[str, np.ndarray, int]],
    affected_persons: set[str],
    stats: dict[str, Any],
) -> tuple[int, int, int]:
    """Analyze one frame/photo, extract assets for kept detections, and persist to SQLite.

    Returns (total_faces, kept_faces, discarded_faces).
    """
    face_pack = settings.get("insightface_pack", "buffalo_l")
    yolo_model = settings.get("yolo_model", "yolov8n-seg.pt")
    threshold = float(settings.get("similarity_threshold", "0.45"))
    auto_scene = settings.get("auto_scene_analysis", "false").lower() == "true"
    scene_backend = settings.get("scene_backend", "heuristic")

    analysis = model_manager.analyze_frame(
        image_bgr,
        run_segmentation=True,
        face_pack=face_pack,
        yolo_model=yolo_model,
    )
    if not analysis.faces:
        return 0, 0, 0

    used_body_indices: set[int] = set()
    frame_assigned_persons: set[str] = set()
    kept_in_frame = 0
    discarded_in_frame = 0
    shared_frame_name: str | None = None

    # Sort faces by descending size so prominent foreground faces match bodies first
    sorted_faces = sorted(
        analysis.faces,
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True,
    )

    for face in sorted_faces:
        metrics = score_detection(
            image_bgr, face, analysis.bodies, used_body_indices=used_body_indices
        )
        if metrics.matched_body_idx is not None:
            used_body_indices.add(metrics.matched_body_idx)

        person_id, person_name, is_new = match_or_create_person(
            database,
            face.embedding,
            threshold,
            in_memory_centroids,
            exclude_person_ids=frame_assigned_persons,
        )
        frame_assigned_persons.add(person_id)
        affected_persons.add(person_id)
        if is_new:
            stats["new_persons"] += 1

        status, keep_category, discard_reason = evaluate_keep_or_discard(
            database,
            person_id=person_id,
            person_name=person_name,
            media_id=media_id,
            timestamp_sec=timestamp_sec,
            metrics=metrics,
            settings=settings,
        )

        det_id = uuid.uuid4().hex[:12]

        # Instant zero-cost heuristic context (always computed cleanly from YOLO COCO objects)
        scene_summary, scene_tags = scene_analyzer.analyze_heuristic(
            num_people=max(len(analysis.faces), len(analysis.bodies)),
            is_full_body=metrics.is_full_body,
            frontality_score=metrics.frontality_score,
            objects=analysis.objects,
        )
        used_scene_backend = "heuristic"

        if status == "kept":
            assets = extract_all_assets(
                image_bgr,
                det_id=det_id,
                face_bbox=face.bbox,
                matched_body=metrics.matched_body,
                all_bodies=analysis.bodies,
                shared_frame_filename=shared_frame_name,
            )
            if shared_frame_name is None and assets.frame_path:
                shared_frame_name = assets.frame_path

            # Optional local VLM captioning (only if user opted in via settings)
            if auto_scene and scene_backend in ("blip", "ollama") and assets.frame_path:
                scene_summary, scene_tags, used_scene_backend = scene_analyzer.analyze_image_file(
                    FRAMES_DIR / assets.frame_path,
                    backend=scene_backend,
                    ollama_url=settings.get("ollama_url", "http://localhost:11434"),
                    ollama_model=settings.get("ollama_model", "llava"),
                    fallback_context={
                        "objects": analysis.objects,
                        "num_people": max(len(analysis.faces), len(analysis.bodies)),
                        "is_full_body": metrics.is_full_body,
                        "frontality_score": metrics.frontality_score,
                    },
                )

            database.insert_detection(
                {
                    "id": det_id,
                    "media_id": media_id,
                    "person_id": person_id,
                    "timestamp_sec": round(timestamp_sec, 2),
                    "frame_idx": frame_idx,
                    "face_bbox": [round(v, 1) for v in face.bbox],
                    "body_bbox": [round(v, 1) for v in metrics.matched_body.bbox]
                    if metrics.matched_body
                    else None,
                    "det_score": round(face.det_score, 4),
                    "portrait_score": metrics.portrait_score,
                    "body_score": metrics.body_score,
                    "overall_score": metrics.overall_score,
                    "sharpness_score": metrics.sharpness_score,
                    "frontality_score": metrics.frontality_score,
                    "separation_score": metrics.separation_score,
                    "has_cutout": assets.has_cutout,
                    "is_full_body": metrics.is_full_body,
                    "embedding": face.embedding,
                    "frame_path": assets.frame_path,
                    "portrait_path": assets.portrait_path,
                    "cutout_path": assets.cutout_path,
                    "crop_path": assets.crop_path,
                    "thumb_path": assets.thumb_path,
                    "status": "kept",
                    "keep_category": keep_category,
                    "discard_reason": None,
                    "scene_summary": scene_summary,
                    "scene_tags": scene_tags,
                    "scene_backend": used_scene_backend,
                }
            )
            prune_person_excess_kept(database, person_id, person_name, settings)
            kept_in_frame += 1
        else:
            thumb_name = extract_thumbnail_only(image_bgr, det_id=det_id, face_bbox=face.bbox)
            database.insert_detection(
                {
                    "id": det_id,
                    "media_id": media_id,
                    "person_id": person_id,
                    "timestamp_sec": round(timestamp_sec, 2),
                    "frame_idx": frame_idx,
                    "face_bbox": [round(v, 1) for v in face.bbox],
                    "body_bbox": [round(v, 1) for v in metrics.matched_body.bbox]
                    if metrics.matched_body
                    else None,
                    "det_score": round(face.det_score, 4),
                    "portrait_score": metrics.portrait_score,
                    "body_score": metrics.body_score,
                    "overall_score": metrics.overall_score,
                    "sharpness_score": metrics.sharpness_score,
                    "frontality_score": metrics.frontality_score,
                    "separation_score": metrics.separation_score,
                    "has_cutout": False,
                    "is_full_body": metrics.is_full_body,
                    "embedding": face.embedding,
                    "frame_path": None,
                    "portrait_path": None,
                    "cutout_path": None,
                    "crop_path": None,
                    "thumb_path": thumb_name,
                    "status": "discarded",
                    "keep_category": None,
                    "discard_reason": discard_reason,
                    "scene_summary": scene_summary,
                    "scene_tags": scene_tags,
                    "scene_backend": used_scene_backend,
                }
            )
            discarded_in_frame += 1

    return len(sorted_faces), kept_in_frame, discarded_in_frame


def process_media_batch_sync(
    media_ids: list[str],
    job_id: str,
    *,
    database: Database | None = None,
    sample_fps_override: float | None = None,
) -> dict[str, Any]:
    """Process a batch of uploaded photos and/or videos synchronously (invoked inside worker thread)."""
    db = database or default_db
    settings = db.get_settings()
    sample_fps = sample_fps_override or float(settings.get("video_sample_fps", "1.0"))

    db.update_job(
        job_id,
        status="running",
        progress=0.02,
        total_items=len(media_ids),
        processed_items=0,
        current_step="Cargando modelos locales de detección y segmentación...",
    )

    # Seed in-memory person centroids from DB for fast consistent clustering across the batch
    in_memory_centroids: dict[str, tuple[str, np.ndarray, int]] = {}
    for pid, name, emb in db.get_all_person_embeddings():
        in_memory_centroids[pid] = (name, emb, 3)
    preexisting_person_ids = set(in_memory_centroids.keys())

    affected_persons: set[str] = set()
    stats: dict[str, Any] = {
        "total_media": len(media_ids),
        "useful_media": 0,
        "discarded_media": 0,
        "no_people_media": 0,
        "total_frames_analyzed": 0,
        "total_detections": 0,
        "kept_detections": 0,
        "discarded_detections": 0,
        "new_persons": 0,
        "media_results": [],
    }

    try:
        for idx, media_id in enumerate(media_ids):
            media = db.get_media(media_id)
            if not media:
                continue

            file_path = Path(media["file_path"])
            filename = media["original_filename"]
            base_progress = idx / max(1, len(media_ids))
            step_span = 1.0 / max(1, len(media_ids))

            db.update_job(
                job_id,
                progress=base_progress,
                processed_items=idx,
                current_step=f"Analizando {filename} ({idx + 1}/{len(media_ids)})...",
            )

            total_faces = 0
            kept_faces = 0
            discarded_faces = 0
            frames_analyzed = 0

            if media["media_type"] == "image":
                img = cv2.imread(str(file_path))
                if img is None:
                    db.update_media(
                        media_id,
                        status="error",
                        verdict="discarded",
                        verdict_reason="No se pudo leer la imagen",
                    )
                    stats["discarded_media"] += 1
                    continue

                h, w = img.shape[:2]
                thumb_name = _make_media_thumb(img, media_id)
                f_total, f_kept, f_disc = _process_single_frame(
                    db,
                    media_id,
                    img,
                    timestamp_sec=0.0,
                    frame_idx=0,
                    settings=settings,
                    in_memory_centroids=in_memory_centroids,
                    affected_persons=affected_persons,
                    stats=stats,
                )
                total_faces += f_total
                kept_faces += f_kept
                discarded_faces += f_disc
                frames_analyzed = 1

                if total_faces == 0:
                    verdict = "no_people"
                    verdict_reason = "No se detectaron rostros claros en esta foto"
                    stats["no_people_media"] += 1
                elif kept_faces > 0:
                    verdict = "useful"
                    verdict_reason = (
                        f"Útil: {kept_faces} captura(s) de alta calidad extraída(s)"
                        + (f" ({discarded_faces} descartada(s))" if discarded_faces else "")
                    )
                    stats["useful_media"] += 1
                else:
                    verdict = "discarded"
                    verdict_reason = (
                        f"Descartada: las {discarded_faces} persona(s) detectadas ya tenían capturas de mejor calidad"
                    )
                    stats["discarded_media"] += 1

                db.update_media(
                    media_id,
                    width=w,
                    height=h,
                    thumb_path=thumb_name,
                    frames_analyzed=frames_analyzed,
                    detections_count=total_faces,
                    kept_count=kept_faces,
                    discarded_count=discarded_faces,
                    status="done",
                    verdict=verdict,
                    verdict_reason=verdict_reason,
                )

            else:  # video
                cap = cv2.VideoCapture(str(file_path))
                if not cap.isOpened():
                    db.update_media(
                        media_id,
                        status="error",
                        verdict="discarded",
                        verdict_reason="No se pudo abrir el archivo de vídeo",
                    )
                    stats["discarded_media"] += 1
                    continue

                vid_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
                if vid_fps <= 0.1 or vid_fps > 240:
                    vid_fps = 25.0
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                duration_sec = (frame_count / vid_fps) if frame_count > 0 else 0.0

                stride = max(1, int(round(vid_fps / max(0.2, sample_fps))))
                thumb_name: str | None = None
                cur_frame_idx = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if thumb_name is None:
                        thumb_name = _make_media_thumb(frame, media_id)
                        if width == 0 or height == 0:
                            height, width = frame.shape[:2]

                    if cur_frame_idx % stride == 0:
                        ts = cur_frame_idx / vid_fps
                        frames_analyzed += 1
                        f_total, f_kept, f_disc = _process_single_frame(
                            db,
                            media_id,
                            frame,
                            timestamp_sec=ts,
                            frame_idx=cur_frame_idx,
                            settings=settings,
                            in_memory_centroids=in_memory_centroids,
                            affected_persons=affected_persons,
                            stats=stats,
                        )
                        total_faces += f_total
                        kept_faces += f_kept
                        discarded_faces += f_disc

                        if frame_count > 0:
                            frac = min(1.0, cur_frame_idx / float(frame_count))
                            db.update_job(
                                job_id,
                                progress=base_progress + step_span * frac,
                                current_step=(
                                    f"{filename}: {ts:.1f}s / {duration_sec:.1f}s "
                                    f"({kept_faces} mejores, {total_faces} detecciones)"
                                ),
                            )
                    cur_frame_idx += 1

                cap.release()
                if duration_sec == 0.0 and cur_frame_idx > 0:
                    duration_sec = cur_frame_idx / vid_fps

                if total_faces == 0:
                    verdict = "no_people"
                    verdict_reason = f"No se detectaron personas en {frames_analyzed} frames analizados"
                    stats["no_people_media"] += 1
                elif kept_faces > 0:
                    verdict = "useful"
                    verdict_reason = (
                        f"Útil: {kept_faces} mejores frames extraídos de {total_faces} detecciones "
                        f"({frames_analyzed} frames analizados)"
                    )
                    stats["useful_media"] += 1
                else:
                    verdict = "discarded"
                    verdict_reason = (
                        f"Indexado ({total_faces} apariciones guardadas para búsqueda), "
                        "pero ya existían capturas de mayor calidad"
                    )
                    stats["discarded_media"] += 1

                db.update_media(
                    media_id,
                    width=width,
                    height=height,
                    duration_sec=round(duration_sec, 2),
                    fps=round(vid_fps, 2),
                    thumb_path=thumb_name,
                    frames_analyzed=frames_analyzed,
                    detections_count=total_faces,
                    kept_count=kept_faces,
                    discarded_count=discarded_faces,
                    status="done",
                    verdict=verdict,
                    verdict_reason=verdict_reason,
                )

            stats["total_frames_analyzed"] += frames_analyzed
            stats["total_detections"] += total_faces
            stats["kept_detections"] += kept_faces
            stats["discarded_detections"] += discarded_faces
            stats["media_results"].append(
                {
                    "media_id": media_id,
                    "filename": filename,
                    "media_type": media["media_type"],
                    "verdict": verdict,
                    "verdict_reason": verdict_reason,
                    "detections": total_faces,
                    "kept": kept_faces,
                    "discarded": discarded_faces,
                }
            )

        # Recompute final centroids & best portrait/body pointers for all affected persons
        detected_persons = []
        for pid in sorted(affected_persons):
            db.recompute_person_stats(pid)
            p = db.get_person_summary(pid)
            if p:
                cover = p.get("cover_detection") or p.get("best_portrait") or {}
                detected_persons.append(
                    {
                        "id": p["id"],
                        "name": p["name"],
                        "is_new": pid not in preexisting_person_ids,
                        "thumb_path": cover.get("portrait_path") or cover.get("thumb_path"),
                        "captures_in_batch": p.get("total_detections", 1),
                    }
                )

        stats["affected_person_ids"] = sorted(affected_persons)
        stats["detected_persons"] = detected_persons
        db.update_job(
            job_id,
            status="done",
            progress=1.0,
            processed_items=len(media_ids),
            current_step="Análisis completado",
            summary=stats,
        )
        return stats

    except Exception as exc:
        logger.exception("Error processing media batch for job %s", job_id)
        db.update_job(
            job_id,
            status="error",
            error=str(exc),
            current_step=f"Error: {exc}",
        )
        raise


def submit_media_batch(
    media_ids: list[str],
    *,
    sample_fps_override: float | None = None,
    job_type: str = "ingest",
) -> str:
    """Enqueue background processing for uploaded media files and return job_id."""
    job_id = default_db.create_job(
        job_type=job_type,
        total_items=len(media_ids),
        current_step="En cola para análisis...",
    )
    _executor.submit(
        process_media_batch_sync,
        media_ids,
        job_id,
        database=default_db,
        sample_fps_override=sample_fps_override,
    )
    return job_id


def regenerate_detection_assets(det_id: str, database: Database | None = None) -> dict[str, Any] | None:
    """Generate full-res portrait, transparent PNG cutout, crop, and frame for a detection (e.g. when rescuing a discarded capture)."""
    db = database or default_db
    det = db.get_detection(det_id)
    if not det:
        return None
    if det.get("portrait_path") and det.get("frame_path"):
        return det

    media = db.get_media(det["media_id"])
    if not media or media.get("source_deleted"):
        return det

    file_path = Path(media["file_path"])
    if not file_path.exists():
        return det

    frame_bgr: np.ndarray | None = None
    if media["media_type"] == "image":
        frame_bgr = cv2.imread(str(file_path))
    else:
        cap = cv2.VideoCapture(str(file_path))
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(det.get("frame_idx", 0)))
            ok, frame = cap.read()
            if ok:
                frame_bgr = frame
            cap.release()

    if frame_bgr is None:
        return det

    face_bbox = det.get("face_bbox")
    if not face_bbox:
        return det

    bodies, _objects = model_manager.detect_bodies_and_objects(frame_bgr)
    # Create a dummy face to reuse body matching
    dummy_face = DetectedFace(
        bbox=[float(v) for v in face_bbox],
        det_score=float(det.get("det_score", 0.9)),
        embedding=np.zeros(512, dtype=np.float32),
    )
    metrics = score_detection(frame_bgr, dummy_face, bodies)
    assets = extract_all_assets(
        frame_bgr,
        det_id=det_id,
        face_bbox=dummy_face.bbox,
        matched_body=metrics.matched_body,
        all_bodies=bodies,
    )
    updated = db.update_detection(
        det_id,
        frame_path=assets.frame_path,
        portrait_path=assets.portrait_path,
        cutout_path=assets.cutout_path,
        crop_path=assets.crop_path,
        thumb_path=assets.thumb_path,
        has_cutout=1 if assets.has_cutout else 0,
        body_score=metrics.body_score,
        is_full_body=1 if metrics.is_full_body else 0,
    )
    if det.get("person_id"):
        db.recompute_person_stats(det["person_id"])
    return updated


def classify_media_type(filename: str) -> str | None:
    ext = Path(filename).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    return None


def recluster_all_persons(database: Database | None = None) -> dict[str, int]:
    """Global quality-first 512-D ArcFace re-clustering with Cannot-Link constraints & hierarchical post-merge."""
    import re

    db = database or default_db
    settings = db.get_settings()
    base_thresh = float(settings.get("similarity_threshold", "0.45"))
    # Calibrated ArcFace cosine threshold (0.41 at default 0.45): unites real matches across angles/lighting
    # while Cannot-Link per frame + runner-up margin prevent merging different people.
    clean_sim_thresh = max(0.32, min(0.72, base_thresh - 0.04))
    lowq_sim_thresh = max(0.35, min(0.75, base_thresh - 0.01))

    existing_persons = db.list_persons(include_ignored=False)
    all_dets = db.get_all_detections_with_embeddings()
    persons_before = len(existing_persons)
    if not all_dets:
        return {"persons_before": persons_before, "persons_after": persons_before, "reassigned_count": 0}

    det_by_id = {d["id"]: d for d in all_dets}
    clusters: list[dict[str, Any]] = []
    assigned: set[str] = set()

    def is_custom_name(name: str) -> bool:
        return not re.match(r"^Persona \d+$", name.strip(), re.IGNORECASE)

    def frame_key_of(d: dict[str, Any]) -> str | None:
        if d.get("face_bbox"):
            return f"{d['media_id']}:{d.get('frame_idx', 0)}"
        return None

    def compute_norm_mean(vecs: list[np.ndarray]) -> np.ndarray:
        mean_vec = np.mean(vecs, axis=0)
        norm = float(np.linalg.norm(mean_vec))
        return mean_vec / norm if norm > 0 else mean_vec

    # 1. Anchor every user-named person (and any photo-less saved identity with a centroid)
    centroid_by_pid = {pid: emb for pid, _n, emb in db.get_all_person_embeddings()}
    for p in existing_persons:
        mine = sorted(
            [d for d in all_dets if d.get("person_id") == p["id"] and d.get("embedding_vec") is not None],
            key=lambda d: float(d["portrait_score"]) + 0.35 * float(d["frontality_score"]),
            reverse=True,
        )
        is_photoless_saved = len(mine) == 0 and p["id"] in centroid_by_pid
        if not is_custom_name(p["name"]) and not is_photoless_saved:
            continue
        cover_id = p.get("cover_detection_id")
        anchor_det = det_by_id.get(cover_id) if cover_id else None
        if anchor_det is None or anchor_det.get("embedding_vec") is None:
            anchor_det = mine[0] if mine else None

        if anchor_det is not None and anchor_det.get("embedding_vec") is not None:
            emb = anchor_det["embedding_vec"]
            fkeys: set[str] = set()
            fk0 = frame_key_of(anchor_det)
            if fk0:
                fkeys.add(fk0)
            anchors = [emb.copy()]
            det_ids = [anchor_det["id"]]
            assigned.add(anchor_det["id"])

            # Preserve compatible detections already assigned to this named person (keeps manual merges intact
            # while still letting clearly unrelated faces split out cleanly)
            for extra in mine:
                if extra["id"] == anchor_det["id"]:
                    continue
                efk = frame_key_of(extra)
                if efk and efk in fkeys:
                    continue
                if cosine_similarity(extra["embedding_vec"], emb) >= max(0.28, clean_sim_thresh - 0.06):
                    det_ids.append(extra["id"])
                    assigned.add(extra["id"])
                    if efk:
                        fkeys.add(efk)
                    if len(anchors) < 8:
                        anchors.append(extra["embedding_vec"].copy())

            clusters.append(
                {
                    "id": p["id"],
                    "name": p["name"],
                    "is_named": True,
                    "centroid": compute_norm_mean(anchors),
                    "anchors": anchors,
                    "det_ids": det_ids,
                    "frame_keys": fkeys,
                    "existing": True,
                }
            )
        elif p["id"] in centroid_by_pid:
            emb = centroid_by_pid[p["id"]]
            clusters.append(
                {
                    "id": p["id"],
                    "name": p["name"],
                    "is_named": True,
                    "centroid": emb.copy(),
                    "anchors": [emb.copy()],
                    "det_ids": [],
                    "frame_keys": set(),
                    "existing": True,
                }
            )

    # 2. Sort remaining detections by quality + frontality DESC so crisp shots form pure anchors first
    remaining = sorted(
        [d for d in all_dets if d["id"] not in assigned and d.get("embedding_vec") is not None],
        key=lambda d: float(d["portrait_score"]) + 0.35 * float(d["frontality_score"]),
        reverse=True,
    )

    auto_idx = 1

    def next_name() -> str:
        nonlocal auto_idx
        used = {c["name"] for c in clusters}
        while f"Persona {auto_idx}" in used:
            auto_idx += 1
        res = f"Persona {auto_idx}"
        auto_idx += 1
        return res

    for d in remaining:
        emb = d["embedding_vec"]
        fk = frame_key_of(d)
        is_low_q = float(d["portrait_score"]) < 32.0 or float(d["frontality_score"]) < 30.0
        is_clean_anchor = float(d["portrait_score"]) >= 42.0 and float(d["frontality_score"]) >= 38.0
        req_sim = lowq_sim_thresh if is_low_q else clean_sim_thresh

        best_c = None
        best_score = -1.0
        best_ex_max = -1.0
        second_best_score = -1.0

        for c in clusters:
            # Cannot-Link: two faces in the same photo/frame cannot belong to the same person
            if fk and fk in c["frame_keys"]:
                continue

            c_sim = cosine_similarity(emb, c["centroid"])
            a_sims = sorted((cosine_similarity(emb, a) for a in c["anchors"]), reverse=True)
            best_ex = a_sims[0]
            top_k = a_sims[: min(3, len(a_sims))]
            avg_top_sim = sum(top_k) / len(top_k)
            score = 0.45 * c_sim + 0.35 * avg_top_sim + 0.20 * best_ex

            direct_match = c_sim >= req_sim - 0.03 and avg_top_sim >= req_sim - 0.02
            strong_ex_match = best_ex >= req_sim + 0.05 and c_sim >= req_sim - 0.06

            if direct_match or strong_ex_match:
                if score > best_score:
                    second_best_score = best_score
                    best_score = score
                    best_ex_max = best_ex
                    best_c = c
                elif score > second_best_score:
                    second_best_score = score

        is_borderline = best_score < req_sim + 0.03
        is_ambiguous = (
            is_borderline
            and second_best_score > 0
            and (best_score - second_best_score) < 0.03
            and best_ex_max < req_sim + 0.04
        )

        if best_c is not None and best_score >= req_sim - 0.015 and not is_ambiguous:
            best_c["det_ids"].append(d["id"])
            if fk:
                best_c["frame_keys"].add(fk)
            if is_clean_anchor and len(best_c["anchors"]) < 10:
                best_c["anchors"].append(emb.copy())
                best_c["centroid"] = compute_norm_mean(best_c["anchors"])
        else:
            fkeys_new: set[str] = set()
            if fk:
                fkeys_new.add(fk)
            clusters.append(
                {
                    "id": None,
                    "name": next_name(),
                    "is_named": False,
                    "centroid": emb.copy(),
                    "anchors": [emb.copy()],
                    "det_ids": [d["id"]],
                    "frame_keys": fkeys_new,
                    "existing": False,
                }
            )

    # 3. Post-clustering hierarchical cluster-to-cluster merge pass (reunites fragments via denoised centroids)
    merged_any = True
    while merged_any:
        merged_any = False
        best_pair: tuple[int, int] | None = None
        best_pair_sim = -1.0

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                a = clusters[i]
                b = clusters[j]
                if a["is_named"] and b["is_named"]:
                    continue
                if a["frame_keys"] and b["frame_keys"] and not a["frame_keys"].isdisjoint(b["frame_keys"]):
                    continue

                c_sim = cosine_similarity(a["centroid"], b["centroid"])
                if c_sim < clean_sim_thresh - 0.01:
                    continue

                pair_sims = [
                    cosine_similarity(ea, eb) for ea in a["anchors"] for eb in b["anchors"]
                ]
                max_link = max(pair_sims) if pair_sims else c_sim
                avg_link = sum(pair_sims) / len(pair_sims) if pair_sims else c_sim

                if (
                    c_sim >= clean_sim_thresh
                    and (avg_link >= clean_sim_thresh - 0.04 or max_link >= clean_sim_thresh + 0.05)
                    and c_sim > best_pair_sim
                ):
                    best_pair_sim = c_sim
                    best_pair = (i, j)

        if best_pair is not None:
            i, j = best_pair
            if clusters[j]["is_named"] or (
                not clusters[i]["is_named"] and len(clusters[j]["det_ids"]) > len(clusters[i]["det_ids"])
            ):
                keep_idx, drop_idx = j, i
            else:
                keep_idx, drop_idx = i, j
            keep = clusters[keep_idx]
            drop = clusters[drop_idx]
            keep["det_ids"].extend(drop["det_ids"])
            keep["frame_keys"].update(drop["frame_keys"])
            for anc in drop["anchors"]:
                if len(keep["anchors"]) < 10:
                    keep["anchors"].append(anc)
            keep["centroid"] = compute_norm_mean(keep["anchors"])
            clusters.pop(drop_idx)
            merged_any = True

    # 4. Persist clusters cleanly & remove obsolete unnamed persons
    reassigned_count = 0
    active_ids = set()
    for c in clusters:
        if c["id"] is None:
            created = db.create_person(name=c["name"], centroid=c["centroid"])
            c["id"] = created["id"]
        active_ids.add(c["id"])
        cluster_dets = sorted(
            [det_by_id[did] for did in c["det_ids"] if did in det_by_id],
            key=lambda x: float(x["overall_score"]),
            reverse=True,
        )
        has_kept = any(x["status"] == "kept" for x in cluster_dets)
        for i, d in enumerate(cluster_dets):
            if d.get("person_id") != c["id"]:
                reassigned_count += 1
            updates: dict[str, Any] = {"person_id": c["id"]}
            if not has_kept and i == 0:
                updates["status"] = "kept"
                updates["discard_reason"] = None
            db.update_detection(d["id"], **updates)
        db.recompute_person_stats(c["id"])

    # Remove obsolete empty person records
    for p in existing_persons:
        if p["id"] not in active_ids:
            db.delete_person(p["id"])

    return {
        "persons_before": persons_before,
        "persons_after": len(clusters),
        "reassigned_count": reassigned_count,
    }

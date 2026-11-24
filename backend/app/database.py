"""SQLite database initialization and repository layer for FriendFinder."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

import numpy as np

from .config import DB_PATH, DEFAULT_SETTINGS, ensure_data_dirs


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def encode_embedding(vec: np.ndarray | None) -> bytes | None:
    if vec is None:
        return None
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr = arr / norm
    return arr.tobytes()


def decode_embedding(blob: bytes | None) -> np.ndarray | None:
    if not blob:
        return None
    return np.frombuffer(blob, dtype=np.float32).copy()


class Database:
    """Thread-safe SQLite repository for media, persons, detections, and jobs."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or DB_PATH
        self._lock = threading.RLock()
        ensure_data_dirs()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        with self._lock:
            conn = sqlite3.connect(str(self.db_path), timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA journal_mode = WAL;")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _init_schema(self) -> None:
        with self.connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS media (
                    id TEXT PRIMARY KEY,
                    original_filename TEXT NOT NULL,
                    media_type TEXT NOT NULL CHECK (media_type IN ('image', 'video')),
                    file_path TEXT NOT NULL,
                    thumb_path TEXT,
                    file_size INTEGER NOT NULL DEFAULT 0,
                    width INTEGER NOT NULL DEFAULT 0,
                    height INTEGER NOT NULL DEFAULT 0,
                    duration_sec REAL,
                    fps REAL,
                    frames_analyzed INTEGER NOT NULL DEFAULT 0,
                    detections_count INTEGER NOT NULL DEFAULT 0,
                    kept_count INTEGER NOT NULL DEFAULT 0,
                    discarded_count INTEGER NOT NULL DEFAULT 0,
                    source_deleted INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'pending',
                    verdict TEXT NOT NULL DEFAULT 'pending',
                    verdict_reason TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS persons (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    notes TEXT DEFAULT '',
                    centroid BLOB,
                    cover_detection_id TEXT,
                    best_portrait_id TEXT,
                    best_body_id TEXT,
                    is_ignored INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS detections (
                    id TEXT PRIMARY KEY,
                    media_id TEXT NOT NULL,
                    person_id TEXT,
                    timestamp_sec REAL NOT NULL DEFAULT 0.0,
                    frame_idx INTEGER NOT NULL DEFAULT 0,
                    face_bbox TEXT,
                    body_bbox TEXT,
                    det_score REAL NOT NULL DEFAULT 0.0,
                    portrait_score REAL NOT NULL DEFAULT 0.0,
                    body_score REAL NOT NULL DEFAULT 0.0,
                    overall_score REAL NOT NULL DEFAULT 0.0,
                    sharpness_score REAL NOT NULL DEFAULT 0.0,
                    frontality_score REAL NOT NULL DEFAULT 0.0,
                    separation_score REAL NOT NULL DEFAULT 0.0,
                    has_cutout INTEGER NOT NULL DEFAULT 0,
                    is_full_body INTEGER NOT NULL DEFAULT 0,
                    embedding BLOB,
                    frame_path TEXT,
                    portrait_path TEXT,
                    cutout_path TEXT,
                    crop_path TEXT,
                    thumb_path TEXT,
                    status TEXT NOT NULL DEFAULT 'kept' CHECK (status IN ('kept', 'discarded')),
                    keep_category TEXT,
                    discard_reason TEXT,
                    scene_summary TEXT,
                    scene_tags TEXT,
                    scene_backend TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (media_id) REFERENCES media(id) ON DELETE CASCADE,
                    FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE SET NULL
                );

                CREATE INDEX IF NOT EXISTS idx_detections_person ON detections(person_id);
                CREATE INDEX IF NOT EXISTS idx_detections_media ON detections(media_id);
                CREATE INDEX IF NOT EXISTS idx_detections_status ON detections(status);

                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued',
                    progress REAL NOT NULL DEFAULT 0.0,
                    total_items INTEGER NOT NULL DEFAULT 0,
                    processed_items INTEGER NOT NULL DEFAULT 0,
                    current_step TEXT DEFAULT '',
                    summary_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            for key, value in DEFAULT_SETTINGS.items():
                conn.execute(
                    "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
                    (key, value),
                )

    # ---------- Settings ----------
    def get_settings(self) -> dict[str, str]:
        with self.connection() as conn:
            rows = conn.execute("SELECT key, value FROM settings").fetchall()
            merged = dict(DEFAULT_SETTINGS)
            for r in rows:
                merged[r["key"]] = r["value"]
            return merged

    def update_settings(self, updates: dict[str, Any]) -> dict[str, str]:
        with self.connection() as conn:
            for key, value in updates.items():
                if key in DEFAULT_SETTINGS and value is not None:
                    str_val = (
                        "true"
                        if value is True
                        else "false"
                        if value is False
                        else str(value)
                    )
                    conn.execute(
                        "INSERT INTO settings (key, value) VALUES (?, ?) "
                        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                        (key, str_val),
                    )
        return self.get_settings()

    # ---------- Jobs ----------
    def create_job(self, job_type: str, total_items: int = 0, current_step: str = "") -> str:
        job_id = uuid.uuid4().hex[:12]
        now = utc_now_iso()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO jobs (id, job_type, status, progress, total_items, processed_items, current_step, created_at, updated_at)
                VALUES (?, ?, 'queued', 0.0, ?, 0, ?, ?, ?)
                """,
                (job_id, job_type, total_items, current_step, now, now),
            )
        return job_id

    def update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        progress: float | None = None,
        total_items: int | None = None,
        processed_items: int | None = None,
        current_step: str | None = None,
        summary: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        fields: list[str] = ["updated_at = ?"]
        values: list[Any] = [utc_now_iso()]
        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if progress is not None:
            fields.append("progress = ?")
            values.append(float(max(0.0, min(1.0, progress))))
        if total_items is not None:
            fields.append("total_items = ?")
            values.append(int(total_items))
        if processed_items is not None:
            fields.append("processed_items = ?")
            values.append(int(processed_items))
        if current_step is not None:
            fields.append("current_step = ?")
            values.append(current_step)
        if summary is not None:
            fields.append("summary_json = ?")
            values.append(json.dumps(summary, ensure_ascii=False))
        if error is not None:
            fields.append("error = ?")
            values.append(error)
        values.append(job_id)
        with self.connection() as conn:
            conn.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?", values)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            return self._job_row_to_dict(row) if row else None

    def list_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [self._job_row_to_dict(r) for r in rows]

    @staticmethod
    def _job_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        raw_summary = d.pop("summary_json", None)
        d["summary"] = json.loads(raw_summary) if raw_summary else None
        return d

    # ---------- Media ----------
    def create_media(
        self,
        *,
        media_id: str,
        original_filename: str,
        media_type: str,
        file_path: str,
        file_size: int,
    ) -> dict[str, Any]:
        now = utc_now_iso()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO media (id, original_filename, media_type, file_path, file_size, status, verdict, created_at)
                VALUES (?, ?, ?, ?, ?, 'processing', 'pending', ?)
                """,
                (media_id, original_filename, media_type, file_path, file_size, now),
            )
            row = conn.execute("SELECT * FROM media WHERE id = ?", (media_id,)).fetchone()
            return dict(row)

    def update_media(self, media_id: str, **kwargs: Any) -> None:
        if not kwargs:
            return
        fields = [f"{k} = ?" for k in kwargs]
        values = list(kwargs.values()) + [media_id]
        with self.connection() as conn:
            conn.execute(f"UPDATE media SET {', '.join(fields)} WHERE id = ?", values)

    def get_media(self, media_id: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM media WHERE id = ?", (media_id,)).fetchone()
            return dict(row) if row else None

    def list_media(self, media_type: str | None = None) -> list[dict[str, Any]]:
        with self.connection() as conn:
            if media_type in ("image", "video"):
                rows = conn.execute(
                    "SELECT * FROM media WHERE media_type = ? ORDER BY created_at DESC",
                    (media_type,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM media ORDER BY created_at DESC"
                ).fetchall()
            result = []
            for r in rows:
                item = dict(r)
                persons = conn.execute(
                    """
                    SELECT DISTINCT p.id, p.name
                    FROM detections d
                    JOIN persons p ON d.person_id = p.id
                    WHERE d.media_id = ?
                    ORDER BY p.name
                    """,
                    (item["id"],),
                ).fetchall()
                item["persons"] = [dict(p) for p in persons]
                result.append(item)
            return result

    def mark_media_source_deleted(self, media_id: str) -> None:
        with self.connection() as conn:
            conn.execute(
                "UPDATE media SET source_deleted = 1 WHERE id = ?", (media_id,)
            )

    def delete_media_completely(self, media_id: str) -> None:
        with self.connection() as conn:
            conn.execute("DELETE FROM media WHERE id = ?", (media_id,))

    # ---------- Persons ----------
    def next_person_number(self) -> int:
        with self.connection() as conn:
            row = conn.execute("SELECT COUNT(*) as c FROM persons").fetchone()
            return int(row["c"]) + 1

    def create_person(self, name: str, centroid: np.ndarray | None = None) -> dict[str, Any]:
        person_id = uuid.uuid4().hex[:12]
        now = utc_now_iso()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO persons (id, name, centroid, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (person_id, name, encode_embedding(centroid), now, now),
            )
            row = conn.execute("SELECT * FROM persons WHERE id = ?", (person_id,)).fetchone()
            return dict(row)

    def get_all_person_embeddings(self) -> list[tuple[str, str, np.ndarray]]:
        """Return list of (person_id, name, normalized_centroid)."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT id, name, centroid FROM persons WHERE centroid IS NOT NULL"
            ).fetchall()
            out: list[tuple[str, str, np.ndarray]] = []
            for r in rows:
                emb = decode_embedding(r["centroid"])
                if emb is not None:
                    out.append((r["id"], r["name"], emb))
            return out

    def update_person_metadata(
        self,
        person_id: str,
        *,
        name: str | None = None,
        notes: str | None = None,
        is_ignored: bool | None = None,
        cover_detection_id: str | None = None,
    ) -> dict[str, Any] | None:
        fields = ["updated_at = ?"]
        values: list[Any] = [utc_now_iso()]
        if name is not None:
            fields.append("name = ?")
            values.append(name.strip())
        if notes is not None:
            fields.append("notes = ?")
            values.append(notes)
        if is_ignored is not None:
            fields.append("is_ignored = ?")
            values.append(1 if is_ignored else 0)
        if cover_detection_id is not None:
            fields.append("cover_detection_id = ?")
            values.append(cover_detection_id)
        values.append(person_id)
        with self.connection() as conn:
            conn.execute(f"UPDATE persons SET {', '.join(fields)} WHERE id = ?", values)
            row = conn.execute("SELECT * FROM persons WHERE id = ?", (person_id,)).fetchone()
            return dict(row) if row else None

    def recompute_person_stats(self, person_id: str) -> None:
        """Recalculate centroid, best portrait, best body, and cover detection for a person."""
        with self.connection() as conn:
            dets = conn.execute(
                """
                SELECT id, embedding, portrait_score, body_score, overall_score, status, has_cutout
                FROM detections
                WHERE person_id = ?
                ORDER BY overall_score DESC
                """,
                (person_id,),
            ).fetchall()
            if not dets:
                return

            # Compute weighted centroid from highest quality detections (up to top 20)
            weighted_vecs = []
            for d in dets[:20]:
                emb = decode_embedding(d["embedding"])
                if emb is not None:
                    weight = max(0.1, float(d["portrait_score"]) / 100.0)
                    weighted_vecs.append(emb * weight)
            centroid_blob = None
            if weighted_vecs:
                centroid = np.sum(weighted_vecs, axis=0)
                centroid_blob = encode_embedding(centroid)

            kept_dets = [d for d in dets if d["status"] == "kept"]
            pool = kept_dets if kept_dets else dets

            best_portrait = max(pool, key=lambda d: float(d["portrait_score"]))["id"]
            body_candidates = [d for d in pool if d["has_cutout"] or float(d["body_score"]) > 0]
            best_body = (
                max(body_candidates, key=lambda d: float(d["body_score"]))["id"]
                if body_candidates
                else None
            )

            existing = conn.execute(
                "SELECT cover_detection_id FROM persons WHERE id = ?", (person_id,)
            ).fetchone()
            existing_cover = existing["cover_detection_id"] if existing else None
            valid_ids = {d["id"] for d in dets}
            cover_id = existing_cover if existing_cover in valid_ids else best_portrait

            conn.execute(
                """
                UPDATE persons
                SET centroid = ?, best_portrait_id = ?, best_body_id = ?, cover_detection_id = ?, updated_at = ?
                WHERE id = ?
                """,
                (centroid_blob, best_portrait, best_body, cover_id, utc_now_iso(), person_id),
            )

    def merge_persons(self, target_person_id: str, source_person_ids: list[str]) -> dict[str, Any] | None:
        source_ids = [sid for sid in source_person_ids if sid != target_person_id]
        if not source_ids:
            return self.get_person_summary(target_person_id)
        with self.connection() as conn:
            for sid in source_ids:
                conn.execute(
                    "UPDATE detections SET person_id = ? WHERE person_id = ?",
                    (target_person_id, sid),
                )
                conn.execute("DELETE FROM persons WHERE id = ?", (sid,))
        self.recompute_person_stats(target_person_id)
        return self.get_person_summary(target_person_id)

    def delete_person(self, person_id: str) -> None:
        with self.connection() as conn:
            conn.execute("DELETE FROM detections WHERE person_id = ?", (person_id,))
            conn.execute("DELETE FROM persons WHERE id = ?", (person_id,))

    def list_persons(self, include_ignored: bool = False) -> list[dict[str, Any]]:
        with self.connection() as conn:
            where = "" if include_ignored else "WHERE p.is_ignored = 0"
            rows = conn.execute(
                f"SELECT p.* FROM persons p {where} ORDER BY p.updated_at DESC"
            ).fetchall()
            return [self._hydrate_person_summary(conn, dict(r)) for r in rows]

    def get_person_summary(self, person_id: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM persons WHERE id = ?", (person_id,)).fetchone()
            if not row:
                return None
            return self._hydrate_person_summary(conn, dict(row))

    def _hydrate_person_summary(self, conn: sqlite3.Connection, person: dict[str, Any]) -> dict[str, Any]:
        person.pop("centroid", None)
        pid = person["id"]
        stats = conn.execute(
            """
            SELECT
                COUNT(*) as total_detections,
                SUM(CASE WHEN status = 'kept' THEN 1 ELSE 0 END) as kept_count,
                SUM(CASE WHEN status = 'discarded' THEN 1 ELSE 0 END) as discarded_count,
                SUM(CASE WHEN status = 'kept' AND has_cutout = 1 THEN 1 ELSE 0 END) as cutout_count,
                MAX(portrait_score) as max_portrait_score,
                MAX(body_score) as max_body_score,
                COUNT(DISTINCT media_id) as media_count
            FROM detections
            WHERE person_id = ?
            """,
            (pid,),
        ).fetchone()
        person["total_detections"] = int(stats["total_detections"] or 0)
        person["kept_count"] = int(stats["kept_count"] or 0)
        person["discarded_count"] = int(stats["discarded_count"] or 0)
        person["cutout_count"] = int(stats["cutout_count"] or 0)
        person["max_portrait_score"] = round(float(stats["max_portrait_score"] or 0.0), 1)
        person["max_body_score"] = round(float(stats["max_body_score"] or 0.0), 1)
        person["media_count"] = int(stats["media_count"] or 0)

        for key, col in [
            ("cover_detection", "cover_detection_id"),
            ("best_portrait", "best_portrait_id"),
            ("best_body", "best_body_id"),
        ]:
            det_id = person.get(col)
            if det_id:
                drow = conn.execute(
                    "SELECT * FROM detections WHERE id = ?", (det_id,)
                ).fetchone()
                person[key] = self._detection_row_to_dict(drow) if drow else None
            else:
                person[key] = None
        return person

    # ---------- Detections ----------
    def insert_detection(self, det: dict[str, Any]) -> dict[str, Any]:
        det_id = det.get("id") or uuid.uuid4().hex[:12]
        now = utc_now_iso()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO detections (
                    id, media_id, person_id, timestamp_sec, frame_idx,
                    face_bbox, body_bbox, det_score, portrait_score, body_score,
                    overall_score, sharpness_score, frontality_score, separation_score,
                    has_cutout, is_full_body, embedding,
                    frame_path, portrait_path, cutout_path, crop_path, thumb_path,
                    status, keep_category, discard_reason,
                    scene_summary, scene_tags, scene_backend, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    det_id,
                    det["media_id"],
                    det.get("person_id"),
                    float(det.get("timestamp_sec", 0.0)),
                    int(det.get("frame_idx", 0)),
                    json.dumps(det.get("face_bbox")) if det.get("face_bbox") else None,
                    json.dumps(det.get("body_bbox")) if det.get("body_bbox") else None,
                    float(det.get("det_score", 0.0)),
                    float(det.get("portrait_score", 0.0)),
                    float(det.get("body_score", 0.0)),
                    float(det.get("overall_score", 0.0)),
                    float(det.get("sharpness_score", 0.0)),
                    float(det.get("frontality_score", 0.0)),
                    float(det.get("separation_score", 0.0)),
                    1 if det.get("has_cutout") else 0,
                    1 if det.get("is_full_body") else 0,
                    encode_embedding(det.get("embedding")),
                    det.get("frame_path"),
                    det.get("portrait_path"),
                    det.get("cutout_path"),
                    det.get("crop_path"),
                    det.get("thumb_path"),
                    det.get("status", "kept"),
                    det.get("keep_category"),
                    det.get("discard_reason"),
                    det.get("scene_summary"),
                    json.dumps(det.get("scene_tags"), ensure_ascii=False)
                    if det.get("scene_tags") is not None
                    else None,
                    det.get("scene_backend"),
                    now,
                ),
            )
            row = conn.execute("SELECT * FROM detections WHERE id = ?", (det_id,)).fetchone()
            return self._detection_row_to_dict(row)

    def update_detection(self, det_id: str, **kwargs: Any) -> dict[str, Any] | None:
        if "scene_tags" in kwargs and isinstance(kwargs["scene_tags"], list):
            kwargs["scene_tags"] = json.dumps(kwargs["scene_tags"], ensure_ascii=False)
        if not kwargs:
            return self.get_detection(det_id)
        fields = [f"{k} = ?" for k in kwargs]
        values = list(kwargs.values()) + [det_id]
        with self.connection() as conn:
            conn.execute(f"UPDATE detections SET {', '.join(fields)} WHERE id = ?", values)
            row = conn.execute("SELECT * FROM detections WHERE id = ?", (det_id,)).fetchone()
            return self._detection_row_to_dict(row) if row else None

    def get_detection(self, det_id: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT d.*, m.original_filename as media_filename, m.media_type, m.source_deleted
                FROM detections d
                JOIN media m ON d.media_id = m.id
                WHERE d.id = ?
                """,
                (det_id,),
            ).fetchone()
            return self._detection_row_to_dict(row) if row else None

    def list_detections_for_person(
        self,
        person_id: str,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        with self.connection() as conn:
            query = """
                SELECT d.*, m.original_filename as media_filename, m.media_type, m.source_deleted
                FROM detections d
                JOIN media m ON d.media_id = m.id
                WHERE d.person_id = ?
            """
            params: list[Any] = [person_id]
            if status in ("kept", "discarded"):
                query += " AND d.status = ?"
                params.append(status)
            query += " ORDER BY d.overall_score DESC, d.created_at DESC"
            rows = conn.execute(query, params).fetchall()
            return [self._detection_row_to_dict(r) for r in rows]

    def list_detections_for_media(self, media_id: str) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT d.*, p.name as person_name, m.original_filename as media_filename, m.media_type, m.source_deleted
                FROM detections d
                LEFT JOIN persons p ON d.person_id = p.id
                JOIN media m ON d.media_id = m.id
                WHERE d.media_id = ?
                ORDER BY d.timestamp_sec ASC, d.overall_score DESC
                """,
                (media_id,),
            ).fetchall()
            return [self._detection_row_to_dict(r) for r in rows]

    def get_all_detections_with_embeddings(self) -> list[dict[str, Any]]:
        """Used for fast local reverse search across all indexed media."""
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT d.*, p.name as person_name,
                       m.original_filename as media_filename, m.media_type, m.source_deleted, m.duration_sec
                FROM detections d
                LEFT JOIN persons p ON d.person_id = p.id
                JOIN media m ON d.media_id = m.id
                WHERE d.embedding IS NOT NULL
                """
            ).fetchall()
            out = []
            for r in rows:
                emb = decode_embedding(r["embedding"])
                item = self._detection_row_to_dict(r)
                item["embedding_vec"] = emb
                out.append(item)
            return out

    def delete_detection(self, det_id: str) -> str | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT person_id FROM detections WHERE id = ?", (det_id,)
            ).fetchone()
            person_id = row["person_id"] if row else None
            conn.execute("DELETE FROM detections WHERE id = ?", (det_id,))
            return person_id

    @staticmethod
    def _detection_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        d.pop("embedding", None)
        if d.get("face_bbox") and isinstance(d["face_bbox"], str):
            d["face_bbox"] = json.loads(d["face_bbox"])
        if d.get("body_bbox") and isinstance(d["body_bbox"], str):
            d["body_bbox"] = json.loads(d["body_bbox"])
        if d.get("scene_tags") and isinstance(d["scene_tags"], str):
            d["scene_tags"] = json.loads(d["scene_tags"])
        d["has_cutout"] = bool(d.get("has_cutout"))
        d["is_full_body"] = bool(d.get("is_full_body"))
        if "source_deleted" in d:
            d["source_deleted"] = bool(d["source_deleted"])
        return d


# Shared default DB instance
db = Database()

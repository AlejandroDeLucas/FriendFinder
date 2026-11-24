"""Export game-ready person asset packs (portraits, transparent PNG cutouts, crops, originals, metadata)."""

from __future__ import annotations

import json
import re
import uuid
import zipfile
from pathlib import Path
from typing import Any

from ..config import CROPS_DIR, CUTOUTS_DIR, EXPORTS_DIR, FRAMES_DIR, PORTRAITS_DIR, ensure_data_dirs
from ..database import Database, db as default_db


def _safe_folder_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑ _-]+", "", name).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned or "Persona"


def build_export_zip(
    *,
    person_ids: list[str] | None = None,
    include_portraits: bool = True,
    include_cutouts: bool = True,
    include_crops: bool = True,
    include_originals: bool = False,
    only_kept: bool = True,
    database: Database | None = None,
) -> Path:
    """Create a structured ZIP archive ready for game character/avatar pipelines."""
    db = database or default_db
    ensure_data_dirs()

    if person_ids:
        persons = [p for pid in person_ids if (p := db.get_person_summary(pid)) is not None]
    else:
        persons = db.list_persons(include_ignored=False)

    zip_name = f"FriendFinder_Export_{uuid.uuid4().hex[:8]}.zip"
    zip_path = EXPORTS_DIR / zip_name

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        used_folders: dict[str, int] = {}
        manifest: list[dict[str, Any]] = []

        for person in persons:
            base = _safe_folder_name(person["name"])
            idx = used_folders.get(base, 0)
            used_folders[base] = idx + 1
            folder = f"{base}_{idx + 1}" if idx > 0 else base

            dets = db.list_detections_for_person(
                person["id"], status="kept" if only_kept else None
            )
            person_meta = {
                "person_id": person["id"],
                "name": person["name"],
                "notes": person.get("notes", ""),
                "best_portrait_score": person.get("max_portrait_score", 0),
                "best_body_score": person.get("max_body_score", 0),
                "captures": [],
            }

            for rank, d in enumerate(dets, start=1):
                prefix = f"{rank:02d}_score{int(round(d['overall_score']))}_{d['id'][:6]}"
                capture_entry: dict[str, Any] = {
                    "detection_id": d["id"],
                    "source_file": d.get("media_filename"),
                    "media_type": d.get("media_type"),
                    "timestamp_sec": d.get("timestamp_sec"),
                    "portrait_score": d.get("portrait_score"),
                    "body_score": d.get("body_score"),
                    "overall_score": d.get("overall_score"),
                    "is_full_body": d.get("is_full_body"),
                    "scene_summary": d.get("scene_summary"),
                    "scene_tags": d.get("scene_tags"),
                    "files": {},
                }

                if include_portraits and d.get("portrait_path"):
                    src = PORTRAITS_DIR / d["portrait_path"]
                    if src.exists():
                        arc = f"{folder}/portraits/{prefix}_portrait.jpg"
                        zf.write(src, arcname=arc)
                        capture_entry["files"]["portrait"] = arc

                if include_cutouts and d.get("cutout_path"):
                    src = CUTOUTS_DIR / d["cutout_path"]
                    if src.exists():
                        arc = f"{folder}/cutouts_png/{prefix}_cutout.png"
                        zf.write(src, arcname=arc)
                        capture_entry["files"]["cutout_png"] = arc

                if include_crops and d.get("crop_path"):
                    src = CROPS_DIR / d["crop_path"]
                    if src.exists():
                        arc = f"{folder}/crops/{prefix}_crop.jpg"
                        zf.write(src, arcname=arc)
                        capture_entry["files"]["crop"] = arc

                if include_originals and d.get("frame_path"):
                    src = FRAMES_DIR / d["frame_path"]
                    if src.exists():
                        arc = f"{folder}/originals/{prefix}_frame.jpg"
                        zf.write(src, arcname=arc)
                        capture_entry["files"]["original_frame"] = arc

                person_meta["captures"].append(capture_entry)

            zf.writestr(
                f"{folder}/metadata.json",
                json.dumps(person_meta, indent=2, ensure_ascii=False),
            )
            manifest.append(
                {
                    "name": person["name"],
                    "folder": folder,
                    "captures_exported": len(person_meta["captures"]),
                }
            )

        zf.writestr(
            "manifest.json",
            json.dumps({"persons": manifest}, indent=2, ensure_ascii=False),
        )

    return zip_path

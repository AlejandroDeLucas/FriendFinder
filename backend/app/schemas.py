"""Pydantic request and response schemas."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class PersonUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    notes: str | None = Field(default=None, max_length=1000)
    is_ignored: bool | None = None
    cover_detection_id: str | None = None


class MergePersonsRequest(BaseModel):
    source_person_ids: list[str] = Field(min_length=1)


class DetectionStatusUpdateRequest(BaseModel):
    status: str = Field(pattern="^(kept|discarded)$")
    keep_category: str | None = None
    discard_reason: str | None = None


class ReassignDetectionRequest(BaseModel):
    person_id: str | None = None
    new_person_name: str | None = None


class SettingsUpdateRequest(BaseModel):
    video_sample_fps: float | None = Field(default=None, ge=0.2, le=10.0)
    similarity_threshold: float | None = Field(default=None, ge=0.2, le=0.85)
    min_quality_score: float | None = Field(default=None, ge=0.0, le=95.0)
    max_portraits_per_person: int | None = Field(default=None, ge=1, le=50)
    max_fullbody_per_person: int | None = Field(default=None, ge=1, le=50)
    min_video_gap_seconds: float | None = Field(default=None, ge=0.0, le=30.0)
    auto_scene_analysis: bool | None = None
    scene_backend: str | None = Field(default=None, pattern="^(heuristic|blip|ollama)$")
    ollama_url: str | None = None
    ollama_model: str | None = None


class SceneAnalyzeRequest(BaseModel):
    backend: str | None = Field(default=None, pattern="^(heuristic|blip|ollama)$")


class ExportRequest(BaseModel):
    person_ids: list[str] | None = None
    include_portraits: bool = True
    include_cutouts: bool = True
    include_crops: bool = True
    include_originals: bool = False
    only_kept: bool = True
    extra: dict[str, Any] | None = None

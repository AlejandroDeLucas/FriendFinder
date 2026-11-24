"""Face-to-body matching and multi-factor portrait / full-body quality scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .ml_models import DetectedBody, DetectedFace


@dataclass
class QualityMetrics:
    portrait_score: float
    body_score: float
    overall_score: float
    sharpness_score: float
    frontality_score: float
    separation_score: float
    is_full_body: bool
    matched_body: DetectedBody | None = None
    matched_body_idx: int | None = None


def _clamp_box(bbox: list[float], w: int, h: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(w - 1, int(round(bbox[0]))))
    y1 = max(0, min(h - 1, int(round(bbox[1]))))
    x2 = max(x1 + 1, min(w, int(round(bbox[2]))))
    y2 = max(y1 + 1, min(h, int(round(bbox[3]))))
    return x1, y1, x2, y2


def compute_sharpness_score(region_bgr: np.ndarray) -> float:
    """Compute normalized sharpness score (0..100) using Laplacian variance."""
    if region_bgr.size == 0 or region_bgr.shape[0] < 8 or region_bgr.shape[1] < 8:
        return 0.0
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    # Normalize size to 160px on longest side so Laplacian variance is comparable across resolutions
    max_dim = max(gray.shape[:2])
    if max_dim > 160:
        scale = 160.0 / max_dim
        gray = cv2.resize(gray, (max(8, int(gray.shape[1] * scale)), max(8, int(gray.shape[0] * scale))))
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Smooth saturation curve: ~30 -> ~38, ~120 -> ~72, ~300+ -> ~92+
    score = 100.0 * (1.0 - math.exp(-lap_var / 110.0))
    return float(max(0.0, min(100.0, score)))


def compute_lighting_score(region_bgr: np.ndarray) -> float:
    """Score exposure and dynamic range (0..100), penalizing over/under-exposed faces."""
    if region_bgr.size == 0:
        return 50.0
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))

    # Sweet spot for mean brightness: 85..180
    if 85.0 <= mean_val <= 180.0:
        brightness_score = 100.0
    elif mean_val < 85.0:
        brightness_score = max(0.0, 100.0 * (mean_val / 85.0))
    else:
        brightness_score = max(0.0, 100.0 * (1.0 - (mean_val - 180.0) / 75.0))

    # Contrast sweet spot: std >= 32
    contrast_score = min(100.0, (std_val / 40.0) * 100.0)
    return 0.65 * brightness_score + 0.35 * contrast_score


def compute_frontality_score(face: DetectedFace) -> float:
    """Estimate how frontal and clear the face pose is (0..100)."""
    if face.pose is not None:
        pitch, yaw, roll = face.pose
        yaw_pen = min(1.0, abs(yaw) / 55.0)
        pitch_pen = min(1.0, abs(pitch) / 45.0)
        roll_pen = min(1.0, abs(roll) / 45.0)
        combined = 0.55 * yaw_pen + 0.30 * pitch_pen + 0.15 * roll_pen
        return float(max(0.0, min(100.0, 100.0 * (1.0 - combined))))

    # Fallback using 5-point landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
    if face.kps is not None and face.kps.shape == (5, 2):
        left_eye, right_eye, nose = face.kps[0], face.kps[1], face.kps[2]
        eye_dist = float(np.linalg.norm(right_eye - left_eye))
        if eye_dist > 1.0:
            eye_mid_x = 0.5 * (left_eye[0] + right_eye[0])
            horiz_offset = abs(nose[0] - eye_mid_x) / eye_dist
            eye_tilt = abs(right_eye[1] - left_eye[1]) / eye_dist
            penalty = min(1.0, horiz_offset * 1.4 + eye_tilt * 0.8)
            return float(max(0.0, min(100.0, 100.0 * (1.0 - penalty))))

    return 70.0


def match_face_to_body(
    face: DetectedFace,
    bodies: list[DetectedBody],
    used_body_indices: set[int] | None = None,
) -> tuple[int | None, DetectedBody | None]:
    """Match a detected face to the most plausible person body instance."""
    if not bodies:
        return None, None

    fx1, fy1, fx2, fy2 = face.bbox
    fcx = 0.5 * (fx1 + fx2)
    fcy = 0.5 * (fy1 + fy2)
    face_h = max(1.0, fy2 - fy1)
    face_w = max(1.0, fx2 - fx1)

    best_idx: int | None = None
    best_score = -1.0

    for idx, body in enumerate(bodies):
        if used_body_indices and idx in used_body_indices:
            continue
        bx1, by1, bx2, by2 = body.bbox
        bw = max(1.0, bx2 - bx1)
        bh = max(1.0, by2 - by1)

        # Face center should lie horizontally inside (with small margin) and in the upper 55% of body
        if not (bx1 - 0.15 * face_w <= fcx <= bx2 + 0.15 * face_w):
            continue
        if not (by1 - 0.25 * face_h <= fcy <= by1 + 0.58 * bh):
            continue

        # Check mask hit if available
        mask_bonus = 0.0
        if body.mask is not None:
            mh, mw = body.mask.shape[:2]
            iy = max(0, min(mh - 1, int(fcy)))
            ix = max(0, min(mw - 1, int(fcx)))
            if body.mask[iy, ix] > 0:
                mask_bonus = 0.35

        # Prefer face near horizontal center & upper third of body, and reasonable size ratio
        horiz_center = 1.0 - min(1.0, abs(fcx - 0.5 * (bx1 + bx2)) / (0.6 * bw))
        vert_pos = 1.0 - min(1.0, abs((fcy - by1) / bh - 0.16) / 0.45)
        area_overlap_x = max(0.0, min(fx2, bx2) - max(fx1, bx1))
        area_overlap_y = max(0.0, min(fy2, by2) - max(fy1, by1))
        face_in_body = (area_overlap_x * area_overlap_y) / max(1.0, face_w * face_h)

        score = 0.40 * face_in_body + 0.25 * horiz_center + 0.20 * vert_pos + mask_bonus
        if score > best_score and face_in_body >= 0.4:
            best_score = score
            best_idx = idx

    if best_idx is None:
        return None, None
    return best_idx, bodies[best_idx]


def compute_separation_score(
    body_idx: int,
    bodies: list[DetectedBody],
) -> float:
    """Measure how cleanly separated a person is from other people in the frame (0..100)."""
    if len(bodies) <= 1:
        return 100.0

    target = bodies[body_idx]
    tx1, ty1, tx2, ty2 = target.bbox
    target_area = max(1.0, (tx2 - tx1) * (ty2 - ty1))

    max_overlap = 0.0
    for j, other in enumerate(bodies):
        if j == body_idx:
            continue
        ox1, oy1, ox2, oy2 = other.bbox
        ix = max(0.0, min(tx2, ox2) - max(tx1, ox1))
        iy = max(0.0, min(ty2, oy2) - max(ty1, oy1))
        overlap_ratio = (ix * iy) / target_area
        if overlap_ratio > max_overlap:
            max_overlap = overlap_ratio

        # Also check mask boundary touching if masks exist
        if target.mask is not None and other.mask is not None:
            inter = np.logical_and(target.mask > 0, other.mask > 0).sum()
            t_pixels = max(1, int((target.mask > 0).sum()))
            mask_overlap = float(inter) / float(t_pixels)
            max_overlap = max(max_overlap, mask_overlap * 1.4)

    return float(max(0.0, min(100.0, 100.0 * (1.0 - min(1.0, max_overlap * 1.35)))))


def score_detection(
    image_bgr: np.ndarray,
    face: DetectedFace,
    bodies: list[DetectedBody],
    used_body_indices: set[int] | None = None,
) -> QualityMetrics:
    """Compute portrait, full-body, and overall quality scores for a detected person."""
    h, w = image_bgr.shape[:2]
    fx1, fy1, fx2, fy2 = _clamp_box(face.bbox, w, h)
    face_crop = image_bgr[fy1:fy2, fx1:fx2]

    face_h = max(1, fy2 - fy1)
    face_w = max(1, fx2 - fx1)
    face_min_dim = min(face_h, face_w)

    sharpness = compute_sharpness_score(face_crop)
    frontality = compute_frontality_score(face)
    lighting = compute_lighting_score(face_crop)
    det_conf_score = min(100.0, max(0.0, (face.det_score - 0.35) / 0.55 * 100.0))

    # Resolution score: 40px -> ~25, 110px -> ~70, 200px+ -> 100
    res_score = min(100.0, max(0.0, (face_min_dim / 170.0) ** 0.75 * 100.0))

    # Penalty if face touches image border (cut off)
    border_margin = max(2, int(0.01 * min(w, h)))
    cut_off = (
        fx1 <= border_margin
        or fy1 <= border_margin
        or fx2 >= w - border_margin
        or fy2 >= h - border_margin
    )
    border_mult = 0.85 if cut_off else 1.0

    portrait_score = border_mult * (
        0.32 * sharpness
        + 0.25 * res_score
        + 0.23 * frontality
        + 0.10 * det_conf_score
        + 0.10 * lighting
    )
    portrait_score = float(max(0.0, min(100.0, portrait_score)))

    # Match body & compute body / cutout score
    body_idx, body = match_face_to_body(face, bodies, used_body_indices=used_body_indices)
    if body is None or body_idx is None:
        return QualityMetrics(
            portrait_score=round(portrait_score, 1),
            body_score=0.0,
            overall_score=round(portrait_score, 1),
            sharpness_score=round(sharpness, 1),
            frontality_score=round(frontality, 1),
            separation_score=100.0 if len(bodies) <= 1 else 60.0,
            is_full_body=False,
            matched_body=None,
            matched_body_idx=None,
        )

    bx1, by1, bx2, by2 = _clamp_box(body.bbox, w, h)
    bh = max(1, by2 - by1)
    bw = max(1, bx2 - bx1)
    body_crop = image_bgr[by1:by2, bx1:bx2]
    body_sharpness = compute_sharpness_score(body_crop)
    separation = compute_separation_score(body_idx, bodies)

    # Estimate whether this is a full/three-quarter body vs just head/shoulders
    body_to_face_ratio = bh / max(1.0, float(face_h))
    aspect_ratio = bh / max(1.0, float(bw))
    is_full_body = body_to_face_ratio >= 3.6 and aspect_ratio >= 1.35

    # Reward larger body height & full-body coverage
    body_res_score = min(100.0, (bh / 420.0) ** 0.7 * 100.0)
    coverage_score = min(100.0, (body_to_face_ratio / 5.5) * 100.0)
    mask_bonus = 100.0 if body.mask is not None else 40.0

    body_score = (
        0.25 * body_res_score
        + 0.25 * coverage_score
        + 0.22 * separation
        + 0.13 * body_sharpness
        + 0.10 * portrait_score
        + 0.05 * mask_bonus
    )
    body_score = float(max(0.0, min(100.0, body_score)))
    overall_score = max(portrait_score, 0.58 * portrait_score + 0.42 * body_score)

    return QualityMetrics(
        portrait_score=round(portrait_score, 1),
        body_score=round(body_score, 1),
        overall_score=round(overall_score, 1),
        sharpness_score=round(sharpness, 1),
        frontality_score=round(frontality, 1),
        separation_score=round(separation, 1),
        is_full_body=is_full_body,
        matched_body=body,
        matched_body_idx=body_idx,
    )

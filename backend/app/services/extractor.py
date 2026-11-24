"""Image asset extraction: portrait crops, transparent PNG cutouts (removing others), rectangular crops, and original frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..config import CROPS_DIR, CUTOUTS_DIR, FRAMES_DIR, PORTRAITS_DIR, THUMBS_DIR, ensure_data_dirs
from .ml_models import DetectedBody


@dataclass
class ExtractedAssets:
    frame_path: str | None = None
    portrait_path: str | None = None
    cutout_path: str | None = None
    crop_path: str | None = None
    thumb_path: str | None = None
    has_cutout: bool = False


def _save_jpeg(path: Path, img_bgr: np.ndarray, quality: int = 93) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return path.name


def _resize_max(img: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return img
    scale = max_dim / float(m)
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def compute_portrait_box(face_bbox: list[float], img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Expand face bbox into a balanced head-and-shoulders portrait box."""
    fx1, fy1, fx2, fy2 = face_bbox
    fw = max(1.0, fx2 - fx1)
    fh = max(1.0, fy2 - fy1)
    cx = 0.5 * (fx1 + fx2)
    cy = 0.5 * (fy1 + fy2)

    # Portrait framing: generous headroom + shoulders
    px1 = int(round(cx - 1.15 * fw))
    px2 = int(round(cx + 1.15 * fw))
    py1 = int(round(fy1 - 0.75 * fh))
    py2 = int(round(fy2 + 1.25 * fh))

    px1 = max(0, min(img_w - 1, px1))
    py1 = max(0, min(img_h - 1, py1))
    px2 = max(px1 + 1, min(img_w, px2))
    py2 = max(py1 + 1, min(img_h, py2))
    return px1, py1, px2, py2


def _refine_person_mask(
    target_mask: np.ndarray,
    other_bodies: list[DetectedBody],
) -> np.ndarray:
    """Clean up segmentation mask, subtract other people's masks, and apply soft alpha edges."""
    mask = (target_mask > 127).astype(np.uint8) * 255

    # Explicitly subtract pixels belonging to OTHER people so nobody else bleeds in
    for other in other_bodies:
        if other.mask is not None and other.mask.shape == mask.shape:
            other_bin = (other.mask > 127).astype(np.uint8) * 255
            # Slight erosion on other mask avoids eating the target's shoulder boundary
            k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            other_core = cv2.erode(other_bin, k_small, iterations=1)
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(other_core))

    # Close tiny internal pinholes and keep largest connected component
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    # Gentle edge feathering for crisp studio-style transparent cutout
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    alpha = np.where(mask > 0, np.maximum(blurred, 210), np.minimum(blurred, 90)).astype(np.uint8)
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    return alpha


def extract_thumbnail_only(
    image_bgr: np.ndarray,
    det_id: str,
    face_bbox: list[float],
) -> str:
    """Save a lightweight portrait thumbnail (used for discarded detections so disk stays lean)."""
    ensure_data_dirs()
    h, w = image_bgr.shape[:2]
    px1, py1, px2, py2 = compute_portrait_box(face_bbox, w, h)
    crop = _resize_max(image_bgr[py1:py2, px1:px2], 320)
    return _save_jpeg(THUMBS_DIR / f"{det_id}.jpg", crop, quality=82)


def extract_all_assets(
    image_bgr: np.ndarray,
    det_id: str,
    face_bbox: list[float],
    matched_body: DetectedBody | None,
    all_bodies: list[DetectedBody],
    *,
    shared_frame_filename: str | None = None,
) -> ExtractedAssets:
    """Generate all 3 requested formats + thumbnail:
    1. Portrait crop (head + shoulders)
    2. Transparent PNG cutout isolating ONLY this person (removing others & background)
    3. Full original frame + clean rectangular bounding-box crop
    """
    ensure_data_dirs()
    h, w = image_bgr.shape[:2]

    # 1. Original frame (reused if multiple people share the same frame)
    if shared_frame_filename and (FRAMES_DIR / shared_frame_filename).exists():
        frame_name = shared_frame_filename
    else:
        frame_name = _save_jpeg(FRAMES_DIR / f"{det_id}_frame.jpg", image_bgr, quality=92)

    # 2. High-res Portrait (head + shoulders)
    px1, py1, px2, py2 = compute_portrait_box(face_bbox, w, h)
    portrait_img = image_bgr[py1:py2, px1:px2]
    portrait_name = _save_jpeg(PORTRAITS_DIR / f"{det_id}_portrait.jpg", portrait_img, quality=95)

    # Lightweight thumb for fast gallery rendering
    thumb_name = _save_jpeg(THUMBS_DIR / f"{det_id}.jpg", _resize_max(portrait_img, 360), quality=85)

    # 3. Rectangular person crop (full body if detected, otherwise expanded portrait)
    if matched_body is not None:
        bx1, by1, bx2, by2 = matched_body.bbox
        pad_x = int(0.06 * (bx2 - bx1))
        pad_y = int(0.04 * (by2 - by1))
        cx1 = max(0, int(bx1) - pad_x)
        cy1 = max(0, int(by1) - pad_y)
        cx2 = min(w, int(bx2) + pad_x)
        cy2 = min(h, int(by2) + pad_y)
    else:
        cx1, cy1, cx2, cy2 = px1, py1, px2, py2
    crop_img = image_bgr[cy1:cy2, cx1:cx2]
    crop_name = _save_jpeg(CROPS_DIR / f"{det_id}_crop.jpg", crop_img, quality=94)

    # 4. Transparent PNG cutout isolating ONLY this person (other people & background removed!)
    cutout_name: str | None = None
    has_cutout = False
    if matched_body is not None and matched_body.mask is not None:
        other_bodies = [b for b in all_bodies if b is not matched_body]
        alpha = _refine_person_mask(matched_body.mask, other_bodies)
        ys, xs = np.where(alpha > 20)
        if len(xs) > 30 and len(ys) > 30:
            pad = 12
            mx1 = max(0, int(xs.min()) - pad)
            my1 = max(0, int(ys.min()) - pad)
            mx2 = min(w, int(xs.max()) + pad)
            my2 = min(h, int(ys.max()) + pad)

            bgr_region = image_bgr[my1:my2, mx1:mx2]
            alpha_region = alpha[my1:my2, mx1:mx2]
            bgra = cv2.cvtColor(bgr_region, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = alpha_region

            cutout_path = CUTOUTS_DIR / f"{det_id}_cutout.png"
            cv2.imwrite(str(cutout_path), bgra)
            cutout_name = cutout_path.name
            has_cutout = True

    return ExtractedAssets(
        frame_path=frame_name,
        portrait_path=portrait_name,
        cutout_path=cutout_name,
        crop_path=crop_name,
        thumb_path=thumb_name,
        has_cutout=has_cutout,
    )

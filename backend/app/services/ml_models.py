"""Lazy-loaded local ML models (InsightFace + YOLOv8-seg) with auto hardware acceleration."""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from ..config import MODELS_DIR, detect_compute_device, ensure_data_dirs

logger = logging.getLogger(__name__)


@dataclass
class DetectedFace:
    bbox: list[float]  # [x1, y1, x2, y2]
    det_score: float
    embedding: np.ndarray  # normalized 512-D float32
    kps: np.ndarray | None = None  # 5x2 facial landmarks
    pose: tuple[float, float, float] | None = None  # (pitch, yaw, roll) degrees if available


@dataclass
class DetectedBody:
    bbox: list[float]  # [x1, y1, x2, y2]
    conf: float
    mask: np.ndarray | None = None  # uint8 (H, W) binary mask (0 or 255) at frame size


@dataclass
class DetectedObject:
    label: str
    conf: float
    bbox: list[float]


@dataclass
class FrameAnalysis:
    faces: list[DetectedFace] = field(default_factory=list)
    bodies: list[DetectedBody] = field(default_factory=list)
    objects: list[DetectedObject] = field(default_factory=list)


class ModelManager:
    """Thread-safe lazy loader for local face recognition and person segmentation models."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._face_app: Any = None
        self._yolo_model: Any = None
        self._device_info = detect_compute_device()
        self._face_pack = "buffalo_l"
        self._yolo_name = "yolov8n-seg.pt"

    @property
    def device_info(self) -> dict[str, str]:
        return dict(self._device_info)

    def _ensure_face_app(self, pack: str = "buffalo_l") -> Any:
        with self._lock:
            if self._face_app is not None and self._face_pack == pack:
                return self._face_app

            ensure_data_dirs()
            from insightface.app import FaceAnalysis

            providers = self._device_info["onnx_providers"].split(",")
            ctx_id = 0 if "CUDAExecutionProvider" in providers else -1

            logger.info("Loading InsightFace (%s) with providers=%s", pack, providers)
            app = FaceAnalysis(
                name=pack,
                root=str(MODELS_DIR / "insightface"),
                providers=providers,
            )
            app.prepare(ctx_id=ctx_id, det_size=(640, 640), det_thresh=0.45)
            self._face_app = app
            self._face_pack = pack
            return self._face_app

    def _ensure_yolo(self, model_name: str = "yolov8n-seg.pt") -> Any:
        with self._lock:
            if self._yolo_model is not None and self._yolo_name == model_name:
                return self._yolo_model

            ensure_data_dirs()
            from ultralytics import YOLO

            model_path = MODELS_DIR / model_name
            target = str(model_path) if model_path.exists() else model_name
            logger.info(
                "Loading Ultralytics YOLOv8-seg (%s) on %s",
                target,
                self._device_info["torch_device"],
            )
            prev_cwd = os.getcwd()
            try:
                os.chdir(str(MODELS_DIR))
                model = YOLO(target)
            finally:
                os.chdir(prev_cwd)
            self._yolo_model = model
            self._yolo_name = model_name
            return self._yolo_model

    def detect_faces_only(self, image_bgr: np.ndarray, pack: str = "buffalo_l") -> list[DetectedFace]:
        """Run InsightFace face detection + ArcFace embedding extraction."""
        app = self._ensure_face_app(pack=pack)
        with self._lock:
            raw_faces = app.get(image_bgr)

        results: list[DetectedFace] = []
        h, w = image_bgr.shape[:2]
        for f in raw_faces:
            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                raw_emb = getattr(f, "embedding", None)
                if raw_emb is None:
                    continue
                norm = float(np.linalg.norm(raw_emb))
                emb = raw_emb / norm if norm > 0 else raw_emb
            emb = np.asarray(emb, dtype=np.float32).reshape(-1)

            x1, y1, x2, y2 = [float(v) for v in f.bbox]
            x1 = max(0.0, min(float(w - 1), x1))
            y1 = max(0.0, min(float(h - 1), y1))
            x2 = max(x1 + 1.0, min(float(w), x2))
            y2 = max(y1 + 1.0, min(float(h), y2))

            pose_arr = getattr(f, "pose", None)
            pose_tuple = None
            if pose_arr is not None and len(pose_arr) >= 3:
                pose_tuple = (float(pose_arr[0]), float(pose_arr[1]), float(pose_arr[2]))

            kps = getattr(f, "kps", None)
            kps_arr = np.asarray(kps, dtype=np.float32) if kps is not None else None

            results.append(
                DetectedFace(
                    bbox=[x1, y1, x2, y2],
                    det_score=float(getattr(f, "det_score", 0.0)),
                    embedding=emb,
                    kps=kps_arr,
                    pose=pose_tuple,
                )
            )
        return results

    def detect_bodies_and_objects(
        self, image_bgr: np.ndarray, model_name: str = "yolov8n-seg.pt"
    ) -> tuple[list[DetectedBody], list[DetectedObject]]:
        """Run YOLOv8-seg to get per-person pixel masks + COCO context objects."""
        model = self._ensure_yolo(model_name=model_name)
        h, w = image_bgr.shape[:2]
        with self._lock:
            preds = model.predict(
                source=image_bgr,
                device=self._device_info["torch_device"],
                conf=0.35,
                iou=0.5,
                verbose=False,
                retina_masks=True,
            )

        bodies: list[DetectedBody] = []
        objects: list[DetectedObject] = []

        if not preds:
            return bodies, objects

        res = preds[0]
        boxes = res.boxes
        masks = res.masks
        names = res.names or {}

        if boxes is None or len(boxes) == 0:
            return bodies, objects

        xyxy_all = boxes.xyxy.cpu().numpy()
        cls_all = boxes.cls.cpu().numpy().astype(int)
        conf_all = boxes.conf.cpu().numpy()

        mask_data = None
        if masks is not None and masks.data is not None:
            mask_data = masks.data.cpu().numpy()

        for i in range(len(boxes)):
            cls_id = int(cls_all[i])
            conf = float(conf_all[i])
            bbox = [float(v) for v in xyxy_all[i]]
            label = str(names.get(cls_id, cls_id))

            if cls_id == 0:  # COCO class 0 = person
                person_mask: np.ndarray | None = None
                if mask_data is not None and i < len(mask_data):
                    m = mask_data[i]
                    if m.shape[0] != h or m.shape[1] != w:
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
                    person_mask = (m > 0.5).astype(np.uint8) * 255
                bodies.append(DetectedBody(bbox=bbox, conf=conf, mask=person_mask))
            else:
                objects.append(DetectedObject(label=label, conf=conf, bbox=bbox))

        return bodies, objects

    def analyze_frame(
        self,
        image_bgr: np.ndarray,
        *,
        run_segmentation: bool = True,
        face_pack: str = "buffalo_l",
        yolo_model: str = "yolov8n-seg.pt",
    ) -> FrameAnalysis:
        """Analyze a frame for faces, person segmentation masks, and context objects."""
        faces = self.detect_faces_only(image_bgr, pack=face_pack)
        if not faces:
            return FrameAnalysis(faces=[], bodies=[], objects=[])
        bodies: list[DetectedBody] = []
        objects: list[DetectedObject] = []
        if run_segmentation:
            bodies, objects = self.detect_bodies_and_objects(image_bgr, model_name=yolo_model)
        return FrameAnalysis(faces=faces, bodies=bodies, objects=objects)


model_manager = ModelManager()

"""Optional 100% local scene and activity analyzer (Heuristics + Local BLIP VLM + Ollama)."""

from __future__ import annotations

import base64
import logging
import threading
from pathlib import Path
from typing import Any

import cv2
import httpx
from PIL import Image

from .ml_models import DetectedObject

logger = logging.getLogger(__name__)

# Spanish translation of common COCO object labels from YOLOv8
COCO_ES: dict[str, str] = {
    "bicycle": "en bicicleta",
    "car": "junto a un coche",
    "motorcycle": "en moto",
    "airplane": "cerca de un avión",
    "bus": "en autobús",
    "train": "en tren",
    "boat": "en barco",
    "bench": "en un banco",
    "bird": "con aves cerca",
    "cat": "con un gato",
    "dog": "con un perro",
    "horse": "con un caballo",
    "backpack": "con mochila",
    "umbrella": "con paraguas",
    "handbag": "con bolso",
    "suitcase": "de viaje con maleta",
    "frisbee": "jugando con frisbee",
    "skis": "esquiando",
    "snowboard": "haciendo snowboard",
    "sports ball": "jugando con balón",
    "skateboard": "en skate",
    "surfboard": "haciendo surf",
    "tennis racket": "jugando al tenis",
    "bottle": "con bebida",
    "wine glass": "con una copa",
    "cup": "tomando algo",
    "fork": "comiendo",
    "pizza": "comiendo pizza",
    "cake": "celebrando con tarta",
    "chair": "sentado/a o junto a sillas",
    "couch": "en el sofá",
    "bed": "en una habitación",
    "dining table": "en una mesa",
    "tv": "frente a una pantalla",
    "laptop": "usando un portátil",
    "cell phone": "con el móvil",
    "book": "leyendo",
    "potted plant": "con plantas de fondo",
}


class LocalSceneAnalyzer:
    """Modular local scene/activity analyzer with lazy model loading."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._blip_processor: Any = None
        self._blip_model: Any = None

    def analyze_heuristic(
        self,
        *,
        num_people: int,
        is_full_body: bool,
        frontality_score: float,
        objects: list[DetectedObject],
    ) -> tuple[str, list[str]]:
        """Instant zero-overhead contextual summary from YOLOv8 COCO objects & framing."""
        parts: list[str] = []
        tags: list[str] = []

        if is_full_body:
            parts.append("Plano de cuerpo entero")
            tags.append("cuerpo entero")
        else:
            parts.append("Retrato / plano cercano")
            tags.append("retrato")

        if frontality_score >= 78:
            parts.append("mirando hacia la cámara")
            tags.append("frontal")
        elif frontality_score >= 52:
            parts.append("en ángulo semi-lateral")
            tags.append("semi-perfil")
        else:
            parts.append("de perfil")
            tags.append("perfil")

        if num_people <= 1:
            parts.append("solo/a en la escena")
            tags.append("solo")
        elif num_people == 2:
            parts.append("acompañado/a (2 personas)")
            tags.append("pareja/dúo")
        else:
            parts.append(f"en grupo ({num_people} personas)")
            tags.append("grupo")

        seen_labels: set[str] = set()
        context_phrases: list[str] = []
        for obj in sorted(objects, key=lambda o: o.conf, reverse=True):
            if obj.label in seen_labels or obj.conf < 0.42:
                continue
            seen_labels.add(obj.label)
            if obj.label in COCO_ES:
                context_phrases.append(COCO_ES[obj.label])
                tags.append(obj.label)
            if len(context_phrases) >= 2:
                break

        if context_phrases:
            parts.append(", ".join(context_phrases))

        summary = "; ".join(parts[:2])
        if len(parts) > 2:
            summary += " · " + ", ".join(parts[2:])
        return summary, tags[:8]

    def _ensure_blip(self) -> tuple[Any, Any]:
        with self._lock:
            if self._blip_model is not None and self._blip_processor is not None:
                return self._blip_processor, self._blip_model

            from transformers import BlipForConditionalGeneration, BlipProcessor

            model_id = "Salesforce/blip-image-captioning-base"
            logger.info("Loading local BLIP model (%s)...", model_id)
            processor = BlipProcessor.from_pretrained(model_id)
            model = BlipForConditionalGeneration.from_pretrained(model_id)
            model.eval()
            self._blip_processor = processor
            self._blip_model = model
            return processor, model

    def analyze_image_file(
        self,
        image_path: Path,
        *,
        backend: str = "heuristic",
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "llava",
        fallback_context: dict[str, Any] | None = None,
    ) -> tuple[str, list[str], str]:
        """Run optional local scene/activity analysis on an image file.

        Returns (summary_text, tags, backend_used).
        """
        if backend == "blip":
            try:
                processor, model = self._ensure_blip()
                pil_img = Image.open(image_path).convert("RGB")
                inputs = processor(pil_img, text="a photo of a person", return_tensors="pt")
                out = model.generate(**inputs, max_new_tokens=45)
                caption = processor.decode(out[0], skip_special_tokens=True).strip()
                tags = [w.strip(".,").lower() for w in caption.split() if len(w) > 3][:6]
                return caption, tags, "blip"
            except Exception as exc:
                logger.warning("BLIP local analysis failed (%s), falling back to heuristic", exc)

        if backend == "ollama":
            try:
                raw_bytes = image_path.read_bytes()
                b64 = base64.b64encode(raw_bytes).decode("ascii")
                payload = {
                    "model": ollama_model,
                    "prompt": (
                        "Describe brevemente en español (1 frase corta) qué está haciendo "
                        "la persona principal en esta imagen y el entorno."
                    ),
                    "images": [b64],
                    "stream": False,
                }
                resp = httpx.post(
                    f"{ollama_url.rstrip('/')}/api/generate",
                    json=payload,
                    timeout=45.0,
                )
                resp.raise_for_status()
                text = (resp.json().get("response") or "").strip()
                if text:
                    return text, ["ollama", ollama_model], "ollama"
            except Exception as exc:
                logger.warning("Ollama local analysis failed (%s), falling back to heuristic", exc)

        # Instant local heuristic using YOLOv8 objects on the image
        ctx = fallback_context or {}
        if "objects" in ctx:
            summary, tags = self.analyze_heuristic(
                num_people=int(ctx.get("num_people", 1)),
                is_full_body=bool(ctx.get("is_full_body", False)),
                frontality_score=float(ctx.get("frontality_score", 75.0)),
                objects=ctx["objects"],
            )
            return summary, tags, "heuristic"

        # If called standalone on an image path without precomputed objects, run YOLO quickly
        from .ml_models import model_manager

        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return "Escena analizada", [], "heuristic"
        bodies, objects = model_manager.detect_bodies_and_objects(img_bgr)
        summary, tags = self.analyze_heuristic(
            num_people=max(1, len(bodies)),
            is_full_body=bool(ctx.get("is_full_body", False)),
            frontality_score=float(ctx.get("frontality_score", 75.0)),
            objects=objects,
        )
        return summary, tags, "heuristic"


scene_analyzer = LocalSceneAnalyzer()

# core/yolo/service.py

"""Servicios para interactuar con los modelos YOLOv8 general y especialista."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from django.conf import settings
from ultralytics import YOLO


@dataclass(slots=True)
class BoundingBox:
  x1: float;
  y1: float;
  x2: float;
  y2: float


@dataclass(slots=True)
class DetectionResult:
  class_id: int
  class_name: str
  confidence: float
  bounding_box: BoundingBox


class HelmetDetectionService:
  """Servicio que gestiona un modelo general y un modelo especialista en cascos."""

  def __init__(self) -> None:
    config = getattr(settings, "YOLO_CONFIG", {})

    # --- Cargar Modelo General (yolov8n.pt) ---
    general_weights_path_str = config.get("general_model_weights")
    if not general_weights_path_str:
      raise ValueError("La ruta 'general_model_weights' no está definida en YOLO_CONFIG.")

    general_weights_path = Path(general_weights_path_str)
    if not general_weights_path.exists():
      raise FileNotFoundError(f"No se encontró el modelo general en: {general_weights_path}")

    self.general_model = YOLO(str(general_weights_path))
    self.general_model_names = self.general_model.names
    print(f"✅ Modelo General cargado desde: {general_weights_path}")

    # --- Cargar Modelo Especialista (best.pt) ---
    helmet_weights_path_str = config.get("helmet_model_weights")
    if not helmet_weights_path_str:
      raise ValueError("La ruta 'helmet_model_weights' no está definida en YOLO_CONFIG.")

    helmet_weights_path = Path(helmet_weights_path_str)
    if not helmet_weights_path.exists():
      raise FileNotFoundError(f"No se encontró el modelo de cascos en: {helmet_weights_path}")

    self.helmet_model = YOLO(str(helmet_weights_path))
    self.helmet_model_names = self.helmet_model.names
    print(f"✅ Modelo Especialista de Cascos cargado desde: {helmet_weights_path}")
    print(f"👍 Clases del especialista: {list(self.helmet_model_names.values())}")

  def extract_detections(self, result, model_names) -> list[DetectionResult]:
    """Extrae detecciones de un resultado de YOLO usando los nombres de clase correctos."""
    detections: list[DetectionResult] = []
    boxes = getattr(result, "boxes", None)
    if not boxes: return detections

    for box in boxes:
      class_id = int(box.cls[0])
      detections.append(
        DetectionResult(
          class_id=class_id,
          class_name=model_names.get(class_id, f"clase_{class_id}"),
          confidence=float(box.conf[0]),
          bounding_box=BoundingBox(*map(float, box.xyxy[0])),
        )
      )
    return detections

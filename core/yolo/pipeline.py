from __future__ import annotations

import base64
import json
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict

import cv2
import google.generativeai as genai
import numpy as np
from PIL import Image
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from django.conf import settings
from django.utils import timezone
from django.utils.timezone import timedelta
from scipy.spatial import distance as dist

from .service import HelmetDetectionService, BoundingBox
from ..models import Camara, Deteccion


class DetectorError(RuntimeError):
  pass


@dataclass(slots=True)
class DetectorSummary:
  frames_processed: int = 0
  detections_with_helmet: int = 0
  detections_without_helmet: int = 0
  saved_infractions: int = 0


@dataclass
class TrackedObject:
  objectID: int
  centroid: tuple
  rect: tuple
  frames_tracked: int = 0
  frames_disappeared: int = 0
  best_frame: np.ndarray | None = None
  best_rect: tuple | None = None
  best_frame_area: int = 0
  best_frame_quality: float = 0.0
  already_processed: bool = False


class CentroidTracker:
  """
  Tracker mejorado con:
  - Persistencia de IDs únicos
  - Captura más agresiva de frames
  - Mejor manejo de oclusiones
  """

  def __init__(self, maxDisappeared=30, minFramesTracked=3, minFrameQuality=100):
    self.nextObjectID = 0
    self.objects: Dict[int, TrackedObject] = OrderedDict()
    self.disappeared = OrderedDict()
    self.maxDisappeared = maxDisappeared  # REDUCIDO: 30 frames (~1 seg a 30fps)
    self.minFramesTracked = minFramesTracked  # REDUCIDO: de 5 a 3 frames
    self.minFrameQuality = minFrameQuality  # NUEVO: umbral mínimo de calidad
    self.finished_tracks: list[TrackedObject] = []
    self.all_used_ids = set()  # NUEVO: IDs que ya se usaron

  def register(self, centroid, rect):
    new_id = self.nextObjectID
    self.objects[new_id] = TrackedObject(
      objectID=new_id,
      centroid=centroid,
      rect=rect,
      frames_tracked=1
    )
    self.disappeared[new_id] = 0
    self.all_used_ids.add(new_id)  # Marcar ID como usado
    self.nextObjectID += 1
    print(f"  🆕 Nueva moto registrada: ID {new_id}")

  def deregister(self, objectID):
    obj = self.objects.get(objectID)
    if not obj:
      return

    # CAMBIO CRÍTICO 1: Guardar tracks más cortos
    should_save = (
      obj.frames_tracked >= self.minFramesTracked and
      obj.best_frame is not None and
      obj.best_frame_quality >= self.minFrameQuality
    )

    if should_save:
      print(f"  🏁 Pista finalizada: Moto ID {obj.objectID} "
            f"({obj.frames_tracked} frames, calidad={obj.best_frame_quality:.0f})")
      self.finished_tracks.append(obj)
    else:
      # CAMBIO CRÍTICO 2: Avisar cuando se pierde un track sin guardar
      print(f"  ⚠️  Pista descartada: Moto ID {obj.objectID} - "
            f"frames={obj.frames_tracked}/{self.minFramesTracked}, "
            f"calidad={obj.best_frame_quality:.0f}/{self.minFrameQuality}, "
            f"best_frame={'✓' if obj.best_frame is not None else '✗'}")

    if objectID in self.objects:
      del self.objects[objectID]
    if objectID in self.disappeared:
      del self.disappeared[objectID]

  def update(self, rects, frame=None):
    """
    NUEVO: Acepta el frame actual para captura proactiva
    """
    # Si no hay detecciones, incrementar contador de desapariciones
    if len(rects) == 0:
      for objectID in list(self.disappeared.keys()):
        self.disappeared[objectID] += 1
        self.objects[objectID].frames_disappeared += 1

        # CAMBIO CRÍTICO 3: Captura de emergencia antes de perder el track
        obj = self.objects[objectID]
        if (self.disappeared[objectID] == self.maxDisappeared - 5 and
          frame is not None and
          obj.best_frame is None):
          print(f"  🚨 Captura de emergencia para Moto ID {objectID}")
          try:
            x1, y1, x2, y2 = obj.rect
            crop = frame[int(y1):int(y2), int(x1):int(x2)]
            if crop.size > 0:
              obj.best_frame = frame.copy()
              obj.best_rect = obj.rect
              obj.best_frame_quality = calculate_sharpness(crop) * 0.5  # Penalización
          except Exception as e:
            print(f"  ❌ Error en captura de emergencia: {e}")

        if self.disappeared[objectID] > self.maxDisappeared:
          self.deregister(objectID)
      return

    inputCentroids = np.array([((r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0) for r in rects])

    # Registrar nuevos objetos si no hay tracks activos
    if len(self.objects) == 0:
      for i in range(len(rects)):
        self.register(inputCentroids[i], rects[i])
    else:
      objectIDs = list(self.objects.keys())
      objectCentroids = [o.centroid for o in self.objects.values()]

      # Calcular distancias
      D = dist.cdist(np.array(objectCentroids), inputCentroids)

      # CAMBIO CRÍTICO 4: Matching más estricto para evitar swaps
      MAX_DISTANCE = 150  # píxeles
      rows = D.min(axis=1).argsort()
      cols = D.argmin(axis=1)[rows]

      usedRows, usedCols = set(), set()

      for row, col in zip(rows, cols):
        if row in usedRows or col in usedCols:
          continue

        # Rechazar matches con distancia muy grande
        if D[row, col] > MAX_DISTANCE:
          print(f"  ⚠️  Match rechazado por distancia: {D[row, col]:.0f}px > {MAX_DISTANCE}px")
          continue

        objectID = objectIDs[row]
        self.objects[objectID].centroid = inputCentroids[col]
        self.objects[objectID].rect = rects[col]
        self.objects[objectID].frames_tracked += 1
        self.disappeared[objectID] = 0
        self.objects[objectID].frames_disappeared = 0
        usedRows.add(row)
        usedCols.add(col)

      # Manejar objetos que desaparecieron
      unusedRows = set(range(D.shape[0])).difference(usedRows)
      unusedCols = set(range(D.shape[1])).difference(usedCols)

      for row in unusedRows:
        objectID = objectIDs[row]
        self.disappeared[objectID] += 1
        self.objects[objectID].frames_disappeared += 1
        if self.disappeared[objectID] > self.maxDisappeared:
          self.deregister(objectID)

      # Registrar nuevos objetos
      for col in unusedCols:
        self.register(inputCentroids[col], rects[col])


def _resolve_camera_source(camara: Camara) -> Path:
  if camara.tipo_fuente != "archivo" or not camara.archivo_video:
    raise DetectorError("La detección solo está disponible para cámaras con un archivo de video local.")
  relative_path = Path(camara.archivo_video)
  source_path = Path(settings.BASE_DIR) / "static" / relative_path
  if not source_path.exists():
    raise DetectorError(f"El archivo de video para '{camara.nombre}' no existe en la ruta: {source_path}")
  return source_path


def boxes_overlap(box1, box2, threshold=0.5):
  x1_min, y1_min, x1_max, y1_max = box1 if isinstance(box1, tuple) else (box1.x1, box1.y1, box1.x2, box1.y2)
  x2_min, y2_min, x2_max, y2_max = box2 if isinstance(box2, tuple) else (box2.x1, box2.y1, box2.x2, box2.y2)
  inter_xmin = max(x1_min, x2_min)
  inter_ymin = max(y1_min, y2_min)
  inter_xmax = min(x1_max, x2_max)
  inter_ymax = min(y1_max, y2_max)
  inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
  box1_area = (x1_max - x1_min) * (y1_max - y1_min)
  box2_area = (x2_max - x2_min) * (y2_max - y2_min)
  union_area = box1_area + box2_area - inter_area
  if union_area == 0:
    return False
  return (inter_area / union_area) > threshold


def calculate_sharpness(image: np.ndarray) -> float:
  if image.size == 0:
    return 0.0
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  gray = cv2.GaussianBlur(gray, (3, 3), 0)
  return cv2.Laplacian(gray, cv2.CV_64F).var()


def preprocess_for_yolo(frame: np.ndarray, target_size: int = 640) -> Tuple[np.ndarray, float, Tuple[int, int]]:
  orig_h, orig_w = frame.shape[:2]
  ratio = min(target_size / orig_w, target_size / orig_h)
  new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)
  resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
  padded_frame = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
  pad_w, pad_h = (target_size - new_w) // 2, (target_size - new_h) // 2
  padded_frame[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized_frame
  return padded_frame, ratio, (pad_w, pad_h)


def scale_bounding_box(box: BoundingBox, orig_shape: Tuple[int, int], ratio: float,
                       pad: Tuple[int, int]) -> BoundingBox:
  pad_w, pad_h = pad
  orig_h, orig_w = orig_shape
  orig_x1 = np.clip((box.x1 - pad_w) / ratio, 0, orig_w)
  orig_y1 = np.clip((box.y1 - pad_h) / ratio, 0, orig_h)
  orig_x2 = np.clip((box.x2 - pad_w) / ratio, 0, orig_w)
  orig_y2 = np.clip((box.y2 - pad_h) / ratio, 0, orig_h)
  return BoundingBox(x1=orig_x1, y1=orig_y1, x2=orig_x2, y2=orig_y2)


def extract_motorcycle_crop(frame: np.ndarray, rect: tuple) -> np.ndarray:
  x1, y1, x2, y2 = map(int, rect)
  h, w = frame.shape[:2]
  center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
  half_size = 320
  crop_x1, crop_y1 = max(0, center_x - half_size), max(0, center_y - half_size)
  crop_x2, crop_y2 = min(w, center_x + half_size), min(h, center_y + half_size)
  crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
  if crop.shape[:2] != (640, 640):
    padded = np.full((640, 640, 3), 114, dtype=np.uint8)
    h_c, w_c = crop.shape[:2]
    p_y, p_x = (640 - h_c) // 2, (640 - w_c) // 2
    padded[p_y:p_y + h_c, p_x:p_x + w_c] = crop
    return padded
  return crop


def validate_with_gemini(crop_image: np.ndarray, initial_prediction: str, confidence: float) -> Tuple[str, float]:
  if initial_prediction == 'con_casco' and confidence >= 0.88:
    print(f"   ✅ Confianza alta en 'con_casco' ({confidence:.2f}), aceptando sin Gemini.")
    return 'con_casco', confidence
  try:
    api_key = "AIzaSyC0J0E4vp6sHy7UhDj18hSZxB4v3VNkxnY"
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-flash-latest')
    pil_image = Image.fromarray(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB))
    prompt = f'Eres un experto en seguridad vial. Analiza esta imagen y da tu veredicto final sobre si el conductor lleva casco. La primera IA clasificó esto como "{initial_prediction}" con confianza {confidence:.2f}. Responde SOLO con un JSON con el formato: {{"presencia_valida": true/false, "usa_casco": true/false, "confianza_veredicto": 0.0-1.0, "razonamiento": "Explica brevemente."}}'
    response = model.generate_content([prompt, pil_image])
    result = json.loads(response.text.strip().replace("```json", "").replace("```", "").strip())
    is_valid, use_helmet, gemini_conf = result.get("presencia_valida", False), result.get("usa_casco", False), float(
      result.get("confianza_veredicto", 0.0))
    print(
      f"\n🤖 Veredicto de Gemini:\n   - ¿Válida?: {'SÍ' if is_valid else 'NO'}\n   - ¿Casco?: {'SÍ' if use_helmet else 'NO'}\n   - Confianza: {gemini_conf:.2f}\n   - Razón: {result.get('razonamiento', '')}")
    if not is_valid:
      return "descartado", 0.0
    final_pred = "con_casco" if use_helmet else "sin_casco"
    if gemini_conf > 0.90:
      return final_pred, gemini_conf
    return final_pred, (confidence + gemini_conf) / 2.0
  except Exception as e:
    print(f"❌ Error en validación Gemini: {e}. Se usará la predicción original.")
    return initial_prediction, confidence


def run_detector_for_camera(camara: Camara, *, max_frames: Optional[int], frames_por_segundo: int,
                            save_detections: bool, draw_boxes: bool) -> DetectorSummary:
  source_path = _resolve_camera_source(camara)
  channel_layer, room_group_name = get_channel_layer(), f'detect_{camara.numero}'
  config = getattr(settings, "YOLO_CONFIG", {})
  helmet_classes = config.get("helmet_classes", set())
  no_helmet_classes = config.get("no_helmet_classes", set())
  service, summary = HelmetDetectionService(), DetectorSummary()
  cap = cv2.VideoCapture(str(source_path))
  if not cap.isOpened():
    raise DetectorError(f"No se pudo abrir el video: {source_path}")

  video_fps = cap.get(cv2.CAP_PROP_FPS) or 30

  # CAMBIO EN LA INICIALIZACIÓN DEL TRACKER
  tracker = CentroidTracker(
    maxDisappeared=20,  # Reducido de 45 a 20 frames (~0.7s a 30fps)
    minFramesTracked=3,  # Reducido de 5 a 3 frames
    minFrameQuality=50  # Umbral mínimo de sharpness
  )

  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  vid_stride = max(1, round(video_fps / frames_por_segundo))
  limit = max_frames if max_frames else total_frames

  print("🚀 Iniciando Fase 1: Tracking Inteligente y Dibujo en Tiempo Real...")
  frame_idx, processed_count = 0, 0

  while cap.isOpened() and processed_count < limit:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
      break
    processed_count += 1
    frame_idx += vid_stride

    annotated_frame = frame.copy()
    pre_frame, ratio, pad = preprocess_for_yolo(frame, 640)
    results = service.general_model.predict(pre_frame, conf=0.4, verbose=False, classes=[0, 3])
    detections = service.extract_detections(results[0], service.general_model.names)

    for det in detections:
      det.bounding_box = scale_bounding_box(det.bounding_box, frame.shape[:2], ratio, pad)

    persons = [d for d in detections if d.class_name == 'person']
    motorcycles = [d for d in detections if d.class_name == 'motorcycle']
    rects, pairs = [], []

    for m in motorcycles:
      op = [p for p in persons if boxes_overlap(m.bounding_box, p.bounding_box, 0.1)]
      if op:
        x1, y1 = min(m.bounding_box.x1, *[p.bounding_box.x1 for p in op]), min(m.bounding_box.y1,
                                                                               *[p.bounding_box.y1 for p in op])
        x2, y2 = max(m.bounding_box.x2, *[p.bounding_box.x2 for p in op]), max(m.bounding_box.y2,
                                                                               *[p.bounding_box.y2 for p in op])
        rects.append((x1, y1, x2, y2))
        pairs.append(BoundingBox(x1, y1, x2, y2))

    # CAMBIO EN EL UPDATE: Pasar el frame actual
    tracker.update(rects, frame=frame)

    for obj in tracker.objects.values():
      if not any(boxes_overlap(BoundingBox(*obj.rect), p_box, 0.3) for p_box in pairs):
        continue

      x1, y1, x2, y2 = obj.rect
      area = (x2 - x1) * (y2 - y1)

      # CAMBIO CRÍTICO 5: Captura más agresiva
      try:
        crop = frame[int(y1):int(y2), int(x1):int(x2)]
        if crop.size == 0:
          continue

        sharpness = calculate_sharpness(crop)
        quality = (area * 0.7) + (sharpness * 0.3)

        # Captura forzada en los primeros frames
        if obj.best_frame is None and obj.frames_tracked <= 10:
          obj.best_frame_quality = quality
          obj.best_frame = frame.copy()
          obj.best_rect = obj.rect
          print(f"  📸 [Captura Inicial] Moto ID {obj.objectID}: "
                f"frame={obj.frames_tracked}, área={area:.0f}, score={quality:.2f}")
        # Captura mejorada si hay mejor calidad
        elif quality > obj.best_frame_quality:
          obj.best_frame_quality = quality
          obj.best_frame = frame.copy()
          obj.best_rect = obj.rect
          print(f"  📸 [Captura Mejorada] Moto ID {obj.objectID}: "
                f"área={area:.0f}, score={quality:.2f}")
      except Exception as e:
        print(f"  ❌ Error capturando frame para ID {obj.objectID}: {e}")

    if draw_boxes:
      colors = {"person": (255, 182, 8), "motorcycle": (8, 182, 255)}
      for det in detections:
        b = det.bounding_box
        x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)
        c = colors.get(det.class_name, (128, 128, 128))
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), c, 2)
        lbl, f, s, t = f"{det.class_name}", 0, 0.5, 1
        (tw, th), _ = cv2.getTextSize(lbl, f, s, t)
        cv2.rectangle(annotated_frame, (x1, y1), (x1 + tw + 4, y1 - th - 6), c, -1)
        cv2.putText(annotated_frame, lbl, (x1 + 2, y1 - 4), f, s, (255, 255, 255), t)

      for obj in tracker.objects.values():
        if obj.frames_disappeared == 0:
          x1, y1, _, _ = map(int, obj.rect)
          cv2.putText(annotated_frame, f"ID {obj.objectID}", (x1, y1 - 20), 0, 0.6, (0, 255, 0), 2)

    _, buf = cv2.imencode('.jpg', annotated_frame)
    b64 = base64.b64encode(buf).decode('utf-8')
    prog = (processed_count / limit) * 50
    async_to_sync(channel_layer.group_send)(room_group_name, {
      'type': 'detection_update',
      'message': {
        'type': 'frame',
        'progress': prog,
        'image': f'data:image/jpeg;base64,{b64}',
        'status': f'Rastreando... ({len(tracker.objects)} motos activas)'
      }
    })

  cap.release()

  print("\n🔍 Iniciando Fase 2: Clasificación + Veredicto Gemini...")
  ready_to_process = list(tracker.finished_tracks)

  for obj in tracker.objects.values():
    if obj.frames_tracked >= tracker.minFramesTracked and obj.best_frame is not None:
      ready_to_process.append(obj)
      print(f"   Pista activa ID {obj.objectID} añadida al final del video.")

  ready_to_process = list({obj.objectID: obj for obj in ready_to_process}.values())
  print(f"Total de pistas para procesar: {len(ready_to_process)}")

  detecciones_ejecucion, infracciones_ejecucion = [], []

  if save_detections:
    from ..reporting import generar_reporte_infraccion

    for idx, obj in enumerate(ready_to_process):
      if obj.best_frame is None or obj.best_rect is None or obj.already_processed:
        continue
      obj.already_processed = True

      try:
        print(f"\n📋 Procesando Moto ID {obj.objectID} ({idx + 1}/{len(ready_to_process)})")
        crop = extract_motorcycle_crop(obj.best_frame, obj.best_rect)
        results = service.helmet_model.predict(crop, conf=0.5, verbose=False)
        detections = service.extract_detections(results[0], service.helmet_model.names)

        has_h = any(d.class_name in helmet_classes for d in detections)
        has_no_h = any(d.class_name in no_helmet_classes for d in detections)
        init_pred = "con_casco" if has_h and not has_no_h else "sin_casco"
        init_conf = max((d.confidence for d in detections if d.class_name in helmet_classes.union(no_helmet_classes)),
                        default=0.5)

        print(f"  🎯 best.pt dice: {init_pred} ({init_conf:.2f})")
        final_pred, final_conf = validate_with_gemini(crop, init_pred, init_conf)

        if final_pred == "descartado":
          continue

        annotated_final_image = obj.best_frame.copy()
        is_infraction = final_pred == "sin_casco"
        x1, y1, x2, y2 = map(int, obj.best_rect)
        color, label = ((0, 0, 255), f"INFRACTOR (SIN CASCO) - {final_conf:.0%}") if is_infraction else ((0, 255, 0),
                                                                                                         f"CON CASCO - {final_conf:.0%}")

        cv2.rectangle(annotated_final_image, (x1, y1), (x2, y2), color, 3)
        font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        cv2.rectangle(annotated_final_image, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
        cv2.putText(annotated_final_image, label, (x1 + 5, y1 - 5), font, scale, (255, 255, 255), thickness)

        now = timezone.now()
        date_path, filename = now.strftime('%Y/%m/%d'), f"detection_{now.timestamp()}_{obj.objectID}.jpg"
        rel_path = Path('detecciones') / date_path / filename
        full_path = Path(settings.MEDIA_ROOT) / rel_path

        full_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(full_path), annotated_final_image)

        usa_casco_bool = not is_infraction
        summary.detections_without_helmet += 1 if is_infraction else 0
        summary.saved_infractions += 1 if is_infraction else 0
        summary.detections_with_helmet += 1 if not is_infraction else 0

        instance = Deteccion.objects.create(
          camara=camara,
          procesada=True,
          confianza=final_conf * 100,
          usa_casco=usa_casco_bool,
          fecha_hora=now,
          imagen_captura=str(rel_path).replace("\\", "/")
        )

        info = {
          'id': instance.id,
          'confianza': float(final_conf * 100),
          'fecha_hora': now.strftime('%H:%M:%S'),
          'imagen_url': instance.imagen_captura.url,
          'usa_casco': usa_casco_bool
        }

        if is_infraction:
          pdf_path = generar_reporte_infraccion(instance, full_path)
          if pdf_path:
            info['pdf_url'] = str(pdf_path).replace(str(settings.MEDIA_ROOT), '/media').replace('\\', '/')
          infracciones_ejecucion.append(info)

        detecciones_ejecucion.append(info)

      except Exception as e:
        print(f"❌ Error CRÍTICO procesando Moto ID {obj.objectID}: {e}")
        traceback.print_exc()

  print("\n✅ Proceso completado. Calculando resumen final GLOBAL...")
  ventana_24h = timezone.now() - timedelta(hours=24)
  global_detections = Deteccion.objects.filter(fecha_hora__gte=ventana_24h, procesada=True)

  final_summary = {
    'total_24h': global_detections.count(),
    'with_helmet_24h': global_detections.filter(usa_casco=True).count(),
    'without_helmet_24h': global_detections.filter(usa_casco=False).count()
  }

  final_message = {
    'type': 'finished',
    'summary': final_summary,
    'infracciones': infracciones_ejecucion,
    'detecciones': detecciones_ejecucion,
    'progress': 100,
    'status': 'Análisis completado'
  }

  print(f"📦 Enviando mensaje final al cliente: {json.dumps(final_message, indent=2)}")
  async_to_sync(channel_layer.group_send)(room_group_name, {
    'type': 'detection_update',
    'message': final_message
  })

  return summary
#
#
# from __future__ import annotations
#
# import base64
# import json
# import traceback
# from collections import OrderedDict
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Optional, Tuple, Dict
#
# import cv2
# import google.generativeai as genai
# import numpy as np
# from PIL import Image
# from asgiref.sync import async_to_sync
# from channels.layers import get_channel_layer
# from django.conf import settings
# from django.utils import timezone
# from django.utils.timezone import timedelta
# from scipy.spatial import distance as dist
#
# from .service import HelmetDetectionService, BoundingBox
# from ..models import Camara, Deteccion
#
#
# class DetectorError(RuntimeError):
#   pass
#
#
# @dataclass(slots=True)
# class DetectorSummary:
#   frames_processed: int = 0
#   detections_with_helmet: int = 0
#   detections_without_helmet: int = 0
#   saved_infractions: int = 0
#
#
# @dataclass
# class TrackedObject:
#   objectID: int
#   centroid: tuple
#   rect: tuple
#   frames_tracked: int = 0
#   frames_disappeared: int = 0
#   best_frame: np.ndarray | None = None
#   best_rect: tuple | None = None
#   best_frame_area: int = 0
#   best_frame_quality: float = 0.0
#   already_processed: bool = False
#
#
# class CentroidTracker:
#   def __init__(self, maxDisappeared=15, minFramesTracked=5):
#     self.nextObjectID = 0
#     self.objects: Dict[int, TrackedObject] = OrderedDict()
#     self.disappeared = OrderedDict()
#     self.maxDisappeared = maxDisappeared
#     self.minFramesTracked = minFramesTracked
#     self.finished_tracks: list[TrackedObject] = []
#
#   def register(self, centroid, rect):
#     new_id = self.nextObjectID
#     self.objects[new_id] = TrackedObject(
#       objectID=new_id, centroid=centroid, rect=rect, frames_tracked=1
#     )
#     self.disappeared[new_id] = 0
#     self.nextObjectID += 1
#
#   def deregister(self, objectID):
#     obj = self.objects.get(objectID)
#     if obj and obj.frames_tracked >= self.minFramesTracked and obj.best_frame is not None:
#       print(f"  🏁 Pista finalizada para Moto ID {obj.objectID}. Guardada para Fase 2.")
#       self.finished_tracks.append(obj)
#     if objectID in self.objects:
#       del self.objects[objectID]
#     if objectID in self.disappeared:
#       del self.disappeared[objectID]
#
#   def update(self, rects):
#     if len(rects) == 0:
#       for objectID in list(self.disappeared.keys()):
#         self.disappeared[objectID] += 1
#         self.objects[objectID].frames_disappeared += 1
#         if self.disappeared[objectID] > self.maxDisappeared:
#           self.deregister(objectID)
#       return
#
#     inputCentroids = np.array([((r[0] + r[2]) / 2.0, (r[1] + r[3]) / 2.0) for r in rects])
#
#     if len(self.objects) == 0:
#       for i in range(len(rects)):
#         self.register(inputCentroids[i], rects[i])
#     else:
#       objectIDs = list(self.objects.keys())
#       objectCentroids = [o.centroid for o in self.objects.values()]
#       D = dist.cdist(np.array(objectCentroids), inputCentroids)
#       rows, cols = D.min(axis=1).argsort(), D.argmin(axis=1)[D.min(axis=1).argsort()]
#       usedRows, usedCols = set(), set()
#       for row, col in zip(rows, cols):
#         if row in usedRows or col in usedCols: continue
#         objectID = objectIDs[row]
#         self.objects[objectID].centroid = inputCentroids[col]
#         self.objects[objectID].rect = rects[col]
#         self.objects[objectID].frames_tracked += 1
#         self.disappeared[objectID] = self.objects[objectID].frames_disappeared = 0
#         usedRows.add(row), usedCols.add(col)
#
#       unusedRows = set(range(D.shape[0])).difference(usedRows)
#       unusedCols = set(range(D.shape[1])).difference(usedCols)
#       for row in unusedRows:
#         objectID = objectIDs[row]
#         self.disappeared[objectID] += 1
#         self.objects[objectID].frames_disappeared += 1
#         if self.disappeared[objectID] > self.maxDisappeared: self.deregister(objectID)
#       for col in unusedCols: self.register(inputCentroids[col], rects[col])
#
#
# def _resolve_camera_source(camara: Camara) -> Path:
#   if camara.tipo_fuente != "archivo" or not camara.archivo_video:
#     raise DetectorError("La detección solo está disponible para cámaras con un archivo de video local.")
#   relative_path = Path(camara.archivo_video)
#   source_path = Path(settings.BASE_DIR) / "static" / relative_path
#   if not source_path.exists():
#     raise DetectorError(f"El archivo de video para '{camara.nombre}' no existe en la ruta: {source_path}")
#   return source_path
#
#
# def boxes_overlap(box1, box2, threshold=0.5):
#   x1_min, y1_min, x1_max, y1_max = box1 if isinstance(box1, tuple) else (box1.x1, box1.y1, box1.x2, box1.y2)
#   x2_min, y2_min, x2_max, y2_max = box2 if isinstance(box2, tuple) else (box2.x1, box2.y1, box2.x2, box2.y2)
#   inter_xmin = max(x1_min, x2_min);
#   inter_ymin = max(y1_min, y2_min)
#   inter_xmax = min(x1_max, x2_max);
#   inter_ymax = min(y1_max, y2_max)
#   inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
#   box1_area = (x1_max - x1_min) * (y1_max - y1_min);
#   box2_area = (x2_max - x2_min) * (y2_max - y2_min)
#   union_area = box1_area + box2_area - inter_area
#   if union_area == 0: return False
#   return (inter_area / union_area) > threshold
#
#
# def calculate_sharpness(image: np.ndarray) -> float:
#   if image.size == 0: return 0.0
#   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#   gray = cv2.GaussianBlur(gray, (3, 3), 0)
#   return cv2.Laplacian(gray, cv2.CV_64F).var()
#
#
# def preprocess_for_yolo(frame: np.ndarray, target_size: int = 640) -> Tuple[np.ndarray, float, Tuple[int, int]]:
#   orig_h, orig_w = frame.shape[:2];
#   ratio = min(target_size / orig_w, target_size / orig_h)
#   new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)
#   resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
#   padded_frame = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
#   pad_w, pad_h = (target_size - new_w) // 2, (target_size - new_h) // 2
#   padded_frame[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized_frame
#   return padded_frame, ratio, (pad_w, pad_h)
#
#
# def scale_bounding_box(box: BoundingBox, orig_shape: Tuple[int, int], ratio: float,
#                        pad: Tuple[int, int]) -> BoundingBox:
#   pad_w, pad_h = pad;
#   orig_h, orig_w = orig_shape
#   orig_x1 = np.clip((box.x1 - pad_w) / ratio, 0, orig_w);
#   orig_y1 = np.clip((box.y1 - pad_h) / ratio, 0, orig_h)
#   orig_x2 = np.clip((box.x2 - pad_w) / ratio, 0, orig_w);
#   orig_y2 = np.clip((box.y2 - pad_h) / ratio, 0, orig_h)
#   return BoundingBox(x1=orig_x1, y1=orig_y1, x2=orig_x2, y2=orig_y2)
#
#
# def extract_motorcycle_crop(frame: np.ndarray, rect: tuple) -> np.ndarray:
#   x1, y1, x2, y2 = map(int, rect);
#   h, w = frame.shape[:2]
#   center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2;
#   half_size = 320
#   crop_x1, crop_y1 = max(0, center_x - half_size), max(0, center_y - half_size)
#   crop_x2, crop_y2 = min(w, center_x + half_size), min(h, center_y + half_size)
#   crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
#   if crop.shape[:2] != (640, 640):
#     padded = np.full((640, 640, 3), 114, dtype=np.uint8)
#     h_c, w_c = crop.shape[:2];
#     p_y, p_x = (640 - h_c) // 2, (640 - w_c) // 2
#     padded[p_y:p_y + h_c, p_x:p_x + w_c] = crop
#     return padded
#   return crop
#
#
# def validate_with_gemini(crop_image: np.ndarray, initial_prediction: str, confidence: float) -> Tuple[str, float]:
#   if initial_prediction == 'con_casco' and confidence >= 0.88:
#     print(f"   ✅ Confianza alta en 'con_casco' ({confidence:.2f}), aceptando sin Gemini.")
#     return 'con_casco', confidence
#   try:
#     api_key = "AIzaSyC0J0E4vp6sHy7UhDj18hSZxB4v3VNkxnY";
#     genai.configure(api_key=api_key)
#     model = genai.GenerativeModel('gemini-flash-latest')
#     pil_image = Image.fromarray(cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB))
#     prompt = f'Eres un experto en seguridad vial. Analiza esta imagen y da tu veredicto final sobre si el conductor lleva casco. La primera IA clasificó esto como "{initial_prediction}" con confianza {confidence:.2f}. Responde SOLO con un JSON con el formato: {{"presencia_valida": true/false, "usa_casco": true/false, "confianza_veredicto": 0.0-1.0, "razonamiento": "Explica brevemente."}}'
#     response = model.generate_content([prompt, pil_image])
#     result = json.loads(response.text.strip().replace("```json", "").replace("```", "").strip())
#     is_valid, use_helmet, gemini_conf = result.get("presencia_valida", False), result.get("usa_casco", False), float(
#       result.get("confianza_veredicto", 0.0))
#     print(
#       f"\n🤖 Veredicto de Gemini:\n   - ¿Válida?: {'SÍ' if is_valid else 'NO'}\n   - ¿Casco?: {'SÍ' if use_helmet else 'NO'}\n   - Confianza: {gemini_conf:.2f}\n   - Razón: {result.get('razonamiento', '')}")
#     if not is_valid: return "descartado", 0.0
#     final_pred = "con_casco" if use_helmet else "sin_casco"
#     if gemini_conf > 0.90: return final_pred, gemini_conf
#     return final_pred, (confidence + gemini_conf) / 2.0
#   except Exception as e:
#     print(f"❌ Error en validación Gemini: {e}. Se usará la predicción original.")
#     return initial_prediction, confidence
#
#
# def run_detector_for_camera(camara: Camara, *, max_frames: Optional[int], frames_por_segundo: int,
#                             save_detections: bool, draw_boxes: bool) -> DetectorSummary:
#   source_path = _resolve_camera_source(camara)
#   channel_layer, room_group_name = get_channel_layer(), f'detect_{camara.numero}'
#   config = getattr(settings, "YOLO_CONFIG", {})
#   helmet_classes = config.get("helmet_classes", set())
#   no_helmet_classes = config.get("no_helmet_classes", set())
#   service, summary = HelmetDetectionService(), DetectorSummary()
#   cap = cv2.VideoCapture(str(source_path))
#   if not cap.isOpened(): raise DetectorError(f"No se pudo abrir el video: {source_path}")
#   video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
#   tracker = CentroidTracker(maxDisappeared=int(video_fps * 1.5), minFramesTracked=5)
#   total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#   vid_stride = max(1, round(video_fps / frames_por_segundo))
#   limit = max_frames if max_frames else total_frames
#   print("🚀 Iniciando Fase 1: Tracking Inteligente y Dibujo en Tiempo Real...")
#   frame_idx, processed_count = 0, 0
#
#   while cap.isOpened() and processed_count < limit:
#     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx);
#     ret, frame = cap.read()
#     if not ret: break
#     processed_count += 1;
#     frame_idx += vid_stride
#     annotated_frame = frame.copy()
#     pre_frame, ratio, pad = preprocess_for_yolo(frame, 640)
#     results = service.general_model.predict(pre_frame, conf=0.4, verbose=False, classes=[0, 3])
#     detections = service.extract_detections(results[0], service.general_model.names)
#     for det in detections: det.bounding_box = scale_bounding_box(det.bounding_box, frame.shape[:2], ratio, pad)
#     persons = [d for d in detections if d.class_name == 'person'];
#     motorcycles = [d for d in detections if d.class_name == 'motorcycle']
#     rects, pairs = [], []
#     for m in motorcycles:
#       op = [p for p in persons if boxes_overlap(m.bounding_box, p.bounding_box, 0.1)]
#       if op:
#         x1, y1 = min(m.bounding_box.x1, *[p.bounding_box.x1 for p in op]), min(m.bounding_box.y1,
#                                                                                *[p.bounding_box.y1 for p in op])
#         x2, y2 = max(m.bounding_box.x2, *[p.bounding_box.x2 for p in op]), max(m.bounding_box.y2,
#                                                                                *[p.bounding_box.y2 for p in op])
#         rects.append((x1, y1, x2, y2)), pairs.append(BoundingBox(x1, y1, x2, y2))
#     tracker.update(rects)
#
#     for obj in tracker.objects.values():
#       if not any(boxes_overlap(BoundingBox(*obj.rect), p_box, 0.3) for p_box in pairs): continue
#       x1, y1, x2, y2 = obj.rect;
#       area = (x2 - x1) * (y2 - y1)
#       quality = (area * .7) + (calculate_sharpness(frame[int(y1):int(y2), int(x1):int(x2)]) * .3)
#       if quality > obj.best_frame_quality:
#         obj.best_frame_quality, obj.best_frame = quality, frame.copy()
#         obj.best_rect = obj.rect
#         print(f"  📸 [Captura Mejorada] Moto ID {obj.objectID}: área={area:.0f}, score={quality:.2f}")
#
#     if draw_boxes:
#       colors = {"person": (255, 182, 8), "motorcycle": (8, 182, 255)}
#       for det in detections:
#         b = det.bounding_box;
#         x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2);
#         c = colors.get(det.class_name, (128, 128, 128))
#         cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), c, 2)
#         lbl, f, s, t = f"{det.class_name}", 0, 0.5, 1
#         (tw, th), _ = cv2.getTextSize(lbl, f, s, t);
#         cv2.rectangle(annotated_frame, (x1, y1), (x1 + tw + 4, y1 - th - 6), c, -1);
#         cv2.putText(annotated_frame, lbl, (x1 + 2, y1 - 4), f, s, (255, 255, 255), t)
#       for obj in tracker.objects.values():
#         if obj.frames_disappeared == 0: x1, y1, _, _ = map(int, obj.rect); cv2.putText(annotated_frame,
#                                                                                        f"ID {obj.objectID}",
#                                                                                        (x1, y1 - 20), 0, 0.6,
#                                                                                        (0, 255, 0), 2)
#     _, buf = cv2.imencode('.jpg', annotated_frame)
#     b64 = base64.b64encode(buf).decode('utf-8')
#     prog = (processed_count / limit) * 50
#     async_to_sync(channel_layer.group_send)(room_group_name, {'type': 'detection_update',
#                                                               'message': {'type': 'frame', 'progress': prog,
#                                                                           'image': f'data:image/jpeg;base64,{b64}',
#                                                                           'status': f'Rastreando... ({len(tracker.objects)} motos)'}})
#   cap.release()
#
#   print("\n🔍 Iniciando Fase 2: Clasificación + Veredicto Gemini...")
#   ready_to_process = list(tracker.finished_tracks)
#   for obj in tracker.objects.values():
#     if obj.frames_tracked >= tracker.minFramesTracked and obj.best_frame is not None:
#       ready_to_process.append(obj);
#       print(f"   Pista activa ID {obj.objectID} añadida al final del video.")
#   ready_to_process = list({obj.objectID: obj for obj in ready_to_process}.values())
#   print(f"Total de pistas para procesar: {len(ready_to_process)}")
#
#   detecciones_ejecucion, infracciones_ejecucion = [], []
#   if save_detections:
#     from ..reporting import generar_reporte_infraccion
#     for idx, obj in enumerate(ready_to_process):
#       if obj.best_frame is None or obj.best_rect is None or obj.already_processed: continue
#       obj.already_processed = True
#       try:
#         print(f"\n📋 Procesando Moto ID {obj.objectID} ({idx + 1}/{len(ready_to_process)})")
#         crop = extract_motorcycle_crop(obj.best_frame, obj.best_rect)
#         results = service.helmet_model.predict(crop, conf=0.5, verbose=False)
#         detections = service.extract_detections(results[0], service.helmet_model.names)
#         has_h = any(d.class_name in helmet_classes for d in detections);
#         has_no_h = any(d.class_name in no_helmet_classes for d in detections)
#         init_pred = "con_casco" if has_h and not has_no_h else "sin_casco"
#         init_conf = max((d.confidence for d in detections if d.class_name in helmet_classes.union(no_helmet_classes)),
#                         default=0.5)
#         print(f"  🎯 best.pt dice: {init_pred} ({init_conf:.2f})")
#         final_pred, final_conf = validate_with_gemini(crop, init_pred, init_conf)
#         if final_pred == "descartado": continue
#         annotated_final_image = obj.best_frame.copy();
#         is_infraction = final_pred == "sin_casco"
#         x1, y1, x2, y2 = map(int, obj.best_rect)
#         color, label = ((0, 0, 255), f"INFRACTOR (SIN CASCO) - {final_conf:.0%}") if is_infraction else ((0, 255, 0),
#                                                                                                          f"CON CASCO - {final_conf:.0%}")
#         cv2.rectangle(annotated_final_image, (x1, y1), (x2, y2), color, 3)
#         font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
#         (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
#         cv2.rectangle(annotated_final_image, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
#         cv2.putText(annotated_final_image, label, (x1 + 5, y1 - 5), font, scale, (255, 255, 255), thickness)
#         now = timezone.now()
#         date_path, filename = now.strftime('%Y/%m/%d'), f"detection_{now.timestamp()}_{obj.objectID}.jpg"
#
#         ### CORRECCIÓN CRÍTICA ###
#         rel_path = Path('detecciones') / date_path / filename
#         full_path = Path(settings.MEDIA_ROOT) / rel_path
#
#         full_path.parent.mkdir(parents=True, exist_ok=True)
#         cv2.imwrite(str(full_path), annotated_final_image)
#         usa_casco_bool = not is_infraction
#         summary.detections_without_helmet += 1 if is_infraction else 0;
#         summary.saved_infractions += 1 if is_infraction else 0
#         summary.detections_with_helmet += 1 if not is_infraction else 0
#         instance = Deteccion.objects.create(camara=camara, procesada=True, confianza=final_conf * 100,
#                                             usa_casco=usa_casco_bool,
#                                             fecha_hora=now, imagen_captura=str(rel_path).replace("\\", "/"))
#         info = {'id': instance.id, 'confianza': float(final_conf * 100), 'fecha_hora': now.strftime('%H:%M:%S'),
#                 'imagen_url': instance.imagen_captura.url, 'usa_casco': usa_casco_bool}
#         if is_infraction:
#           pdf_path = generar_reporte_infraccion(instance, full_path)
#           if pdf_path: info['pdf_url'] = str(pdf_path).replace(str(settings.MEDIA_ROOT), '/media').replace('\\', '/')
#           infracciones_ejecucion.append(info)
#         detecciones_ejecucion.append(info)
#       except Exception as e:
#         print(f"❌ Error CRÍTICO procesando Moto ID {obj.objectID}: {e}");
#         traceback.print_exc()
#
#   print("\n✅ Proceso completado. Calculando resumen final GLOBAL...")
#   ventana_24h = timezone.now() - timedelta(hours=24)
#   global_detections = Deteccion.objects.filter(fecha_hora__gte=ventana_24h, procesada=True)
#   final_summary = {'total_24h': global_detections.count(),
#                    'with_helmet_24h': global_detections.filter(usa_casco=True).count(),
#                    'without_helmet_24h': global_detections.filter(usa_casco=False).count()}
#   final_message = {'type': 'finished', 'summary': final_summary, 'infracciones': infracciones_ejecucion,
#                    'detecciones': detecciones_ejecucion, 'progress': 100, 'status': 'Análisis completado'}
#   print(f"📦 Enviando mensaje final al cliente: {json.dumps(final_message, indent=2)}")
#   async_to_sync(channel_layer.group_send)(room_group_name, {'type': 'detection_update', 'message': final_message})
#   return summary

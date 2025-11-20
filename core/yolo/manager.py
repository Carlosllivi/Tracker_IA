"""Gestión de hilo en segundo plano para detecciones automáticas."""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from django.apps import apps

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_worker: Optional[threading.Thread] = None
_stop_event: Optional[threading.Event] = None


def is_running() -> bool:
    return _worker is not None and _worker.is_alive()


def start() -> bool:
    """Inicia el hilo del detector si no está en ejecución."""

    from core.models import DetectorConfig  # Importación perezosa

    with _lock:
        config = DetectorConfig.get_solo()
        if not config.activo:
            return False
        if is_running():
            return False

        global _worker, _stop_event
        _stop_event = threading.Event()
        _worker = threading.Thread(target=_run_loop, args=(_stop_event,), daemon=True, name="yolo-detector")
        _worker.start()
        logger.info("Hilo de detección YOLO iniciado")
        return True


def stop(wait: bool = True) -> bool:
    """Solicita la detención del hilo de detección."""

    global _worker, _stop_event
    with _lock:
        if not is_running():
            return False
        assert _stop_event is not None
        _stop_event.set()
        worker = _worker
    if wait and worker is not None:
        worker.join(timeout=10)
    with _lock:
        if _worker and not _worker.is_alive():
            _worker = None
            _stop_event = None
            logger.info("Hilo de detección YOLO detenido")
    return True


def ensure_alignment() -> None:
    """Ajusta el hilo según la configuración actual."""

    from core.models import DetectorConfig

    config = DetectorConfig.get_solo()
    if config.activo:
        start()
    else:
        stop(wait=False)


def _run_loop(stop_event: threading.Event) -> None:
    Camara = apps.get_model("core", "Camara")
    Config = apps.get_model("core", "DetectorConfig")

    from .pipeline import DetectorError, run_detector_for_camera

    logger.debug("Bucle del detector en ejecución")

    while not stop_event.is_set():
        try:
            config = Config.get_solo()
        except Exception as exc:  # pragma: no cover - errores raros de base
            logger.exception("No se pudo obtener la configuración del detector: %s", exc)
            if stop_event.wait(timeout=10):
                break
            continue

        if not config.activo:
            if stop_event.wait(timeout=5):
                break
            continue

        cameras = (
            Camara.objects.filter(
                habilitada=True,
                tipo_fuente="archivo",
            )
            .exclude(archivo_video="")
            .order_by("numero")
        )

        if not cameras.exists():
            logger.debug("No hay cámaras con archivo para procesar")
            if stop_event.wait(timeout=max(config.intervalo_segundos, 5)):
                break
            continue

        for camara in cameras:
            if stop_event.is_set():
                break

            # Comprobar que la configuración sigue activa en cada iteración
            refreshed = Config.get_solo()
            if not refreshed.activo:
                break

            try:
                logger.info("Procesando cámara %s (%s)", camara.numero, camara.nombre)
                run_detector_for_camera(
                    camara,
                    max_frames=refreshed.max_frames,
                    vid_stride=refreshed.vid_stride,
                    save_media=refreshed.guardar_anotaciones,
                )
            except DetectorError as exc:
                logger.warning("Error de detector en cámara %s: %s", camara.numero, exc)
            except Exception:  # pragma: no cover - protegerse de fallos imprevistos
                logger.exception("Fallo inesperado ejecutando detección en cámara %s", camara.numero)

            if stop_event.wait(timeout=1):
                break

        if stop_event.wait(timeout=max(config.intervalo_segundos, 5)):
            break

    logger.debug("Bucle del detector finalizado")

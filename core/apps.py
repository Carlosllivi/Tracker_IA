from django.apps import AppConfig


class CoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'core'

    def ready(self):
        # Importar señales para inicializar grupos y permisos predeterminados
        try:
            import core.signals  # noqa: F401
        except ImportError:
            # Si las señales no existen aún, no interrumpir el proceso de inicio
            pass

        # Conectar señal para alinear el detector tras el arranque, evitando consultar BD durante ready()
        from django.core.signals import request_started

        def _align_detector(*args, **kwargs):  # pragma: no cover - función de inicialización
            try:
                from django.db.utils import OperationalError, ProgrammingError
                from .models import DetectorConfig
                from .yolo import manager as detector_manager

                try:
                    config = DetectorConfig.get_solo()
                except (OperationalError, ProgrammingError):
                    return

                if config.activo:
                    detector_manager.start()
            except Exception:
                # Evitar que cualquier fallo bloquee el ciclo de peticiones
                pass

        request_started.connect(_align_detector, dispatch_uid="core.detector.align")

from django.conf import settings  # <-- AGREGAR ESTA LÍNEA
from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils import timezone
from .storages import ProfileImageStorage


def _default_detector_config():
  return {
    "activo": False,
    "guardar_anotaciones": False,
    "max_frames": 200,
    "vid_stride": 5,
    "intervalo_segundos": 60,
  }


class Usuario(AbstractUser):
  """
  Modelo extendido de usuario para el sistema
  """
  ROLES = [
    ('admin', 'Administrador'),
    ('operador', 'Operador'),
    ('visualizador', 'Visualizador'),
  ]

  rol = models.CharField(
    max_length=20,
    choices=ROLES,
    default='operador',
    verbose_name="Rol"
  )
  telefono = models.CharField(
    max_length=15,
    blank=True,
    null=True,
    verbose_name="Teléfono"
  )
  profile_image = models.ImageField(
    upload_to='profiles/',
    blank=True,
    null=True,
    verbose_name="Imagen de Perfil",
    help_text="Imagen de avatar o foto de perfil del usuario",
    storage=ProfileImageStorage()  # <--- ESTA ES LA EXCEPCIÓN
  )

  fecha_registro = models.DateTimeField(
    auto_now_add=True,
    verbose_name="Fecha de Registro"
  )
  activo = models.BooleanField(
    default=True,
    verbose_name="Activo"
  )

  class Meta:
    verbose_name = "Usuario"
    verbose_name_plural = "Usuarios"

  def __str__(self):
    return f"{self.get_full_name()} ({self.username})"


class Camara(models.Model):
  """
  Modelo para gestionar las cámaras del sistema
  RF-05, RF-06, RF-07
  """
  TIPOS_FUENTE = [
    ("archivo", "Archivo de video"),
    ("cameraLive", "Cámara en vivo")
  ]
  ESTADOS = [
    ('activa', 'Activa'),
    ('inactiva', 'Inactiva'),
    ('mantenimiento', 'Mantenimiento'),
  ]

  numero = models.AutoField(primary_key=True, verbose_name="Número de Cámara")
  nombre = models.CharField(max_length=200, verbose_name="Nombre de la Cámara")
  tipo_fuente = models.CharField(
    max_length=20,
    choices=TIPOS_FUENTE,
    default="stream",
    verbose_name="Tipo de fuente de video"
  )
  url_streaming = models.URLField(
    max_length=500,
    blank=True,
    null=True,
    verbose_name="URL de Streaming"
  )
  archivo_video = models.CharField(
    max_length=500,
    blank=True,
    verbose_name="Ruta de archivo de video"
  )
  ubicacion = models.CharField(
    max_length=300,
    blank=True,
    verbose_name="Ubicación"
  )
  estado = models.CharField(
    max_length=20,
    choices=ESTADOS,
    default='activa',
    verbose_name="Estado"
  )
  habilitada = models.BooleanField(default=True, verbose_name="Habilitada")
  fecha_instalacion = models.DateField(
    default=timezone.now,
    verbose_name="Fecha de Instalación"
  )
  fecha_creacion = models.DateTimeField(
    auto_now_add=True,
    verbose_name="Fecha de Creación"
  )
  fecha_actualizacion = models.DateTimeField(
    auto_now=True,
    verbose_name="Última Actualización"
  )
  usuario_creador = models.ForeignKey(
    settings.AUTH_USER_MODEL,  # <-- CAMBIAR AQUÍ: User -> settings.AUTH_USER_MODEL
    on_delete=models.SET_NULL,
    null=True,
    related_name='camaras_creadas',
    verbose_name="Usuario Creador"
  )

  class Meta:
    verbose_name = "Cámara"
    verbose_name_plural = "Cámaras"
    ordering = ['numero']

  def __str__(self):
    return f"{self.numero} - {self.nombre}"

  @property
  def usa_streaming(self):
    """Retorna True si la cámara utiliza una URL de streaming."""
    return self.tipo_fuente == "stream"

  @property
  def ruta_video_local(self):
    """Devuelve la ruta relativa del video local si corresponde."""
    if self.tipo_fuente == "archivo" and self.archivo_video:
      return self.archivo_video
    return ""


class Deteccion(models.Model):
  """
  Modelo para registrar cada detección de motociclista
  RF-03, RF-04
  """
  camara = models.ForeignKey(
    Camara,
    on_delete=models.CASCADE,
    related_name='detecciones',
    verbose_name="Cámara"
  )
  fecha_hora = models.DateTimeField(
    default=timezone.now,
    verbose_name="Fecha y Hora"
  )
  usa_casco = models.BooleanField(default=True, verbose_name="Usa Casco")
  confianza = models.DecimalField(
    max_digits=5,
    decimal_places=2,
    help_text="Porcentaje de confianza de la detección (0-100)",
    verbose_name="Confianza (%)"
  )
  imagen_captura = models.ImageField(
    upload_to='detecciones/%Y/%m/%d/',
    blank=True,
    null=True,
    verbose_name="Imagen de Captura"
  )
  procesada = models.BooleanField(
    default=False,
    verbose_name="Procesada"
  )

  class Meta:
    verbose_name = "Detección"
    verbose_name_plural = "Detecciones"
    ordering = ['-fecha_hora']
    indexes = [
      models.Index(fields=['-fecha_hora']),
      models.Index(fields=['camara', '-fecha_hora']),
    ]

  def __str__(self):
    estado = "CON casco" if self.usa_casco else "SIN casco"
    return f"{self.camara.nombre} - {estado} - {self.fecha_hora.strftime('%d/%m/%Y %H:%M')}"

  @property
  def es_infractor(self):
    """Retorna True si el motociclista no usa casco"""
    return not self.usa_casco


class ReporteDiario(models.Model):
  """
  Modelo para almacenar estadísticas diarias por cámara
  RF-08, RF-09
  """
  camara = models.ForeignKey(
    Camara,
    on_delete=models.CASCADE,
    related_name='reportes',
    verbose_name="Cámara"
  )
  fecha = models.DateField(verbose_name="Fecha")
  total_vehiculos = models.IntegerField(
    default=0,
    verbose_name="Total de Vehículos"
  )
  total_infractores = models.IntegerField(
    default=0,
    verbose_name="Total de Infractores"
  )
  total_con_casco = models.IntegerField(
    default=0,
    verbose_name="Total con Casco"
  )
  porcentaje_cumplimiento = models.DecimalField(
    max_digits=5,
    decimal_places=2,
    default=0,
    verbose_name="% Cumplimiento"
  )
  generado = models.BooleanField(
    default=False,
    verbose_name="Reporte Generado"
  )
  archivo_pdf = models.FileField(
    upload_to='reportes/%Y/%m/',
    blank=True,
    null=True,
    verbose_name="Archivo PDF"
  )
  fecha_generacion = models.DateTimeField(
    auto_now_add=True,
    verbose_name="Fecha de Generación"
  )

  class Meta:
    verbose_name = "Reporte Diario"
    verbose_name_plural = "Reportes Diarios"
    ordering = ['-fecha']
    unique_together = ['camara', 'fecha']
    indexes = [
      models.Index(fields=['fecha', 'camara']),
    ]

  def __str__(self):
    return f"Reporte {self.camara.nombre} - {self.fecha.strftime('%d/%m/%Y')}"

  def calcular_estadisticas(self):
    """Calcula las estadísticas del día"""
    if self.total_vehiculos > 0:
      self.porcentaje_cumplimiento = (
                                       self.total_con_casco / self.total_vehiculos
                                     ) * 100
    else:
      self.porcentaje_cumplimiento = 0
    self.save()


class Alerta(models.Model):
  """
  Modelo para gestionar alertas en tiempo real
  RF-04
  """
  TIPOS_ALERTA = [
    ('infraccion', 'Infracción Detectada'),
    ('camara_offline', 'Cámara Fuera de Línea'),
    ('sistema', 'Alerta de Sistema'),
  ]

  PRIORIDADES = [
    ('baja', 'Baja'),
    ('media', 'Media'),
    ('alta', 'Alta'),
  ]

  tipo = models.CharField(
    max_length=20,
    choices=TIPOS_ALERTA,
    verbose_name="Tipo de Alerta"
  )
  prioridad = models.CharField(
    max_length=10,
    choices=PRIORIDADES,
    default='media',
    verbose_name="Prioridad"
  )
  camara = models.ForeignKey(
    Camara,
    on_delete=models.CASCADE,
    related_name='alertas',
    null=True,
    blank=True,
    verbose_name="Cámara"
  )
  deteccion = models.ForeignKey(
    Deteccion,
    on_delete=models.CASCADE,
    null=True,
    blank=True,
    related_name='alertas',
    verbose_name="Detección"
  )
  mensaje = models.TextField(verbose_name="Mensaje")
  fecha_hora = models.DateTimeField(
    auto_now_add=True,
    verbose_name="Fecha y Hora"
  )
  leida = models.BooleanField(default=False, verbose_name="Leída")
  atendida = models.BooleanField(default=False, verbose_name="Atendida")
  usuario_atencion = models.ForeignKey(
    settings.AUTH_USER_MODEL,
    on_delete=models.SET_NULL,
    null=True,
    blank=True,
    related_name='alertas_atendidas',
    verbose_name="Usuario que Atendió"
  )

  class Meta:
    verbose_name = "Alerta"
    verbose_name_plural = "Alertas"
    ordering = ['-fecha_hora']

  def __str__(self):
    return f"{self.get_tipo_display()} - {self.fecha_hora.strftime('%d/%m/%Y %H:%M')}"


class DetectorConfig(models.Model):
  """Configuración global del motor de detección."""

  activo = models.BooleanField(default=False, verbose_name="Detección activa")
  guardar_anotaciones = models.BooleanField(
    default=False,
    verbose_name="Guardar detecciones y generar reportes",
    help_text="Si está activo, guarda frames de infractores y genera reportes automáticos"
  )

  # Parámetros de calidad y velocidad
  frames_por_segundo = models.PositiveSmallIntegerField(
    default=10,
    verbose_name="Frames por segundo a procesar",
    help_text="Mayor valor = más preciso pero más lento. Recomendado: 8-12 fps"
  )

  max_frames = models.PositiveIntegerField(
    default=300,
    verbose_name="Duración máxima (en frames)",
    help_text="Límite de frames totales a procesar por video. 0 = sin límite"
  )

  confianza_minima = models.DecimalField(
    max_digits=3,
    decimal_places=2,
    default=0.45,
    verbose_name="Confianza mínima (%)",
    help_text="Solo guardar detecciones con confianza superior a este valor (0.00-1.00)"
  )

  validar_contexto_moto = models.BooleanField(
    default=True,
    verbose_name="Validar contexto de motocicleta",
    help_text="Solo guardar infractores que estén montados en una motocicleta"
  )

  filtrar_frames_borrosos = models.BooleanField(
    default=True,
    verbose_name="Filtrar frames borrosos",
    help_text="Descartar frames con poca nitidez"
  )

  # Parámetros de procesamiento continuo
  intervalo_segundos = models.PositiveIntegerField(
    default=60,
    verbose_name="Intervalo entre ciclos (segundos)",
    help_text="Tiempo de espera antes de procesar nuevamente (modo continuo)"
  )

  # Tracking visual
  dibujar_recuadros = models.BooleanField(
    default=True,
    verbose_name="Dibujar recuadros de seguimiento",
    help_text="Marcar infractores (rojo) y cumplidores (verde) en el video"
  )

  actualizado = models.DateTimeField(auto_now=True, verbose_name="Última actualización")

  class Meta:
    verbose_name = "Configuración del Detector"
    verbose_name_plural = "Configuración del Detector"

  def __str__(self):
    return "Configuración global del detector"

  @classmethod
  def get_solo(cls):
    defaults = {
      "activo": False,
      "guardar_anotaciones": False,
      "frames_por_segundo": 10,
      "max_frames": 300,
      "confianza_minima": 0.45,
      "validar_contexto_moto": True,
      "filtrar_frames_borrosos": True,
      "intervalo_segundos": 60,
      "dibujar_recuadros": True,
    }
    obj, _ = cls.objects.get_or_create(pk=1, defaults=defaults)
    return obj

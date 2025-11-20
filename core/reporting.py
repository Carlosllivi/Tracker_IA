from __future__ import annotations

from dataclasses import dataclass
from datetime import date as _date
from io import BytesIO
from pathlib import Path

from django.conf import settings
from django.core.files import File
from django.utils import timezone
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, Image

from .models import Camara, Deteccion, ReporteDiario


# ==============================================================================
# NUEVA FUNCIÓN PARA REPORTES DE INFRACCIONES INDIVIDUALES
# ==============================================================================
def generar_reporte_infraccion(deteccion: Deteccion, image_path: Path) -> Path | None:
  """
  Genera un reporte en PDF profesional para una única infracción detectada.

  Args:
      deteccion: Instancia de Deteccion con la infracción
      image_path: Ruta del archivo de imagen con las anotaciones

  Returns:
      Path del PDF generado o None si hubo error
  """
  try:
    # 1. Preparar la ruta de salida para el PDF
    media_root = Path(settings.MEDIA_ROOT)
    fecha = timezone.localtime(deteccion.fecha_hora)
    out_dir = media_root / "reportes_infracciones" / f"{fecha.year:04d}" / f"{fecha.month:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"infraccion_{deteccion.id}_{fecha.strftime('%Y%m%d_%H%M%S')}.pdf"
    destination = out_dir / filename

    # 2. Crear el contenido del PDF con diseño profesional
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = []

    # ===== ENCABEZADO =====
    title = Paragraph(
      "<b>REPORTE DE INFRACCIÓN DE TRÁNSITO</b>",
      styles['Title']
    )
    story.append(title)
    story.append(Spacer(1, 10))

    subtitle = Paragraph(
      "<b>Conductor sin casco de protección</b>",
      styles['Heading2']
    )
    story.append(subtitle)
    story.append(Spacer(1, 20))

    # ===== IMAGEN DE EVIDENCIA =====
    story.append(Paragraph("<b>EVIDENCIA FOTOGRÁFICA</b>", styles['Heading3']))
    story.append(Spacer(1, 10))

    if image_path.exists():
      img = Image(str(image_path), width=480, height=360)
      img.hAlign = 'CENTER'
      story.append(img)
      story.append(Spacer(1, 15))
    else:
      story.append(Paragraph("⚠️ Imagen de evidencia no disponible", styles['Normal']))
      story.append(Spacer(1, 15))

    # ===== DETALLES DE LA INFRACCIÓN =====
    story.append(Paragraph("<b>DETALLES DE LA INFRACCIÓN</b>", styles['Heading3']))
    story.append(Spacer(1, 10))

    details = [
      ["Campo", "Información"],
      ["ID de Detección:", str(deteccion.id)],
      ["Fecha:", fecha.strftime('%d/%m/%Y')],
      ["Hora:", fecha.strftime('%H:%M:%S')],
      ["Cámara:", f"#{deteccion.camara.numero} - {deteccion.camara.nombre}"],
      ["Ubicación:", deteccion.camara.ubicacion or "No especificada"],
      ["Tipo de Infracción:", "Conductor sin casco de protección"],
      ["Nivel de Confianza:", f"{deteccion.confianza:.2f}%"],
      ["Estado:", "Procesada ✓" if deteccion.procesada else "Pendiente"],
    ]

    table = Table(details, colWidths=[150, 320], hAlign="CENTER")
    table.setStyle(TableStyle([
      ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1173d4")),
      ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
      ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
      ('FONTSIZE', (0, 0), (-1, 0), 11),
      ('BACKGROUND', (0, 1), (0, -1), colors.HexColor("#e7edf3")),
      ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
      ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
      ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
      ('GRID', (0, 0), (-1, -1), 1, colors.grey),
      ('PADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(table)
    story.append(Spacer(1, 20))

    # ===== MARCO LEGAL =====
    story.append(Paragraph("<b>MARCO LEGAL</b>", styles['Heading3']))
    story.append(Spacer(1, 8))
    story.append(
      Paragraph(
        "La conducción de motocicletas sin casco de protección constituye una infracción "
        "a las normativas de tránsito vigentes y representa un riesgo para la seguridad del conductor.",
        styles['Normal']
      )
    )
    story.append(Spacer(1, 20))

    # ===== PIE DE PÁGINA =====
    footer_data = [
      ["Generado automáticamente por:", "Sistema TRACKER IA"],
      ["Fecha de generación:", timezone.now().strftime('%d/%m/%Y %H:%M:%S')],
    ]
    footer_table = Table(footer_data, colWidths=[180, 290], hAlign="CENTER")
    footer_table.setStyle(TableStyle([
      ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
      ('FONTSIZE', (0, 0), (-1, -1), 8),
      ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
      ('TEXTCOLOR', (0, 0), (-1, -1), colors.grey),
    ]))
    story.append(footer_table)

    # 5. Construir y guardar el PDF
    doc.build(story)

    with open(destination, "wb") as f:
      f.write(buffer.getvalue())

    print(f"✅ Reporte de infracción generado y guardado en: {destination}")
    return destination

  except Exception as e:
    print(f"❌ Error crítico al generar el PDF de la infracción: {e}")
    import traceback
    traceback.print_exc()
    return None


# ==============================================================================
# CÓDIGO EXISTENTE PARA REPORTES DIARIOS (SIN CAMBIOS)
# ==============================================================================
@dataclass(slots=True)
class DailyCameraStats:
  camera: Camara
  numero: str
  nombre: str
  total: int
  con_casco: int
  sin_casco: int


@dataclass(slots=True)
class DailyReportResult:
  report_date: _date
  output_path: Path
  stats: list[DailyCameraStats]
  total_general: int
  total_con_casco: int
  total_sin_casco: int


def gather_daily_stats(report_date: _date) -> list[DailyCameraStats]:
  """Agrupa las detecciones por cámara en la fecha indicada."""
  cameras = Camara.objects.all().order_by("numero")
  stats: list[DailyCameraStats] = []

  for cam in cameras:
    qs = Deteccion.objects.filter(
      fecha_hora__date=report_date,
      camara=cam,
      procesada=True  # IMPORTANTE: Solo contar detecciones procesadas
    )
    total = qs.count()
    sin_casco = qs.filter(usa_casco=False).count()
    con_casco = total - sin_casco

    stats.append(
      DailyCameraStats(
        camera=cam,
        numero=str(cam.numero),
        nombre=cam.nombre,
        total=total,
        con_casco=con_casco,
        sin_casco=sin_casco,
      )
    )

  return stats


def _prepare_output_path(report_date: _date) -> Path:
  media_root = Path(settings.MEDIA_ROOT)
  out_dir = media_root / "reportes" / f"{report_date.year:04d}" / f"{report_date.month:02d}"
  out_dir.mkdir(parents=True, exist_ok=True)
  filename = f"reporte_{report_date.strftime('%Y%m%d')}.pdf"
  return out_dir / filename


def _build_pdf(report_date: _date, stats: list[DailyCameraStats], destination: Path) -> None:
  """
  Construye el PDF del reporte diario con diseño profesional e imágenes de evidencia.
  """
  buffer = BytesIO()
  doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=30, bottomMargin=30)
  styles = getSampleStyleSheet()
  story = []

  total_general = sum(item.total for item in stats)
  total_con_casco = sum(item.con_casco for item in stats)
  total_sin_casco = sum(item.sin_casco for item in stats)
  cumplimiento = (total_con_casco / total_general * 100) if total_general else 0

  # ===== ENCABEZADO PROFESIONAL =====
  title = Paragraph(
    f"<b>REPORTE DIARIO DE DETECCIONES</b>",
    styles["Title"]
  )
  story.append(title)
  story.append(Spacer(1, 6))

  subtitle = Paragraph(
    f"Fecha: {report_date.strftime('%d/%m/%Y')} | Sistema TRACKER IA",
    styles["Normal"]
  )
  story.append(subtitle)
  story.append(Spacer(1, 20))

  # ===== RESUMEN EJECUTIVO =====
  story.append(Paragraph("<b>RESUMEN EJECUTIVO</b>", styles["Heading2"]))
  story.append(Spacer(1, 8))

  summary_data = [
    ["Métrica", "Valor"],
    ["Total de detecciones", str(total_general)],
    ["Conductores con casco",
     f"{total_con_casco} ({(total_con_casco / total_general * 100):.1f}%)" if total_general else "0"],
    ["Infractores (sin casco)",
     f"{total_sin_casco} ({(total_sin_casco / total_general * 100):.1f}%)" if total_general else "0"],
    ["Cumplimiento normativo", f"{cumplimiento:.2f}%"],
  ]

  summary_table = Table(summary_data, colWidths=[250, 200], hAlign="CENTER")
  summary_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1173d4")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, 0), 12),
    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f6f7f8")),
    ("GRID", (0, 0), (-1, -1), 1, colors.grey),
    ("PADDING", (0, 0), (-1, -1), 10),
  ]))
  story.append(summary_table)
  story.append(Spacer(1, 20))

  # ===== DESGLOSE POR CÁMARA =====
  story.append(Paragraph("<b>DESGLOSE POR CÁMARA</b>", styles["Heading2"]))
  story.append(Spacer(1, 8))

  camera_table_data = [["#", "Cámara", "Total", "Con casco", "Infractores", "% Cumplimiento"]]
  for item in stats:
    cumpl_cam = (item.con_casco / item.total * 100) if item.total else 0
    camera_table_data.append([
      item.numero,
      item.nombre,
      str(item.total),
      str(item.con_casco),
      str(item.sin_casco),
      f"{cumpl_cam:.1f}%"
    ])

  camera_table = Table(camera_table_data, colWidths=[30, 140, 60, 80, 80, 90], hAlign="CENTER")
  camera_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1173d4")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("ALIGN", (0, 0), (-1, 0), "CENTER"),
    ("ALIGN", (2, 1), (-1, -1), "CENTER"),
    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f6f7f8")),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("PADDING", (0, 0), (-1, -1), 8),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
  ]))
  story.append(camera_table)
  story.append(Spacer(1, 20))

  # ===== EVIDENCIA FOTOGRÁFICA DE INFRACCIONES =====
  story.append(Paragraph("<b>EVIDENCIA FOTOGRÁFICA DE INFRACCIONES</b>", styles["Heading2"]))
  story.append(Spacer(1, 8))

  # Buscar infracciones del día
  infracciones = Deteccion.objects.filter(
    fecha_hora__date=report_date,
    procesada=True,
    usa_casco=False
  ).select_related('camara')[:6]  # Máximo 6 imágenes para no saturar el PDF

  if infracciones.exists():
    for infraccion in infracciones:
      # Información de la infracción
      info_text = f"<b>Infracción #{infraccion.id}</b> | Cámara: {infraccion.camara.nombre} | Hora: {infraccion.fecha_hora.strftime('%H:%M:%S')} | Confianza: {infraccion.confianza:.1f}%"
      story.append(Paragraph(info_text, styles["Normal"]))
      story.append(Spacer(1, 6))

      # Intentar cargar la imagen
      if infraccion.imagen_captura:
        try:
          imagen_path = Path(settings.MEDIA_ROOT) / infraccion.imagen_captura.name
          if imagen_path.exists():
            img = Image(str(imagen_path), width=450, height=300)
            img.hAlign = 'CENTER'
            story.append(img)
          else:
            story.append(Paragraph(f"⚠️ Imagen no encontrada: {imagen_path}", styles["Normal"]))
        except Exception as e:
          story.append(Paragraph(f"⚠️ Error al cargar imagen: {str(e)}", styles["Normal"]))
      else:
        story.append(Paragraph("⚠️ Sin imagen de evidencia", styles["Normal"]))

      story.append(Spacer(1, 15))
  else:
    story.append(Paragraph("✅ No se registraron infracciones en esta fecha.", styles["Normal"]))
    story.append(Spacer(1, 10))

  # ===== PIE DE PÁGINA =====
  story.append(Spacer(1, 20))
  footer = Paragraph(
    f"",
    styles["Normal"]
  )
  story.append(footer)

  # Construir el documento
  doc.build(story)

  # Guardar en disco
  with open(destination, "wb") as output_file:
    output_file.write(buffer.getvalue())


def generate_daily_report(report_date: _date) -> DailyReportResult:
  """
  Genera el PDF y actualiza/crea registros ReporteDiario para la fecha indicada.

  IMPORTANTE: Crea UN SOLO registro de ReporteDiario por día (no por cámara),
  pero el PDF contiene el desglose de todas las cámaras.

  Raises:
      ValueError: Si no hay cámaras o no hay datos para generar el reporte
  """
  stats = gather_daily_stats(report_date)
  if not stats:
    raise ValueError("No hay cámaras registradas para generar reportes.")

  # Verificar que haya al menos una detección
  total_detecciones = sum(item.total for item in stats)
  if total_detecciones == 0:
    raise ValueError(
      f"No hay detecciones procesadas para la fecha {report_date.strftime('%d/%m/%Y')}. "
      "Ejecuta primero el proceso de detección."
    )

  output_path = _prepare_output_path(report_date)
  _build_pdf(report_date, stats, output_path)

  # Calcular totales del día
  total_general = sum(item.total for item in stats)
  total_con_casco = sum(item.con_casco for item in stats)
  total_sin_casco = sum(item.sin_casco for item in stats)
  cumplimiento_general = (total_con_casco / total_general * 100) if total_general else 0

  # CAMBIO CRÍTICO: Crear UN SOLO reporte diario (sin cámara específica)
  # Usamos la primera cámara como referencia o None
  camara_principal = stats[0].camera if stats else None

  reporte_obj, _created = ReporteDiario.objects.update_or_create(
    fecha=report_date,
    camara=camara_principal,  # Puedes usar None si modificas el modelo
    defaults={
      "total_vehiculos": total_general,
      "total_infractores": total_sin_casco,
      "total_con_casco": total_con_casco,
      "porcentaje_cumplimiento": cumplimiento_general,
      "generado": True,
    },
  )

  # Guardar el archivo PDF en el modelo
  with open(output_path, "rb") as fh:
    django_file = File(fh)
    subpath = f"reportes/{report_date.year:04d}/{report_date.month:02d}/{output_path.name}"
    reporte_obj.archivo_pdf.save(subpath, django_file, save=True)

  return DailyReportResult(
    report_date=report_date,
    output_path=output_path,
    stats=stats,
    total_general=total_general,
    total_con_casco=total_con_casco,
    total_sin_casco=total_sin_casco,
  )
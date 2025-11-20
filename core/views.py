import json
import threading
from collections import OrderedDict
from datetime import datetime, timedelta

from django.contrib import messages
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required, permission_required
from django.contrib.auth.models import Group, Permission
from django.core.exceptions import PermissionDenied
from django.core.serializers.json import DjangoJSONEncoder
from django.db import transaction
from django.db.utils import OperationalError, ProgrammingError
from django.http import JsonResponse, HttpResponseNotAllowed
from django.shortcuts import get_object_or_404
from django.shortcuts import render, redirect
from django.utils import timezone
from django.views.decorators.http import require_POST

from .forms import ProfileForm
from .forms import RegistroForm, LoginForm, CamaraForm
from .models import Alerta, Camara, DetectorConfig, Deteccion, ReporteDiario, Usuario
from .reporting import generate_daily_report
from .yolo import manager as detector_manager

MANAGED_PERMISSION_CODES = [
  "view_camara",
  "add_camara",
  "change_camara",
  "delete_camara",
  "view_reportediario",
  "add_reportediario",
  "change_reportediario",
  "delete_reportediario",
  "view_alerta",
  "add_alerta",
  "change_alerta",
  "delete_alerta",
  "view_deteccion",
  "add_deteccion",
  "change_deteccion",
  "delete_deteccion",
  "view_detectorconfig",
  "change_detectorconfig",
]

ROLE_GROUP_MAP = {
  "admin": "Administrador",
  "operador": "Operador",
  "visualizador": "Visualizador",
}

GROUP_LEVEL_INFO = {
  "Administrador": {"level": "alto", "label": "Nivel de permisos Alto"},
  "Operador": {"level": "medio", "label": "Nivel de permisos Medio"},
  "Visualizador": {"level": "bajo", "label": "Nivel de permisos Bajo"},
}

PERMISSION_LEVEL_MAP = {
  "alto": {
    "add_camara",
    "change_camara",
    "delete_camara",
    "add_reportediario",
    "change_reportediario",
    "delete_reportediario",
    "add_alerta",
    "change_alerta",
    "delete_alerta",
    "add_deteccion",
    "change_deteccion",
    "delete_deteccion",
    "change_detectorconfig",
  },
  "medio": {
    "view_alerta",
  },
  "bajo": {
    "view_camara",
    "view_reportediario",
    "view_deteccion",
    "view_detectorconfig",
  },
}

PERMISSION_LEVEL_LABELS = {
  "alto": "Permisos Alto (Administrador)",
  "medio": "Permisos Medio (Operador)",
  "bajo": "Permisos Bajo (Visualizador)",
}


# ==================== VISTAS DE AUTENTICACIÓN ====================


def registro_view(request):
  """
  Vista para el registro de nuevos usuarios
  """
  if request.user.is_authenticated:
    return redirect("panel")

  if request.method == "POST":
    form = RegistroForm(request.POST, request.FILES)
    if form.is_valid():
      user = form.save()

      if user.profile_image:
        print("📸 Imagen subida a S3:")
        print(user.profile_image.url)

      username = form.cleaned_data.get("username")
      messages.success(
        request,
        f"Cuenta creada exitosamente para {username}. ¡Ya puedes iniciar sesión!",
      )
      return redirect("login")
    else:
      for field, errors in form.errors.items():
        for error in errors:
          messages.error(request, f"{error}")
  else:
    form = RegistroForm()

  return render(request, "core/registrarse.html", {"form": form})


def login_view(request):
  """
  Vista para el inicio de sesión
  """
  if request.user.is_authenticated:
    return redirect("panel")

  if request.method == "POST":
    form = LoginForm(request, data=request.POST)
    if form.is_valid():
      username = form.cleaned_data.get("username")
      password = form.cleaned_data.get("password")
      user = authenticate(username=username, password=password)

      if user is not None:
        login(request, user)
        messages.success(
          request, f"Bienvenido, {user.get_full_name() or user.username}!"
        )

        # Redirigir según el parámetro 'next' o al panel por defecto
        next_url = request.GET.get("next", "panel")
        return redirect(next_url)
      else:
        messages.error(request, "Usuario o contraseña incorrectos.")
    else:
      messages.error(request, "Usuario o contraseña incorrectos.")
  else:
    form = LoginForm()

  return render(request, "core/login.html", {"form": form})


@login_required
def logout_view(request):
  """
  Vista para cerrar sesión
  """
  logout(request)
  messages.info(request, "Has cerrado sesión exitosamente.")
  return redirect("login")


@login_required
def perfil_view(request):
  """Vista para que el usuario vea/edite su perfil (avatar y datos básicos)."""
  if request.method == "POST":
    form = ProfileForm(request.POST, request.FILES, instance=request.user)
    if form.is_valid():
      user = form.save()

      if user.profile_image:
        print("🖼️ Imagen actualizada en S3:")
        print(user.profile_image.url)

      messages.success(request, "Perfil actualizado correctamente.")
      return redirect("panel")
    else:
      messages.error(request, "Por favor corrige los errores en el formulario.")
  else:
    form = ProfileForm(instance=request.user)

  return render(request, "core/perfil.html", {"form": form})


# ==================== VISTAS PRINCIPALES ====================


@login_required
@permission_required("core.view_camara", raise_exception=True)
def panel_view(request):
  """
  Vista del panel de control (Dashboard)
  Muestra las cámaras activas y su estado, AHORA TAMBIÉN CON INFRACCIONES RECIENTES.
  """
  camaras = Camara.objects.filter(habilitada=True)
  camaras_activas = camaras.filter(estado="activa").count()
  alertas_recientes = Alerta.objects.filter(leida=False).order_by("-fecha_hora")[:5]

  try:
    detector_config = DetectorConfig.get_solo()
  except (OperationalError, ProgrammingError):
    detector_config = None

  ventana_24h = timezone.now() - timedelta(hours=24)
  detecciones_recientes = Deteccion.objects.filter(fecha_hora__gte=ventana_24h, procesada=True)
  detecciones_totales = detecciones_recientes.count()
  detecciones_sin_casco = detecciones_recientes.filter(usa_casco=False).count()
  detecciones_con_casco = detecciones_recientes.filter(usa_casco=True).count()

  infracciones_por_camara = {}
  infracciones_query = Deteccion.objects.filter(
    usa_casco=False,
    fecha_hora__gte=ventana_24h
  ).order_by('camara_id', '-fecha_hora')

  for infraccion in infracciones_query:
    camara_id_str = str(infraccion.camara_id)
    if camara_id_str not in infracciones_por_camara:
      infracciones_por_camara[camara_id_str] = []

    # --- LÍNEA CORREGIDA ---
    # La ruta real de guardado es Año/Mes, no Año/Mes/Día.
    # Usamos strftime("%Y/%m") para que coincida con la estructura de tu carpeta 'media'.
    pdf_directory = infraccion.fecha_hora.strftime("%Y/%m")
    pdf_filename = f'infraccion_{infraccion.id}_{infraccion.fecha_hora.strftime("%Y%m%d_%H%M%S")}.pdf'
    pdf_path = f'/media/reportes_infracciones/{pdf_directory}/{pdf_filename}'
    # --- FIN DE LA CORRECCIÓN ---

    infracciones_por_camara[camara_id_str].append({
      'id': infraccion.id,
      'confianza': float(infraccion.confianza),
      'fecha_hora': infraccion.fecha_hora.strftime('%d/%m/%Y %H:%M:%S'),
      'imagen_url': infraccion.imagen_captura.url if infraccion.imagen_captura else '',
      'pdf_url': pdf_path,
    })

  context = {
    "camaras": camaras,
    "total_camaras": camaras.count(),
    "camaras_activas": camaras_activas,
    "alertas_recientes": alertas_recientes,
    "detecciones_total_24h": detecciones_totales,
    "detecciones_con_casco_24h": detecciones_con_casco,
    "detecciones_sin_casco_24h": detecciones_sin_casco,
    "detector_config": detector_config,
    "infracciones_por_camara_json": json.dumps(infracciones_por_camara, cls=DjangoJSONEncoder),
  }

  return render(request, "core/panel.html", context)


# @login_required
# @permission_required("core.view_camara", raise_exception=True)
# def panel_view(request):
#   """
#   Vista del panel de control (Dashboard)
#   Muestra las cámaras activas y su estado, AHORA TAMBIÉN CON INFRACCIONES RECIENTES.
#   """
#   camaras = Camara.objects.filter(habilitada=True)
#   camaras_activas = camaras.filter(estado="activa").count()
#   alertas_recientes = Alerta.objects.filter(leida=False).order_by("-fecha_hora")[:5]
#
#   try:
#     detector_config = DetectorConfig.get_solo()
#   except (OperationalError, ProgrammingError):
#     detector_config = None
#
#   ventana_24h = timezone.now() - timedelta(hours=24)
#   detecciones_recientes = Deteccion.objects.filter(fecha_hora__gte=ventana_24h, procesada=True)
#   detecciones_totales = detecciones_recientes.count()
#   detecciones_sin_casco = detecciones_recientes.filter(usa_casco=False).count()
#   detecciones_con_casco = detecciones_recientes.filter(usa_casco=True).count()
#
#   # --- INICIO DEL CÓDIGO NUEVO PARA LA PERSISTENCIA ---
#
#   # 1. Obtener todas las infracciones de las últimas 24 horas
#   infracciones_por_camara = {}
#   infracciones_query = Deteccion.objects.filter(
#     usa_casco=False,
#     fecha_hora__gte=ventana_24h
#   ).order_by('camara_id', '-fecha_hora')
#
#   # 2. Agruparlas por cámara y construir una estructura de datos para el frontend
#   for infraccion in infracciones_query:
#     camara_id_str = str(infraccion.camara_id)
#     if camara_id_str not in infracciones_por_camara:
#       infracciones_por_camara[camara_id_str] = []
#
#     # Generar la URL del PDF (simulando la lógica de reporting.py o pipeline.py)
#     # Esto asume que el path del reporte de infracción es predecible
#     pdf_filename = f'infraccion_{infraccion.id}_{infraccion.fecha_hora.strftime("%Y%m%d_%H%M%S")}.pdf'
#     pdf_path = f'/media/reportes_infracciones/{infraccion.fecha_hora.strftime("%Y/%m/%d")}/{pdf_filename}'
#
#     infracciones_por_camara[camara_id_str].append({
#       'id': infraccion.id,
#       'confianza': float(infraccion.confianza),
#       'fecha_hora': infraccion.fecha_hora.strftime('%d/%m/%Y %H:%M:%S'),
#       'imagen_url': infraccion.imagen_captura.url if infraccion.imagen_captura else '',
#       'pdf_url': pdf_path,
#     })
#
#   # --- FIN DEL CÓDIGO NUEVO ---
#
#   context = {
#     "camaras": camaras,
#     "total_camaras": camaras.count(),
#     "camaras_activas": camaras_activas,
#     "alertas_recientes": alertas_recientes,
#     "detecciones_total_24h": detecciones_totales,
#     "detecciones_con_casco_24h": detecciones_con_casco,
#     "detecciones_sin_casco_24h": detecciones_sin_casco,
#     "detector_config": detector_config,
#     # 3. Pasar el diccionario como JSON al contexto de la plantilla
#     "infracciones_por_camara_json": json.dumps(infracciones_por_camara, cls=DjangoJSONEncoder),
#   }
#
#   return render(request, "core/panel.html", context)


@login_required
@permission_required("core.view_camara", raise_exception=True)
def camaras_view(request):
  """
  Vista de gestión de cámaras
  RF-05, RF-06, RF-07
  """
  camaras = Camara.objects.all().order_by("numero")

  # Búsqueda
  buscar = request.GET.get("buscar", "")
  if buscar:
    camaras = camaras.filter(nombre__icontains=buscar)

  context = {
    "camaras": camaras,
    "buscar": buscar,
  }

  return render(request, "core/camaras.html", context)


@login_required
@permission_required("core.view_reportediario", raise_exception=True)
def reportes_view(request):
  """
  Vista de reportes y estadísticas con filtros funcionales.
  RF-08, RF-09
  """
  # ===== OBTENER PARÁMETROS DE FILTRO =====
  fecha_inicio = request.GET.get("fecha_inicio")
  fecha_fin = request.GET.get("fecha_fin")
  camara_id = request.GET.get("camara")

  # ===== CONSULTA BASE: UN REPORTE POR DÍA =====
  # Agrupar por fecha para evitar duplicados
  reportes = (
    ReporteDiario.objects
    .select_related("camara")
    .order_by("-fecha")
  )

  # ===== APLICAR FILTROS =====
  if fecha_inicio:
    try:
      fecha_inicio_obj = datetime.strptime(fecha_inicio, "%Y-%m-%d").date()
      reportes = reportes.filter(fecha__gte=fecha_inicio_obj)
    except ValueError:
      messages.warning(request, "Formato de fecha de inicio inválido.")

  if fecha_fin:
    try:
      fecha_fin_obj = datetime.strptime(fecha_fin, "%Y-%m-%d").date()
      reportes = reportes.filter(fecha__lte=fecha_fin_obj)
    except ValueError:
      messages.warning(request, "Formato de fecha fin inválido.")

  if camara_id:
    try:
      reportes = reportes.filter(camara_id=int(camara_id))
    except (ValueError, TypeError):
      messages.warning(request, "ID de cámara inválido.")

  # ===== ELIMINAR DUPLICADOS: Mantener solo un reporte por fecha =====
  # Si hay múltiples reportes del mismo día, quedarse con el más reciente
  reportes_unicos = {}
  for reporte in reportes:
    if reporte.fecha not in reportes_unicos:
      reportes_unicos[reporte.fecha] = reporte
    elif reporte.fecha_generacion > reportes_unicos[reporte.fecha].fecha_generacion:
      reportes_unicos[reporte.fecha] = reporte

  reportes = list(reportes_unicos.values())
  reportes.sort(key=lambda r: r.fecha, reverse=True)

  # ===== CALCULAR RESUMEN DE ESTADÍSTICAS =====
  if reportes:
    total_vehiculos = sum(r.total_vehiculos for r in reportes)
    total_infractores = sum(r.total_infractores for r in reportes)
    total_con_casco = sum(r.total_con_casco for r in reportes)
    cumplimiento_global = (total_con_casco / total_vehiculos * 100) if total_vehiculos else 0
  else:
    total_vehiculos = 0
    total_infractores = 0
    total_con_casco = 0
    cumplimiento_global = 0

  reportes_count = len(reportes)
  ultimo_reporte = reportes[0] if reportes else None

  # ===== OBTENER CÁMARAS PARA EL FILTRO =====
  camaras = Camara.objects.filter(habilitada=True).order_by("numero")

  # ===== CONTEXTO =====
  context = {
    "reportes": reportes,
    "camaras": camaras,
    "fecha_inicio": fecha_inicio,
    "fecha_fin": fecha_fin,
    "camara_seleccionada": camara_id,
    "puede_generar": request.user.has_perm("core.add_reportediario"),
    "fecha_generacion_default": timezone.localdate().strftime("%Y-%m-%d"),
    "total_vehiculos": total_vehiculos,
    "total_infractores": total_infractores,
    "total_con_casco": total_con_casco,
    "cumplimiento_global": cumplimiento_global,
    "reportes_count": reportes_count,
    "ultimo_reporte": ultimo_reporte,
  }

  return render(request, "core/reporte.html", context)


@login_required
@permission_required("core.add_reportediario", raise_exception=True)
@require_POST
def generar_reporte_view(request):
  """
  Genera un reporte diario en PDF para la fecha especificada.

  Este reporte agrupa todas las detecciones procesadas de todas las cámaras
  en esa fecha específica.
  """
  fecha_str = request.POST.get("fecha")

  if not fecha_str:
    messages.error(request, "Debes seleccionar una fecha para generar el reporte.")
    return redirect("reportes")

  try:
    report_date = datetime.strptime(fecha_str, "%Y-%m-%d").date()
  except ValueError:
    messages.error(request, "El formato de fecha es inválido. Utiliza YYYY-MM-DD.")
    return redirect("reportes")

  try:
    result = generate_daily_report(report_date)

    messages.success(
      request,
      (
        f"✅ Reporte generado correctamente para {report_date.strftime('%d/%m/%Y')}. "
        f"Total: {result.total_general} detecciones "
        f"({result.total_con_casco} con casco, {result.total_sin_casco} sin casco)."
      ),
    )

  except ValueError as exc:
    # Errores de validación (sin datos, sin cámaras, etc.)
    messages.warning(request, str(exc))

  except Exception as exc:
    # Errores inesperados
    print(f"❌ Error al generar reporte: {exc}")
    import traceback
    traceback.print_exc()
    messages.error(
      request,
      f"Ocurrió un error inesperado al generar el reporte: {exc}"
    )

  return redirect("reportes")


@login_required
@permission_required("core.add_camara", raise_exception=True)
def agregar_camara_view(request):
  """
  Vista para agregar una nueva cámara
  """
  if request.method == "POST":
    form = CamaraForm(request.POST)
    if form.is_valid():
      camara = form.save(commit=False)
      camara.usuario_creador = request.user
      camara.save()
      messages.success(request, f'Cámara "{camara.nombre}" creada exitosamente.')
      return redirect("camaras")
    else:
      messages.error(request, "Por favor corrige los errores en el formulario.")
  else:
    form = CamaraForm()

  return render(request, "core/agregar.html", {"form": form})


@login_required
@permission_required("core.change_camara", raise_exception=True)
def editar_camara_view(request, numero):
  """
  Vista para editar una cámara existente
  """
  camara = get_object_or_404(Camara, numero=numero)
  if request.method == "POST":
    form = CamaraForm(request.POST, instance=camara)
    if form.is_valid():
      form.save()
      messages.success(
        request, f'Cámara "{camara.nombre}" actualizada exitosamente.'
      )
      return redirect("camaras")
    else:
      messages.error(request, "Por favor corrige los errores en el formulario.")
  else:
    form = CamaraForm(instance=camara)

  return render(request, "core/editar.html", {"form": form, "camara": camara})


@login_required
@permission_required("core.delete_camara", raise_exception=True)
def eliminar_camara_view(request, numero):
  """
  Vista para eliminar una cámara
  """
  camara = get_object_or_404(Camara, numero=numero)

  if request.method == "POST":
    nombre = camara.nombre
    camara.delete()
    messages.success(request, f'Cámara "{nombre}" eliminada exitosamente.')
  else:
    messages.info(
      request, "La eliminación debe realizarse mediante el botón correspondiente."
    )
  return redirect("camaras")


@login_required
@permission_required("core.add_deteccion", raise_exception=True)
@require_POST
def detectar_camara_view(request, numero):
  """
  Inicia el modelo YOLOv8 en segundo plano y devuelve una URL de WebSocket.
  """
  camara = get_object_or_404(Camara, numero=numero)

  if camara.tipo_fuente != "archivo" or not camara.archivo_video:
    return JsonResponse({
      "success": False,
      "message": "La detección solo está disponible para cámaras con un archivo de video local.",
    }, status=400)

  # --- Obtener parámetros del formulario con valores por defecto y validación ---
  try:
    fps = int(request.POST.get("frames_por_segundo", "10"))
    fps = max(1, min(fps, 30))
  except (ValueError, TypeError):
    fps = 10

  try:
    max_frames = int(request.POST.get("max_frames", "300"))
    max_frames = max(0, min(max_frames, 2000))
  except (ValueError, TypeError):
    max_frames = 300

  save_detections = request.POST.get("save_media") == "1"
  draw_boxes = request.POST.get("draw_boxes", "1") == "1"

  # --- Función que se ejecutará en el hilo ---
  def run_detection_thread():
    from .yolo.pipeline import run_detector_for_camera, DetectorError
    print(f"Iniciando hilo de detección para cámara {camara.numero}...")
    try:
      summary = run_detector_for_camera(
        camara=camara,
        max_frames=max_frames if max_frames > 0 else None,
        frames_por_segundo=fps,
        save_detections=save_detections,
        draw_boxes=draw_boxes,
      )
      print(f"Detección completada para cámara {camara.numero}. Infractores guardados: {summary.saved_infractions}")
    except DetectorError as e:
      print(f"ERROR de Detector en cámara {camara.numero}: {e}")
    except Exception as e:
      print(f"ERROR Inesperado en hilo de detección para cámara {camara.numero}: {e}")

  # --- Iniciar el hilo y responder ---
  thread = threading.Thread(target=run_detection_thread)
  thread.start()

  websocket_url = f'/ws/detect/{camara.numero}/'
  return JsonResponse({
    'success': True,
    'message': 'Iniciando proceso de detección...',
    'websocket_url': websocket_url
  })


@login_required
@permission_required("core.change_detectorconfig", raise_exception=True)
def detector_control_view(request):
  if request.method != "POST":
    return HttpResponseNotAllowed(["POST"])

  config = DetectorConfig.get_solo()
  action = request.POST.get("action")

  if action == "start":
    try:
      max_frames = int(request.POST.get("max_frames", config.max_frames))
      vid_stride = int(request.POST.get("vid_stride", config.vid_stride))
      intervalo = int(request.POST.get("intervalo_segundos", config.intervalo_segundos))
    except (TypeError, ValueError):
      messages.error(request, "Los valores numéricos proporcionados no son válidos.")
      return redirect("panel")

    max_frames = max(1, min(max_frames, 2000))
    vid_stride = max(1, min(vid_stride, 30))
    intervalo = max(5, min(intervalo, 3600))

    config.max_frames = max_frames
    config.vid_stride = vid_stride
    config.intervalo_segundos = intervalo
    config.guardar_anotaciones = request.POST.get("guardar_anotaciones") == "1"
    config.activo = True
    config.save()

    detector_manager.start()
    messages.success(
      request,
      "Detección continua activada. El servicio procesará automáticamente las cámaras con video local.",
    )
  elif action == "stop":
    config.activo = False
    config.save(update_fields=["activo", "actualizado"])
    detector_manager.stop()
    messages.info(request, "Se detuvo la detección continua.")
  else:
    messages.error(request, "Acción no reconocida para el detector.")

  return redirect("panel")


@login_required
def usuarios_view(request):
  """Panel para que superusuarios gestionen roles y permisos de cuentas."""
  if not request.user.is_superuser:
    raise PermissionDenied

  usuarios = list(
    Usuario.objects.all()
    .prefetch_related("groups", "user_permissions")
    .order_by("username")
  )
  for usuario in usuarios:
    usuario.effective_perms = usuario.get_all_permissions()

  grupos = Group.objects.all().order_by("name")
  permisos = (
    Permission.objects.filter(
      codename__in=MANAGED_PERMISSION_CODES, content_type__app_label="core"
    )
    .select_related("content_type")
    .order_by("name")
  )

  permission_sections = OrderedDict((level, []) for level in ["alto", "medio", "bajo"])
  for permiso in permisos:
    level = next(
      (
        level_key
        for level_key, code_set in PERMISSION_LEVEL_MAP.items()
        if permiso.codename in code_set
      ),
      "medio",
    )
    permission_sections[level].append(permiso)

  permission_section_data = [
    {
      "level": level,
      "label": PERMISSION_LEVEL_LABELS.get(level, level.title()),
      "permisos": perms,
    }
    for level, perms in permission_sections.items()
  ]

  usuarios_payload = []
  for usuario in usuarios:
    full_name = (usuario.get_full_name() or "").strip()
    usuarios_payload.append(
      {
        "id": usuario.id,
        "username": usuario.username,
        "full_name": full_name,
        "email": usuario.email or "",
        "rol": usuario.rol,
        "groups": list(usuario.groups.values_list("name", flat=True)),
        "permissions": sorted(usuario.effective_perms),
        "is_superuser": usuario.is_superuser,
      }
    )

  group_permissions_map = {
    group.name: sorted(
      group.permissions.filter(content_type__app_label="core").values_list(
        "codename", flat=True
      )
    )
    for group in grupos
  }

  context = {
    "usuarios": usuarios,
    "grupos": grupos,
    "permission_sections": permission_section_data,
    "rol_choices": Usuario.ROLES,
    "usuarios_json": json.dumps(usuarios_payload, cls=DjangoJSONEncoder),
    "group_permissions_json": json.dumps(group_permissions_map, cls=DjangoJSONEncoder),
    "role_group_json": json.dumps(ROLE_GROUP_MAP, cls=DjangoJSONEncoder),
    "group_level_json": json.dumps(GROUP_LEVEL_INFO, cls=DjangoJSONEncoder),
  }
  return render(request, "core/usuarios.html", context)


@login_required
@require_POST
def actualizar_usuario_permisos_view(request):
  """Actualiza rol, grupos y permisos de un usuario mediante llamada AJAX."""
  if not request.user.is_superuser:
    raise PermissionDenied

  try:
    payload = json.loads(request.body.decode("utf-8"))
  except (json.JSONDecodeError, UnicodeDecodeError):
    return JsonResponse(
      {"success": False, "message": "No se pudo procesar la solicitud."},
      status=400,
    )

  user_id = payload.get("user_id")
  rol = payload.get("rol")
  grupos_nombres = payload.get("groups", [])
  permisos_codenames = payload.get("permissions", [])

  valid_roles = {choice[0] for choice in Usuario.ROLES}
  if rol not in valid_roles:
    return JsonResponse({"success": False, "message": "Rol inválido."}, status=400)

  usuario_objetivo = get_object_or_404(Usuario, pk=user_id)

  grupos = list(Group.objects.filter(name__in=grupos_nombres))
  if len(grupos) != len(set(grupos_nombres)):
    return JsonResponse(
      {"success": False, "message": "Uno de los grupos seleccionados no existe."},
      status=400,
    )

  permisos = list(
    Permission.objects.filter(
      codename__in=permisos_codenames, content_type__app_label="core"
    )
  )
  if len(permisos) != len(set(permisos_codenames)):
    return JsonResponse(
      {
        "success": False,
        "message": "Uno de los permisos seleccionados no es válido.",
      },
      status=400,
    )

  with transaction.atomic():
    usuario_objetivo.rol = rol
    usuario_objetivo.is_staff = usuario_objetivo.is_superuser or rol == "admin"
    usuario_objetivo.save(update_fields=["rol", "is_staff"])
    usuario_objetivo.groups.set(grupos)
    usuario_objetivo.user_permissions.set(permisos)

  return JsonResponse(
    {
      "success": True,
      "message": f"Permisos actualizados para {usuario_objetivo.username}.",
      "data": {
        "rol": usuario_objetivo.get_rol_display(),
        "groups": [group.name for group in grupos],
        "permissions": [perm.codename for perm in permisos],
      },
    }
  )

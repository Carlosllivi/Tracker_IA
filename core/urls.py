from django.urls import path
from . import views

urlpatterns = [
    # Autenticación
    path("", views.login_view, name="login"),
    path("login/", views.login_view, name="login"),
    path("registro/", views.registro_view, name="registro"),
    path("logout/", views.logout_view, name="logout"),
    path("perfil/", views.perfil_view, name="perfil"),
    # Panel principal
    path("panel/", views.panel_view, name="panel"),
    # Gestión de cámaras
    path("camaras/", views.camaras_view, name="camaras"),
    path("camaras/agregar/", views.agregar_camara_view, name="agregar_camara"),
    path(
        "camaras/editar/<int:numero>/", views.editar_camara_view, name="editar_camara"
    ),
    path(
        "camaras/<int:numero>/detectar/",
        views.detectar_camara_view,
        name="detectar_camara",
    ),
    path(
        "camaras/eliminar/<int:numero>/",
        views.eliminar_camara_view,
        name="eliminar_camara",
    ),
    # Reportes
    path("reportes/", views.reportes_view, name="reportes"),
    path(
        "reportes/generar/",
        views.generar_reporte_view,
        name="generar_reporte_diario",
    ),
    # Gestión de usuarios y roles
    path("usuarios/", views.usuarios_view, name="usuarios"),
    path(
        "usuarios/actualizar/",
        views.actualizar_usuario_permisos_view,
        name="actualizar_usuario_permisos",
    ),
    path("detector/control/", views.detector_control_view, name="detector_control"),
]

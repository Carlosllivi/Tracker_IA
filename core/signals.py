"""Señales y utilidades para inicializar grupos y permisos predeterminados."""

from django.contrib.auth.models import Group, Permission
from django.db.models.signals import post_migrate
from django.dispatch import receiver

# Definición de permisos por grupo. Cada clave representa un grupo y su
# valor es un diccionario con el app_label como clave y la lista de codenames
# de permisos como valor. Esto facilita extender los permisos en el futuro.
GROUP_PERMISSIONS = {
    "Administrador": {
        "core": [
            "add_camara",
            "change_camara",
            "delete_camara",
            "view_camara",
            "add_reportediario",
            "change_reportediario",
            "delete_reportediario",
            "view_reportediario",
            "add_alerta",
            "change_alerta",
            "delete_alerta",
            "view_alerta",
            "add_deteccion",
            "change_deteccion",
            "delete_deteccion",
            "view_deteccion",
            "view_detectorconfig",
            "change_detectorconfig",
            "add_usuario",
            "change_usuario",
            "delete_usuario",
            "view_usuario",
        ],
        "auth": [
            "view_group",
            "add_group",
            "change_group",
        ],
    },
    "Operador": {
        "core": [
            "view_camara",
            "view_reportediario",
            "view_alerta",
            "view_deteccion",
            "add_deteccion",
            "view_detectorconfig",
        ],
    },
    "Visualizador": {
        "core": [
            "view_camara",
            "view_reportediario",
            "view_deteccion",
            "view_detectorconfig",
        ],
    },
}


def _get_permissions_for_group(permission_map):
    """Resuelve y devuelve los permisos indicados en *permission_map*."""
    permissions = []
    for app_label, codenames in permission_map.items():
        if not codenames:
            continue
        perms = Permission.objects.filter(
            content_type__app_label=app_label,
            codename__in=codenames,
        )
        permissions.extend(perms)
    return permissions


@receiver(post_migrate)
def ensure_default_groups(sender, **kwargs):
    """Crea los grupos base y asigna sus permisos tras migraciones."""
    if sender.name != "core":
        return

    for group_name, permission_map in GROUP_PERMISSIONS.items():
        group, _ = Group.objects.get_or_create(name=group_name)
        desired_permissions = _get_permissions_for_group(permission_map)
        # Usar set para evitar duplicados y convertir a lista nuevamente.
        unique_permissions = list({perm.id: perm for perm in desired_permissions}.values())
        group.permissions.set(unique_permissions)
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import Usuario, Camara, Deteccion, ReporteDiario, Alerta

@admin.register(Usuario)
class UsuarioAdmin(UserAdmin):
    list_display = ['username', 'email', 'first_name', 'last_name', 'rol', 'activo', 'fecha_registro']
    list_filter = ['rol', 'activo', 'fecha_registro', 'is_staff']
    search_fields = ['username', 'email', 'first_name', 'last_name']
    
    fieldsets = UserAdmin.fieldsets + (
        ('Información Adicional', {
            'fields': ('rol', 'telefono', 'activo', 'profile_image')
        }),
    )

@admin.register(Camara)
class CamaraAdmin(admin.ModelAdmin):
    list_display = ['numero', 'nombre', 'ubicacion', 'estado', 'habilitada', 'fecha_instalacion']
    list_filter = ['estado', 'habilitada', 'fecha_instalacion']
    search_fields = ['nombre', 'ubicacion']
    list_editable = ['habilitada', 'estado']
    readonly_fields = ['fecha_creacion', 'fecha_actualizacion']

@admin.register(Deteccion)
class DeteccionAdmin(admin.ModelAdmin):
    list_display = ['camara', 'fecha_hora', 'usa_casco', 'confianza', 'procesada']
    list_filter = ['usa_casco', 'procesada', 'fecha_hora', 'camara']
    search_fields = ['camara__nombre']
    date_hierarchy = 'fecha_hora'
    readonly_fields = ['fecha_hora']

@admin.register(ReporteDiario)
class ReporteDiarioAdmin(admin.ModelAdmin):
    list_display = ['camara', 'fecha', 'total_vehiculos', 'total_infractores', 
                    'porcentaje_cumplimiento', 'generado']
    list_filter = ['fecha', 'camara', 'generado']
    search_fields = ['camara__nombre']
    date_hierarchy = 'fecha'
    readonly_fields = ['fecha_generacion']

@admin.register(Alerta)
class AlertaAdmin(admin.ModelAdmin):
    list_display = ['tipo', 'prioridad', 'camara', 'fecha_hora', 'leida', 'atendida']
    list_filter = ['tipo', 'prioridad', 'leida', 'atendida', 'fecha_hora']
    search_fields = ['mensaje', 'camara__nombre']
    date_hierarchy = 'fecha_hora'
    readonly_fields = ['fecha_hora']
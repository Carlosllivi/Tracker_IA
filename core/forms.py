import os
from pathlib import Path

from django import forms
from django.conf import settings
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.models import Group

from .models import Usuario, Camara


class ProfileForm(forms.ModelForm):
  """Formulario para que el usuario actualice su perfil y avatar."""

  class Meta:
    model = Usuario
    fields = [
      "profile_image",
      "first_name",
      "last_name",
      "email",
      "telefono",
    ]

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # Add consistent classes to ProfileForm widgets so inputs keep good contrast
    widget_class = (
      "form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-lg "
      "text-[#0d141b] dark:text-white focus:outline-none focus:ring-2 focus:ring-primary/50 "
      "border border-[#cfdbe7] bg-white dark:bg-background-dark dark:border-gray-600 "
      "dark:focus:ring-primary/50 h-14 placeholder:text-[#4c739a] p-[15px] text-base font-normal leading-normal"
    )
    if "profile_image" in self.fields:
      self.fields["profile_image"].widget.attrs.update({"class": "mt-1"})
    for fname in ("first_name", "last_name", "email", "telefono"):
      if fname in self.fields:
        self.fields[fname].widget.attrs.update({"class": widget_class})


class RegistroForm(UserCreationForm):
  """
  Formulario de registro de usuarios
  """

  email = forms.EmailField(
    required=True,
    widget=forms.EmailInput(
      attrs={
        "class": "form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-lg text-[#0d141b] dark:text-white focus:outline-none focus:ring-2 focus:ring-primary/50 border border-[#cfdbe7] bg-white dark:bg-background-dark dark:border-gray-600 dark:focus:ring-primary/50 h-14 placeholder:text-[#4c739a] p-[15px] text-base font-normal leading-normal",
        "placeholder": "tu@email.com",
      }
    ),
  )

  first_name = forms.CharField(
    max_length=150,
    required=True,
    label="Nombre",
    widget=forms.TextInput(
      attrs={
        "class": "form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-lg text-[#0d141b] dark:text-white focus:outline-none focus:ring-2 focus:ring-primary/50 border border-[#cfdbe7] bg-white dark:bg-background-dark dark:border-gray-600 dark:focus:ring-primary/50 h-14 placeholder:text-[#4c739a] p-[15px] text-base font-normal leading-normal",
        "placeholder": "Ingresa tu nombre",
      }
    ),
  )

  last_name = forms.CharField(
    max_length=150,
    required=True,
    label="Apellido",
    widget=forms.TextInput(
      attrs={
        "class": "form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-lg text-[#0d141b] dark:text-white focus:outline-none focus:ring-2 focus:ring-primary/50 border border-[#cfdbe7] bg-white dark:bg-background-dark dark:border-gray-600 dark:focus:ring-primary/50 h-14 placeholder:text-[#4c739a] p-[15px] text-base font-normal leading-normal",
        "placeholder": "Ingresa tu apellido",
      }
    ),
  )

  class Meta:
    model = Usuario
    fields = [
      "username",
      "profile_image",
      "first_name",
      "last_name",
      "email",
      "password1",
      "password2",
    ]

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.fields["username"].widget.attrs.update(
      {
        "class": "form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-lg text-[#0d141b] dark:text-white focus:outline-none focus:ring-2 focus:ring-primary/50 border border-[#cfdbe7] bg-white dark:bg-background-dark dark:border-gray-600 dark:focus:ring-primary/50 h-14 placeholder:text-[#4c739a] p-[15px] text-base font-normal leading-normal",
        "placeholder": "Elige un nombre de usuario",
      }
    )
    self.fields["password1"].widget.attrs.update(
      {
        "class": "form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-lg text-[#0d141b] dark:text-white focus:outline-none focus:ring-2 focus:ring-primary/50 border border-[#cfdbe7] bg-white dark:bg-background-dark dark:border-gray-600 dark:focus:ring-primary/50 h-14 placeholder:text-[#4c739a] p-[15px] rounded-r-none border-r-0 pr-2 text-base font-normal leading-normal",
        "placeholder": "Mínimo 8 caracteres",
      }
    )
    self.fields["password2"].widget.attrs.update(
      {
        "class": "form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-lg text-[#0d141b] dark:text-white focus:outline-none focus:ring-2 focus:ring-primary/50 border border-[#cfdbe7] bg-white dark:bg-background-dark dark:border-gray-600 dark:focus:ring-primary/50 h-14 placeholder:text-[#4c739a] p-[15px] rounded-r-none border-r-0 pr-2 text-base font-normal leading-normal",
        "placeholder": "Repite tu contraseña",
      }
    )

  def save(self, commit=True):
    user = super().save(commit=False)
    user.email = self.cleaned_data["email"]
    user.first_name = self.cleaned_data["first_name"]
    user.last_name = self.cleaned_data["last_name"]
    # Guardar imagen de perfil si fue subida
    profile_image = self.cleaned_data.get("profile_image")
    if profile_image:
      user.profile_image = profile_image
    user.rol = "operador"
    if commit:
      user.save()
      self._assign_default_group(user)
    return user

  def _assign_default_group(self, user):
    """Asigna el grupo Operador por defecto a nuevos registros."""
    try:
      operador_group, _ = Group.objects.get_or_create(name="Operador")
      user.groups.add(operador_group)
    except Exception:
      # Si no es posible asignar el grupo, no interrumpir el registro.
      pass


class LoginForm(AuthenticationForm):
  """
  Formulario de inicio de sesión personalizado
  """

  username = forms.CharField(
    label="Correo Electrónico o Usuario",
    widget=forms.TextInput(
      attrs={
        "class": "form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-lg text-[#0d141b] dark:text-white focus:outline-none focus:ring-2 focus:ring-primary border border-border-light bg-background-light dark:bg-gray-700 dark:border-gray-600 focus:border-primary h-14 placeholder:text-gray-400 p-[15px] rounded-r-none border-r-0 pr-2 text-base font-normal leading-normal",
        "placeholder": "Introduce tu correo electrónico",
      }
    ),
  )

  password = forms.CharField(
    label="Contraseña",
    widget=forms.PasswordInput(
      attrs={
        "class": "form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-lg text-[#0d141b] dark:text-white focus:outline-none focus:ring-2 focus:ring-primary border border-border-light bg-background-light dark:bg-gray-700 dark:border-gray-600 focus:border-primary h-14 placeholder:text-gray-400 p-[15px] rounded-r-none border-r-0 pr-2 text-base font-normal leading-normal",
        "placeholder": "Introduce tu contraseña",
      }
    ),
  )


class CamaraForm(forms.ModelForm):
  """Formulario para crear y editar cámaras."""

  archivo_video = forms.ChoiceField(
    required=False,
    choices=[],
    widget=forms.Select(
      attrs={
        "class": "form-select flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-lg text-[#0d141b] dark:text-white focus:outline-none focus:ring-2 focus:ring-primary border border-gray-300 bg-white dark:bg-gray-700 dark:border-gray-600 h-12 px-4 text-base font-normal leading-normal",
      }
    ),
    label="Archivo de video",
  )

  # Campo oculto para guardar el frame capturado de la cámara
  camera_frame = forms.CharField(
    required=False,
    widget=forms.HiddenInput(),
  )

  class Meta:
    model = Camara
    fields = [
      "nombre",
      "ubicacion",
      "tipo_fuente",
      "url_streaming",
      "archivo_video",
      "camera_frame",
      "estado",
      "habilitada",
    ]
    widgets = {
      "nombre": forms.TextInput(
        attrs={
          "class": "form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-lg text-[#0d141b] dark:text-white focus:outline-none focus:ring-2 focus:ring-primary border border-gray-300 bg-white dark:bg-gray-700 dark:border-gray-600 h-12 placeholder:text-gray-400 px-4 text-base font-normal leading-normal",
          "placeholder": "Ej: Cámara Av. Principal",
        }
      ),
      "ubicacion": forms.TextInput(
        attrs={
          "class": "form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-lg text-[#0d141b] dark:text-white focus:outline-none focus:ring-2 focus:ring-primary border border-gray-300 bg-white dark:bg-gray-700 dark:border-gray-600 h-12 placeholder:text-gray-400 px-4 text-base font-normal leading-normal",
          "placeholder": "Ej: Av. Principal con Calle 5",
        }
      ),
      "tipo_fuente": forms.Select(
        attrs={
          "class": "form-select flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-lg text-[#0d141b] dark:text-white focus:outline-none focus:ring-2 focus:ring-primary border border-gray-300 bg-white dark:bg-gray-700 dark:border-gray-600 h-12 px-4 text-base font-normal leading-normal",
          "id": "id_tipo_fuente"
        }
      ),
      "url_streaming": forms.URLInput(
        attrs={
          "class": "form-input flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-lg text-[#0d141b] dark:text-white focus:outline-none focus:ring-2 focus:ring-primary border border-gray-300 bg-white dark:bg-gray-700 dark:border-gray-600 h-12 placeholder:text-gray-400 px-4 text-base font-normal leading-normal",
          "placeholder": "http://192.168.1.100:8080/stream",
        }
      ),
      "estado": forms.Select(
        attrs={
          "class": "form-select flex w-full min-w-0 flex-1 resize-none overflow-hidden rounded-lg text-[#0d141b] dark:text-white focus:outline-none focus:ring-2 focus:ring-primary border border-gray-300 bg-white dark:bg-gray-700 dark:border-gray-600 h-12 px-4 text-base font-normal leading-normal"
        }
      ),
      "habilitada": forms.CheckboxInput(
        attrs={
          "class": "form-checkbox h-5 w-5 text-primary focus:ring-primary border-gray-300 rounded"
        }
      ),
    }
    labels = {
      "nombre": "Nombre de la Cámara",
      "ubicacion": "Ubicación",
      "tipo_fuente": "Fuente de video",
      "url_streaming": "URL de Streaming",
      "archivo_video": "Archivo de video",
      "estado": "Estado",
      "habilitada": "Cámara Habilitada",
    }
    help_texts = {
      "nombre": "Nombre descriptivo para identificar la cámara",
      "ubicacion": "Dirección o punto de referencia",
      "tipo_fuente": "Selecciona si esta cámara usa una URL en vivo, un video local o cámara web en vivo.",
      "url_streaming": "URL del stream de video (solo para fuente 'URL de streaming')",
      "archivo_video": "Selecciona un archivo `.mp4` ubicado en `static/videos/` (solo para fuente 'Archivo de video')",
      "estado": "Estado operativo actual de la cámara",
      "habilitada": "Marcar para habilitar la cámara en el sistema",
    }

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.fields["archivo_video"].choices = self._build_video_choices()

  def _build_video_choices(self):
    base_path = Path(settings.BASE_DIR) / "static" / "videos"
    choices = [("", "-- Selecciona un video --")]
    if base_path.exists() and base_path.is_dir():
      for video_file in sorted(base_path.glob("*.mp4")):
        relative_path = os.path.join("videos", video_file.name)
        choices.append((relative_path, video_file.name))
    return choices

  def clean(self):
    cleaned_data = super().clean()
    tipo_fuente = cleaned_data.get("tipo_fuente")
    url_streaming = (cleaned_data.get("url_streaming") or "").strip()
    archivo_video = (cleaned_data.get("archivo_video") or "").strip()
    camera_frame = (cleaned_data.get("camera_frame") or "").strip()

    if tipo_fuente == "stream":
      if not url_streaming:
        self.add_error("url_streaming", "Debes proporcionar una URL de streaming.")
      cleaned_data["archivo_video"] = ""
      cleaned_data["camera_frame"] = ""
    elif tipo_fuente == "archivo":
      if not archivo_video:
        self.add_error("archivo_video", "Selecciona un archivo de video disponible.")
      cleaned_data["url_streaming"] = ""
      cleaned_data["camera_frame"] = ""
    elif tipo_fuente == "cameraLive":
      # Para cámara en vivo, limpiamos los otros campos
      cleaned_data["url_streaming"] = ""
      cleaned_data["archivo_video"] = ""

    return cleaned_data

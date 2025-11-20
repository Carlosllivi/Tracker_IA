# core/storages.py

from django.conf import settings
from storages.backends.s3boto3 import S3Boto3Storage


class ProfileImageStorage(S3Boto3Storage):
  """
  Almacenamiento personalizado para guardar las fotos de perfil
  en una carpeta 'media/profiles/' dentro del bucket de S3.
  """
  # Usa las credenciales de AWS definidas en settings.py
  access_key = settings.AWS_ACCESS_KEY_ID
  secret_key = settings.AWS_SECRET_ACCESS_KEY
  bucket_name = settings.AWS_STORAGE_BUCKET_NAME

  # Carpeta raíz dentro del bucket para estos archivos
  location = 'media'

  # No sobrescribir archivos con el mismo nombre
  file_overwrite = False

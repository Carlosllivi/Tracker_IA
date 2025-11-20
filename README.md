# Tracker IA: Sistema Inteligente para Detección Confiable de Infractores por no usar casco

Tracker IA es una plataforma web desarrollada con Django que detecta motociclistas y verifica el uso de casco a partir de video. Combina un pipeline de visión por computador (YOLOv8 + OpenCV) con un panel web en tiempo real (Django Channels/WebSockets), gestión de cámaras, reportes en PDF y control de acceso por roles.

Durante las últimas mejoras se reforzó la confiabilidad del detector y la trazabilidad de evidencias, logrando una detección más robusta de infractores y una experiencia operativa clara para operadores y administradores.

## ✨ Características principales

- Detección en tiempo real de motociclistas y verificación de casco (con casco / sin casco).
- Dashboard interactivo con estado de cámaras, infracciones recientes, KPIs de las últimas 24 h y streaming de frames anotados vía WebSockets.
- Gestión de cámaras: altas, edición, habilitación y parametrización de fuentes de video (en dev: archivos locales).
- Reportes y evidencias: generación de reportes diarios en PDF y acceso a capturas y enlaces a evidencias por infracción.
- Roles y permisos: Administrador, Operador y Visualizador con permisos diferenciados.
- Alertas automáticas por infracción visibles en el panel principal.
- Almacenamiento en la nube para fotos de perfil de usuarios (AWS S3).

## 🧠 ¿Qué hace confiable a este detector?

Mejoras implementadas en el pipeline `core/yolo/pipeline.py` y servicios `core/yolo/service.py`:

- Tracking de centroides robusto con persistencia de IDs, manejo de oclusiones, captura proactiva de frames y criterio de calidad de imagen (sharpness) para seleccionar la mejor evidencia.
- Asociación persona-moto para reducir falsos positivos (solo evalúa cascos en contexto de motocicleta).
- Doble modelo YOLO: uno general (clases base como `person`, `motorcycle`) y un especialista en casco (clases `driver_with_helmet`, `driver_without_helmet`, etc.).
- Validación con Gemini para corroborar predicciones borde, elevando precisión en casos ambiguos.
- Canal de progreso en vivo por WebSockets: cada frame procesado envía un update al frontend con porcentajes y preview.
- Correcciones en rutas de reportes PDF y organización de media para una trazabilidad consistente.

## 🏗️ Arquitectura y servicios implementados

- Backend: Django 5.2 (MVC) + app `core` (modelos, vistas, formularios, reporting, YOLO).
- Tiempo real: Django Channels + Consumer WebSocket (`core/consumers.py`) y routing (`core/routing.py`).
- Detección: servicios YOLO (`core/yolo/service.py`) y pipeline de procesamiento (`core/yolo/pipeline.py`).
- Reportes: generación de PDFs diarios (ReportLab) y agregación de estadísticas por fecha/cámara.
- Seguridad y gestión: autenticación, grupos, permisos y formularios de perfil con imágenes en S3.
- Almacenamiento: evidencias en disco local (`MEDIA_ROOT`) y fotos de perfil en AWS S3.

Diagrama lógico (alto nivel):

Entrada de video → Detección YOLO (general) → Tracking y selección de mejor frame → Clasificación casco (especialista) → Validación (Gemini) → Persistencia (DB + media) → Alerta + Reporte → Notificación en tiempo real (Channels) → Visualización en panel.

## 🛠️ Tecnologías utilizadas

- Lenguaje/Framework: Python 3.13.1, Django 5.2, Django Channels, Daphne (ASGI).
- IA/Visión: Ultralytics YOLOv8, PyTorch, OpenCV, NumPy, SciPy.
- Realtime: Channels + WebSockets (capa de canales en memoria para dev).
- Reportes: ReportLab.
- Almacenamiento: Archivos locales (media/static) y AWS S3 (solo perfiles de usuario).
- Base de datos: SQLite (dev). Recomendado PostgreSQL en producción.
- UI: HTML, CSS, JavaScript, Bootstrap.

## ✅ Requisitos del sistema

- Python 3.13.1 (recomendado) o superior.
- Pip y virtualenv.
- Git.
- Windows, macOS o Linux con FFmpeg/OpenCV funcionando.
- Para tiempo real en producción: Redis 6+ (opcional en dev, recomendado prod), servidor ASGI (Daphne o Uvicorn) y reverse proxy (Nginx).
- Modelos YOLO descargados localmente (ver Configuración).

## ⚙️ Configuración

1) Variables de entorno (recomendado)

Configura tus credenciales y llaves fuera del código fuente. Ejemplo en PowerShell (Windows):

```powershell
$env:DJANGO_DEBUG="True"
$env:AWS_ACCESS_KEY_ID="..."
$env:AWS_SECRET_ACCESS_KEY="..."
$env:AWS_STORAGE_BUCKET_NAME="..."
$env:GEMINI_API_KEY="..."
```

Revisa `tracker_ia/settings.py` para mapear estas variables y no dejar secretos en el código.

2) Modelo YOLO

Coloca los pesos en:

- General: `core/yolo/weights/yolov8m-seg.pt`
- Especialista cascos (modelo entrenado propio): `core/yolo/weights/best.pt`

Descargas recomendadas (no descargar `best.pt` — es un modelo entrenado propio):
- Yolov8 mediano para segmentación (usado como ejemplo general en este repo): [yolov8m-seg.pt](https://github.com/ultralytics/ultralytics/releases/latest/download/yolov8m-seg.pt)

Ejemplos de descarga y ubicación final (colocar los archivos en `core/yolo/weights/`):

PowerShell (Windows):

```powershell

# Descargar yolov8m-seg
Invoke-WebRequest -Uri "https://github.com/ultralytics/ultralytics/releases/latest/download/yolov8m-seg.pt" -OutFile "core\yolo\weights\yolov8m-seg.pt"
```

Unix / curl (macOS / Linux):

```bash
curl -L -o core/yolo/weights/yolov8n.pt https://github.com/ultralytics/ultralytics/releases/latest/download/yolov8n.pt
curl -L -o core/yolo/weights/yolov8m-seg.pt https://github.com/ultralytics/ultralytics/releases/latest/download/yolov8m-seg.pt
```

Estas rutas se leen desde `YOLO_CONFIG` en `settings.py`:

```python
YOLO_CONFIG = {
  "general_model_weights": BASE_DIR / "core" / "yolo" / "weights" / "yolov8m-seg.pt",
  "helmet_model_weights": BASE_DIR / "core" / "yolo" / "weights" / "best.pt",
  # ... otros parámetros
}
```

3) Almacenamiento y archivos

- Archivos estáticos: `static/` y `staticfiles/` (colecta en despliegue).
- Archivos de media (evidencias, PDFs): `media/`.
- Fotos de perfil: S3 (vía `django-storages`). En desarrollo puedes mantener perfiles en local si prefieres.

4) Channels y WebSockets

En desarrollo, la capa configurada es en memoria:

```python
CHANNEL_LAYERS = {
  "default": {"BACKEND": "channels.layers.InMemoryChannelLayer"}
}
```

## 🧪 Instalación y primera ejecución (desarrollo)

1. Clonar y crear entorno virtual

```bash
git clone https://github.com/Carlosllivi/Tracker_IA.git
cd Tracker_IA
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

2. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. Preparar modelos YOLO

- Copia `yolov8m-seg.pt` y `best.pt` en `core/yolo/weights/`.

4. Migraciones y superusuario

```bash
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser
```

5. Ejecutar el servidor de desarrollo

```bash
python manage.py runserver
# Alternativa ASGI (recomendado para WebSockets, proyecto actual y funcional realizar el comando con daphne):
daphne -p 8000 tracker_ia.asgi:application
```

Abre http://127.0.0.1:8000/

## 🚦 Guía de uso

1) Inicia sesión con tu superusuario.

2) Agrega una cámara: desde "Cámaras" → "Agregar", define nombre y la fuente. Para pruebas con archivos, selecciona tipo "archivo" y referencia un video presente bajo `static/` (el pipeline toma la ruta relativa a `static/`).

3) Inicia una detección: desde el panel o la lista de cámaras, inicia el proceso y observa el progreso en vivo (frames anotados y porcentaje). Las detecciones se almacenan en la base y las infracciones se muestran en el dashboard.

4) Reportes: ve a "Reportes" para filtrar por fechas y generar PDFs diarios. Los enlaces a PDFs están organizados por año/mes bajo `media/`.

## 📁 Estructura del proyecto

```
Tracker_IA/
├── core/
│   ├── yolo/                # Pipeline y servicios YOLO
│   ├── templates/           # UI (Bootstrap)
│   ├── models.py            # Modelos: Usuario, Camara, Deteccion, Reporte, etc.
│   ├── views.py             # Vistas: panel, cámaras, reportes, perfiles
│   ├── consumers.py         # WebSocket: updates de detección
│   ├── routing.py           # Rutas WS
│   └── reporting.py         # Generación de PDFs (ReportLab)
├── tracker_ia/
│   ├── settings.py          # Configuración Django, Channels, YOLO_CONFIG
│   └── urls.py
├── media/                   # Evidencias y reportes (dev)
├── static/                  # Archivos estáticos y videos de prueba
├── requirements.txt
├── manage.py
└── README.md
```

## 🔐 Seguridad y buenas prácticas

- Credenciales o datos sensibles (AWS, Gemini, Base de datos) se leen desde variables de entorno.
- Usa Redis como capa de Channels en producción.
- Sirve el proyecto con un servidor ASGI (Daphne).
- Restringe `ALLOWED_HOSTS` y deshabilita `DEBUG`.

## 🧩 Resolución de problemas

- Modelos YOLO no cargan: verifica rutas de `YOLO_CONFIG` y presencia de archivos en `core/yolo/weights/`.
- WebSockets no actualizan: en dev usa Daphne o asegúrate de Channels configurado; en prod utiliza `channels_redis`.
- Video no abre: verifica que la cámara de tipo `archivo` apunte a un video bajo `static/` y que la ruta exista.
- PDFs no aparecen: revisa permisos de escritura en `media/` y que `reporting` esté generando en año/mes.

## 📦 Metodología y proceso

- Enfoque iterativo con pruebas manuales sobre videos de muestra.
- Métricas operativas en dashboard.
- Separación de responsabilidades: servicio de modelos, pipeline de negocio, capa de tiempo real y capa web.

## 📑 Licencia y contribuciones

Las contribuciones son bienvenidas. Abre un issue para discutir cambios o envía un pull request. Asegúrate de no subir secretos y de probar la carga de modelos y el flujo de WebSockets antes de solicitar revisión.

---

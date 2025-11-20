# Integración YOLOv8

Este documento resume cómo utilizar el detector de cascos basado en YOLOv8 que ahora forma parte del proyecto.

## Requisitos previos

- Entorno virtual activado: `env\Scripts\activate`
- Dependencias instaladas (ya agregadas al entorno): `ultralytics` y sus paquetes auxiliares. Si necesitas reinstalar, ejecuta `pip install ultralytics` dentro del entorno virtual.
- Dataset ubicado en `data/` con la estructura `train/`, `valid/` y `test/`, más el archivo `data/data.yaml` (actualizado para rutas locales).

## Entrenamiento del modelo

1. Sitúate en la raíz del proyecto:
   ```powershell
   Set-Location "C:\Users\Familia FF\Desktop\Tracker_IA - copiaV2"
   ```
2. Ejecuta el comando de entrenamiento. Ejemplo para 75 épocas:
   ```powershell
   & env\Scripts\python.exe manage.py train_yolov8 --epochs 75 --imgsz 640 --name casco_yolov8
   ```
   - Puedes aportar otros pesos base con `--weights ruta\al\modelo.pt`.
   - Los resultados y pesos entrenados se guardan en `core/yolo/weights/runs/` y el mejor modelo se copia en `core/yolo/weights/best.pt`.

## Ejecución del detector desde la terminal

- Para procesar una cámara registrada (debe usar archivo de video local):
  ```powershell
  & env\Scripts\python.exe manage.py run_yolov8_detector --camera 1 --max-frames 300 --vid-stride 5 --save-media
  ```
  - `--save-media` guarda las salidas anotadas en la carpeta `core/yolo/weights/runs/`.
  - Puedes forzar un archivo distinto con `--source "ruta\al\video.mp4"`.

- Para procesar un recurso arbitrario sin guardar en base de datos:
  ```powershell
  & env\Scripts\python.exe manage.py run_yolov8_detector --source "data/test/images/ejemplo.jpg" --max-frames 1
  ```

## Ejecución desde la interfaz web

1. Ve al panel principal.
2. Abre la cámara configurada con video local.
3. En el cuadro modal presiona **Ejecutar detección YOLOv8** (disponible para usuarios con permiso `core.add_deteccion`).
4. Opcionalmente define el número de frames a procesar y si deseas guardar las anotaciones.
5. Los resultados se reflejan en el panel dentro del resumen de detecciones y, si corresponde, generan alertas por infracciones.

### Detector continuo

- Los usuarios con permiso `core.change_detectorconfig` disponen de un bloque en el panel para activar o detener el hilo global de detección.
- El botón **Iniciar detección** lanza un proceso en segundo plano que recorre todas las cámaras con archivo local según los parámetros configurados (frames, stride, intervalo y si se guardan anotaciones).
- El botón **Detener** para el hilo y deja la detección en pausa hasta que se vuelva a activar.
- La configuración se almacena en la base de datos (`DetectorConfig`), por lo que persiste entre reinicios; al restaurar el servidor el hilo se reanudará automáticamente si quedó activo.

## Resultados registrados

- Las detecciones se almacenan en el modelo `Deteccion`, ligadas a la cámara y con bandera de casco.
- Las infracciones (`sin casco`) generan alertas automáticas de tipo `infraccion` con prioridad alta.
- El panel muestra un resumen de las últimas 24 horas y las cinco detecciones más recientes.

## Ubicación de archivos generados

- Pesos finales: `core/yolo/weights/best.pt`
- Experimentos de entrenamiento e inferencia (anotaciones): `core/yolo/weights/runs/`
- Capturas y reportes almacenados en base de datos: consultables desde Django Admin o construyendo reportes personalizados.

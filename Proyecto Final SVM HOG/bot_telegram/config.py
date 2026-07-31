"""
Configuración del Bot de Telegram para Detección de Camiones de Carga
Proyecto Integrador - Visión Artificial
Universidad Politécnica Salesiana
"""
import os
from pathlib import Path

# ============= TELEGRAM =============
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
ALLOWED_USER_IDS = [int(TELEGRAM_CHAT_ID)] if TELEGRAM_CHAT_ID else []

# ============= RUTAS =============
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parents[1]
MODELS_DIR = BASE_DIR.parent / 'models'
OUTPUTS_DIR = BASE_DIR.parent / 'outputs'
DETECTIONS_DIR = OUTPUTS_DIR / 'detections'
SEGMENTATIONS_DIR = OUTPUTS_DIR / 'segmentations'
VIDEOS_DIR = OUTPUTS_DIR / 'videos'

# Crear directorios si no existen
for directory in [OUTPUTS_DIR, DETECTIONS_DIR, SEGMENTATIONS_DIR, VIDEOS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============= MODELO YOLO =============
YOLO_CONFIG = {
    'model_name': os.getenv('YOLO_MODEL_NAME', 'yolov8n-seg.pt'),  # Ultralytics lo descarga si no existe
    'conf_threshold': 0.25,                 # Umbral de confianza
    'iou_threshold': 0.45,                  # Umbral IoU para NMS
    # COCO: person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign
    'class_ids': [0, 1, 2, 3, 5, 7, 9, 11],
    'vehicle_classes': ['bicycle', 'car', 'bus', 'truck', 'motorcycle'],
    'traffic_classes': ['traffic light', 'stop sign'],
    'target_class': 'truck',                # Clase objetivo principal
}

# ============= MODELO HOG + SVM (disparador clasico) =============
HOG_CONFIG = {
    'enabled': os.getenv('HOG_ENABLED', '0') == '1',
    'model_path': Path(os.getenv(
        'HOG_MODEL_PATH',
        PROJECT_ROOT / 'targets' / 'camiones_nuevovision' / 'models' / 'detector_lateral_latest.yml',
    )),
    'threshold': os.getenv('HOG_THRESHOLD', 'auto'),
    'max_width': 640,
    'win_stride': (16, 16),
    'padding': (32, 32),
    'scale': 1.1,
    'final_threshold': 4.0,
}

# ============= VIDEO =============
VIDEO_CONFIG = {
    'fps': 15,
    'duration': 5,      # segundos
    'format': 'mp4',
    'quality': 8,
}

# ============= VISUALIZACIÓN =============
VISUALIZATION_CONFIG = {
    'bbox_color': (0, 255, 0),        # Verde (BGR)
    'mask_alpha': 0.4,                 # Transparencia de la máscara
    'text_color': (255, 255, 255),     # Blanco
    'line_thickness': 2,
    'font_scale': 0.6,
}

# ============= MENSAJES =============
MESSAGES = {
    'detection_received': '✅ Detección recibida. Analizando con HOG+SVM y YOLOv8...',
    'processing': '⏳ Procesando imagen con disparador HOG+SVM y segmentación YOLO...',
    'vehicle_detected': '🚛 Vehículos detectados: {}',
    'sending_results': '📤 Enviando resultados...',
    'error': '❌ Error: {}',
    'welcome': '''
🤖 *Bot de Monitoreo Vial - Detector de Camiones*

Sistema inteligente de detección de vehículos del entorno ecuatoriano.

📋 *Arquitectura del sistema:*
1️⃣ HOG+SVM (C++) → Disparador rápido
2️⃣ YOLOv8 (Python) → Segmentación profunda
3️⃣ Telegram → Notificación al usuario

📸 Envía una imagen con vehículos para analizar.

*Comandos disponibles:*
/start - Iniciar el bot
/help - Ayuda
/stats - Estadísticas
/subscribe - Suscribirse a alertas automáticas
/unsubscribe - Desuscribirse
''',
}

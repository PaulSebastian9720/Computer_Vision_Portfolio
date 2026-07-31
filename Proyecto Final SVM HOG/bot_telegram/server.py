"""
Servidor HTTP local (FastAPI) que recibe alertas de la aplicación C++ (HOG+SVM)
y las procesa con YOLOv8 para enviar los resultados segmentados a Telegram.

Adaptado de pedrestrian_detector/bot_telegram/server.py
"""
import logging
import os
import uvicorn
import shutil
from fastapi import FastAPI, HTTPException, File, Form, Request, UploadFile
from pydantic import BaseModel
from pathlib import Path
import asyncio
import json
import cv2

from config import TELEGRAM_BOT_TOKEN, OUTPUTS_DIR, DETECTIONS_DIR, SEGMENTATIONS_DIR, VIDEOS_DIR
from truck_detector import TruckDetector
from video_processor import VideoProcessor
from telegram import Bot

BASE_DIR = Path(__file__).resolve().parent
SUBSCRIBERS_FILE = BASE_DIR / "subscribers.json"


def load_subscribers():
    if not os.path.exists(SUBSCRIBERS_FILE):
        return []
    with open(SUBSCRIBERS_FILE, "r") as f:
        return json.load(f)


# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    global truck_detector, video_processor, telegram_bot
    logger.info("Iniciando servidor y cargando modelos...")

    truck_detector = TruckDetector()
    video_processor = VideoProcessor(truck_detector)
    telegram_bot = Bot(token=TELEGRAM_BOT_TOKEN)

    logger.info("✅ Servidor listo para recibir peticiones del detector C++.")
    yield
    logger.info("Cerrando servidor...")


app = FastAPI(title="Truck Detection Server - Monitoreo Vial", lifespan=lifespan)

# Modelos (se cargan al inicio)
truck_detector = None
video_processor = None
telegram_bot = None


class DetectionRequest(BaseModel):
    file_path: str


def _detection_score(detection_result: dict) -> tuple:
    conf_sum = sum(det.get('confidence', 0.0) for det in detection_result.get('detections', []))
    return (
        1 if detection_result.get('num_hog_triggers', 0) > 0 else 0,
        detection_result.get('num_trucks', 0),
        detection_result.get('num_vehicles', 0),
        detection_result.get('num_hog_triggers', 0),
        conf_sum,
    )


def _as_float(value, default):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _make_hog_filters(
    min_center_y=None,
    max_center_y=None,
    min_area=None,
    max_area=None,
    roi_min_x=None,
    roi_max_x=None,
):
    return {
        'min_center_y': _as_float(min_center_y, 0.0),
        'max_center_y': _as_float(max_center_y, 1.0),
        'min_area': _as_float(min_area, 0.0),
        'max_area': _as_float(max_area, 1.0),
        'roi_min_x': _as_float(roi_min_x, 0.0),
        'roi_max_x': _as_float(roi_max_x, 1.0),
    }


def _best_yolo_frame_from_clip(clip_path: Path, hog_threshold=None, hog_filters=None):
    """Seleccionar un frame del clip donde YOLO tenga la mejor detección."""
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(fps / 10)) if fps and fps > 0 else 3
    frame_index = 0
    best_frame = None
    best_result = None
    best_score = (-1, -1, -1, -1, -1.0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % step == 0 or frame_index == total_frames - 1:
            detection_result = truck_detector.detect(
                frame,
                hog_threshold=hog_threshold,
                hog_filters=hog_filters,
            )
            score = _detection_score(detection_result)
            if score > best_score:
                best_score = score
                best_frame = frame.copy()
                best_result = detection_result

        frame_index += 1

    cap.release()
    return best_frame, best_result


def _best_hog_frame_from_clip(clip_path: Path, hog_threshold=None, hog_filters=None):
    """Seleccionar un frame del clip donde HOG marque candidatos sin relajar el umbral."""
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(fps / 10)) if fps and fps > 0 else 3
    frame_index = 0
    best_frame = None
    best_result = None
    best_score = (-1, -1, -1, -1.0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % step == 0 or frame_index == total_frames - 1:
            detection_result = truck_detector.detect(
                frame,
                hog_threshold=hog_threshold,
                hog_filters=hog_filters,
            )
            conf_sum = sum(det.get('confidence', 0.0) for det in detection_result.get('detections', []))
            score = (
                1 if detection_result.get('num_hog_triggers', 0) > 0 else 0,
                detection_result.get('num_trucks', 0),
                detection_result.get('num_hog_triggers', 0),
                conf_sum,
            )
            if score > best_score:
                best_score = score
                best_frame = frame.copy()
                best_result = detection_result

        frame_index += 1

    cap.release()
    return best_frame, best_result


@app.get("/")
async def root():
    return {"status": "running", "message": "Truck Detection Server activo"}


@app.post("/detect")
async def detect_vehicle(
    request: Request,
    file: UploadFile | None = File(None),
    hog_file: UploadFile | None = File(None),
    clip: UploadFile | None = File(None),
    hog_threshold: str | None = Form(None),
    hog_count: str | None = Form(None),
    min_center_y: str | None = Form(None),
    max_center_y: str | None = Form(None),
    min_area: str | None = Form(None),
    max_area: str | None = Form(None),
    roi_min_x: str | None = Form(None),
    roi_max_x: str | None = Form(None),
):
    """
    Endpoint principal: recibe el archivo (imagen o video) directamente en el payload
    de la petición de red (multipart/form-data), lo guarda temporalmente y lo procesa.
    """
    try:
        clip_file_path = None
        hog_file_path = None

        if file is not None:
            filename = file.filename
            temp_dir = OUTPUTS_DIR / "temp"
            temp_dir.mkdir(exist_ok=True)
            temp_file_path = temp_dir / filename
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(f"📥 Archivo recibido vía API: {filename} -> {temp_file_path}")

            if hog_file is not None:
                hog_file_path = temp_dir / hog_file.filename
                with open(hog_file_path, "wb") as buffer:
                    shutil.copyfileobj(hog_file.file, buffer)
                logger.info(f"🟨 Frame HOG recibido vía API: {hog_file.filename} -> {hog_file_path}")

            if clip is not None:
                clip_file_path = temp_dir / clip.filename
                with open(clip_file_path, "wb") as buffer:
                    shutil.copyfileobj(clip.file, buffer)
                logger.info(f"🎞️ Clip recibido vía API: {clip.filename} -> {clip_file_path}")
        else:
            payload = await request.json()
            file_path = payload.get("file_path")
            if not file_path:
                raise HTTPException(status_code=400, detail="Falta file_path o archivo multipart")
            temp_file_path = Path(file_path)
            if not temp_file_path.exists():
                raise HTTPException(status_code=404, detail=f"No existe: {temp_file_path}")
            logger.info(f"📥 Archivo recibido por ruta local desde C++: {temp_file_path}")

        lower_suffix = temp_file_path.suffix.lower()
        hog_filters = _make_hog_filters(
            min_center_y=min_center_y,
            max_center_y=max_center_y,
            min_area=min_area,
            max_area=max_area,
            roi_min_x=roi_min_x,
            roi_max_x=roi_max_x,
        )

        if lower_suffix in ['.jpg', '.jpeg', '.png']:
            await process_image(
                temp_file_path,
                clip_file_path,
                hog_threshold,
                hog_filters,
                hog_file_path=hog_file_path,
                hog_count=hog_count,
            )
        elif lower_suffix in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
            await process_video(temp_file_path)
        else:
            raise HTTPException(status_code=400, detail=f"Formato no soportado: {lower_suffix}")

        return {"status": "success", "message": "Procesado y enviado a Telegram"}

    except Exception as e:
        logger.error(f"Error procesando archivo subido: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_image(
    image_path: Path,
    clip_path: Path | None = None,
    hog_threshold=None,
    hog_filters=None,
    hog_file_path: Path | None = None,
    hog_count=None,
):
    """Procesar imagen: detectar vehículos con YOLO y enviar a Telegram."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("Error leyendo imagen")

    # Detectar la captura exacta del disparo. Si viene clip, usar el mejor frame
    # del clip para que la evidencia HOG y la evidencia YOLO sean del mismo momento.
    detection_result = truck_detector.detect(
        image,
        hog_threshold=hog_threshold,
        hog_filters=hog_filters,
    )
    hog_image_source = image
    hog_result = detection_result
    yolo_image = image
    yolo_result = detection_result
    if clip_path is not None and hog_file_path is None:
        best_frame, best_result = _best_yolo_frame_from_clip(
            clip_path,
            hog_threshold=hog_threshold,
            hog_filters=hog_filters,
        )
        if best_frame is not None and best_result is not None:
            hog_image_source = best_frame
            hog_result = best_result
            yolo_image = best_frame
            yolo_result = best_result

    num_vehicles = yolo_result['num_vehicles']
    num_trucks = yolo_result['num_trucks']
    num_objects = yolo_result.get('num_objects', num_vehicles)

    # Guardar imágenes separadas: HOG+SVM y YOLO, ambas desde el mejor frame disponible.
    image_hog = (
        cv2.imread(str(hog_file_path))
        if hog_file_path is not None
        else truck_detector.draw_hog_detections(hog_image_source, hog_result)
    )
    if image_hog is None:
        image_hog = truck_detector.draw_hog_detections(hog_image_source, hog_result)
    image_segmented = truck_detector.draw_yolo_detections(yolo_image, yolo_result)

    timestamp = image_path.stem.replace('detection_', '')
    hog_image_path = SEGMENTATIONS_DIR / f"hog_{timestamp}.jpg"
    seg_image_path = SEGMENTATIONS_DIR / f"seg_{timestamp}.jpg"
    cv2.imwrite(str(hog_image_path), image_hog)
    cv2.imwrite(str(seg_image_path), image_segmented)
    if clip_path is not None:
        _, alert_video_path = video_processor.process_video_file(str(clip_path))
    else:
        alert_video_path = video_processor.create_short_clip(image, detection_result)

    # Generar texto de resumen
    if hog_count is not None:
        try:
            hog_summary = f"  HOG+SVM camión: {int(float(hog_count))} candidato(s)"
        except (TypeError, ValueError):
            hog_summary = truck_detector.get_hog_summary_text(hog_result)
    else:
        hog_summary = truck_detector.get_hog_summary_text(hog_result)
    yolo_summary = truck_detector.get_summary_text(yolo_result)
    stats = truck_detector.get_statistics(yolo_result)

    # Enviar a todos los suscriptores en Telegram
    subscribers = load_subscribers()

    try:
        for chat_id in subscribers:
            # 1. Imagen de alerta solo con HOG+SVM
            hog_caption = (
                f"🚨 *ALERTA DE MONITOREO VIAL* 🚨\n\n"
                f"📷 *Vehículo objetivo camión de carga detectado en la escena*\n\n"
                f"🔎 *Detección HOG+SVM:*\n{hog_summary}"
            )
            with open(hog_image_path, 'rb') as f:
                await telegram_bot.send_photo(
                    chat_id=chat_id,
                    photo=f,
                    caption=hog_caption,
                    parse_mode='Markdown',
                )

            # 2. Imagen segmentada solo por YOLO
            caption = (
                f"🔍 *Resultados YOLOv8 (Segmentación):*\n"
                f"{yolo_summary}\n\n"
                f"📊 *Estadísticas:*\n"
                f"  🔹 Total objetos YOLO: {num_objects}\n"
                f"  🚗 Vehículos: {num_vehicles}\n"
                f"  🚛 Camiones: {num_trucks}\n"
                f"  🚶 Personas: {stats.get('num_people', 0)}\n"
                f"  🚦 Señales/semáforos: {stats.get('num_traffic_objects', 0)}\n"
                f"  📈 Confianza promedio: {stats['avg_confidence']:.2%}"
            )
            with open(seg_image_path, 'rb') as f:
                await telegram_bot.send_photo(
                    chat_id=chat_id,
                    photo=f,
                    caption=caption,
                    parse_mode='Markdown',
                )

            # 3. Clip corto con segmentación aplicada
            with open(alert_video_path, 'rb') as f:
                await telegram_bot.send_video(
                    chat_id=chat_id,
                    video=f,
                    caption="🎞️ *Clip corto con segmentación aplicada*",
                    parse_mode='Markdown',
                )

        logger.info(f"✅ Enviado a {len(subscribers)} suscriptor(es): {num_vehicles} vehículos, {num_trucks} camiones")

    except Exception as e:
        logger.error(f"Error enviando a Telegram: {e}")


async def process_video(video_path: Path):
    """Procesar video: detectar vehículos frame a frame y enviar a Telegram."""
    subscribers = load_subscribers()

    try:
        for chat_id in subscribers:
            await telegram_bot.send_message(chat_id=chat_id, text="🎥 Procesando video con YOLOv8...")

        # Procesar video completo
        original_path, seg_video_path = video_processor.process_video_file(str(video_path))
        video_info = video_processor.get_video_info(seg_video_path)

        caption = (
            f"🎥 *Video Analizado con YOLOv8*\n"
            f"⏱️ Duración: {video_info.get('duration', 0):.1f}s\n"
            f"🎞️ Frames: {video_info.get('total_frames', 0)}\n"
            f"📊 FPS: {video_info.get('fps', 0)}"
        )

        for chat_id in subscribers:
            with open(original_path, 'rb') as f:
                await telegram_bot.send_video(
                    chat_id=chat_id,
                    video=f,
                    caption="📹 *Video original del disparador HOG+SVM*",
                    parse_mode='Markdown',
                )

            with open(seg_video_path, 'rb') as f:
                await telegram_bot.send_video(
                    chat_id=chat_id,
                    video=f,
                    caption=caption,
                    parse_mode='Markdown',
                )

        logger.info(f"✅ Video procesado y enviado a {len(subscribers)} suscriptor(es)")

    except Exception as e:
        logger.error(f"Error procesando video: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

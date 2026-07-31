"""
Bot de Telegram para Detección de Camiones de Carga
Recibe imágenes/videos desde la aplicación C++ y realiza segmentación con YOLOv8.
Adaptado de pedrestrian_detector/bot_telegram/main.py
"""
import os
import logging
from pathlib import Path
from datetime import datetime
import asyncio
import cv2
import numpy as np
import json

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

BASE_DIR = Path(__file__).resolve().parent
SUBSCRIBERS_FILE = BASE_DIR / "subscribers.json"


def load_subscribers():
    if not os.path.exists(SUBSCRIBERS_FILE):
        return []
    with open(SUBSCRIBERS_FILE, "r") as f:
        return json.load(f)


def save_subscribers(subscribers):
    with open(SUBSCRIBERS_FILE, "w") as f:
        json.dump(subscribers, f)


def add_subscriber(chat_id):
    subs = load_subscribers()
    if chat_id not in subs:
        subs.append(chat_id)
        save_subscribers(subs)


def remove_subscriber(chat_id):
    subs = load_subscribers()
    if chat_id in subs:
        subs.remove(chat_id)
        save_subscribers(subs)


from config import (
    TELEGRAM_BOT_TOKEN,
    ALLOWED_USER_IDS,
    MESSAGES,
    DETECTIONS_DIR,
    SEGMENTATIONS_DIR,
    VIDEOS_DIR,
)
from truck_detector import TruckDetector
from video_processor import VideoProcessor

# Configurar logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Inicializar detectores
truck_detector = TruckDetector()
video_processor = VideoProcessor(truck_detector)

# Estadísticas globales
stats = {
    'total_images': 0,
    'total_videos': 0,
    'total_vehicles_detected': 0,
    'total_trucks_detected': 0,
    'start_time': datetime.now(),
}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler para el comando /start"""
    user = update.effective_user
    logger.info(f"Usuario {user.id} - {user.username} inició el bot")

    keyboard = [
        [InlineKeyboardButton("📊 Ver Estadísticas", callback_data='stats')],
        [InlineKeyboardButton("❓ Ayuda", callback_data='help')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        MESSAGES['welcome'],
        reply_markup=reply_markup,
        parse_mode='Markdown',
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler para el comando /help"""
    help_text = """
📖 *Ayuda - Bot de Monitoreo Vial*

*¿Cómo funciona el sistema?*
1️⃣ La app C++ (HOG+SVM) vigila la cámara en tiempo real
2️⃣ Cuando detecta un posible camión, envía la captura a este bot
3️⃣ YOLOv8 analiza la imagen con segmentación de instancias
4️⃣ Recibes: imagen original + imagen segmentada + video

*Uso manual:*
📸 Envía una imagen con vehículos para analizarla
🎥 Envía un video corto para procesar frame a frame

*Comandos:*
/start - Iniciar el bot
/help - Mostrar esta ayuda
/stats - Ver estadísticas de uso
/subscribe - Recibir alertas automáticas
/unsubscribe - Dejar de recibir alertas
    """
    msg = update.message or update.callback_query.message
    await msg.reply_text(help_text, parse_mode='Markdown')


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler para el comando /stats"""
    uptime = datetime.now() - stats['start_time']

    stats_text = f"""
📊 *Estadísticas del Bot de Monitoreo Vial*

🖼️ Imágenes procesadas: {stats['total_images']}
🎥 Videos procesados: {stats['total_videos']}
🚗 Vehículos detectados: {stats['total_vehicles_detected']}
🚛 Camiones detectados: {stats['total_trucks_detected']}
⏱️ Tiempo activo: {uptime.days}d {uptime.seconds // 3600}h {(uptime.seconds // 60) % 60}m

💡 *Información del sistema:*
🤖 Disparador: HOG+SVM (OpenCV C++)
🎯 Segmentación: YOLOv8n-seg (Ultralytics)
📍 Confianza mínima: 25%
    """
    msg = update.message or update.callback_query.message
    await msg.reply_text(stats_text, parse_mode='Markdown')


async def subscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    add_subscriber(chat_id)
    await update.message.reply_text("✅ Te has suscrito a las alertas de monitoreo vial.")


async def unsubscribe(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    remove_subscriber(chat_id)
    await update.message.reply_text("❌ Te has desuscrito de las alertas.")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler para imágenes enviadas manualmente al bot."""
    user = update.effective_user
    logger.info(f"Imagen recibida de {user.username}")

    status_msg = await update.message.reply_text(MESSAGES['detection_received'])

    try:
        photo = update.message.photo[-1]
        file = await photo.get_file()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = DETECTIONS_DIR / f"detection_{timestamp}.jpg"
        await file.download_to_drive(str(image_path))

        image = cv2.imread(str(image_path))
        if image is None:
            await status_msg.edit_text("❌ Error al leer la imagen")
            return

        await status_msg.edit_text(MESSAGES['processing'])

        # Detectar candidatos con HOG+SVM y segmentar con YOLO
        detection_result = truck_detector.detect(image)
        num_vehicles = detection_result['num_vehicles']
        num_trucks = detection_result['num_trucks']
        num_objects = detection_result.get('num_objects', num_vehicles)
        num_hog = detection_result.get('num_hog_triggers', 0)

        # Actualizar estadísticas
        stats['total_images'] += 1
        stats['total_vehicles_detected'] += num_vehicles
        stats['total_trucks_detected'] += num_trucks

        # Guardar salidas separadas: HOG+SVM para alerta y YOLO para segmentación.
        image_hog = truck_detector.draw_hog_detections(image, detection_result)
        hog_path = SEGMENTATIONS_DIR / f"hog_{timestamp}.jpg"
        cv2.imwrite(str(hog_path), image_hog)

        image_segmented = truck_detector.draw_yolo_detections(image, detection_result)
        seg_path = SEGMENTATIONS_DIR / f"seg_{timestamp}.jpg"
        cv2.imwrite(str(seg_path), image_segmented)
        clip_path = video_processor.create_short_clip(image, detection_result)

        # Generar resumen
        hog_summary = truck_detector.get_hog_summary_text(detection_result)
        yolo_summary = truck_detector.get_summary_text(detection_result)
        det_stats = truck_detector.get_statistics(detection_result)

        await status_msg.edit_text(MESSAGES['sending_results'])

        # 1. Alerta del disparador HOG+SVM
        hog_caption = (
            f"🚨 *ALERTA DE MONITOREO VIAL* 🚨\n\n"
            f"📷 *Vehículo objetivo camión de carga detectado en la escena*\n\n"
            f"🔎 *Detección HOG+SVM:*\n{hog_summary}"
        )
        with open(hog_path, 'rb') as f:
            await update.message.reply_photo(
                photo=f,
                caption=hog_caption,
                parse_mode='Markdown',
            )

        # 2. Imagen segmentada solo con YOLO
        caption = (
            f"🔍 *Resultados YOLOv8 (Segmentación):*\n"
            f"{yolo_summary}\n\n"
            f"📊 *Estadísticas:*\n"
            f"  🔹 Total objetos YOLO: {num_objects}\n"
            f"  🚗 Vehículos: {num_vehicles}\n"
            f"  🚛 Camiones: {num_trucks}\n"
            f"  🚶 Personas: {det_stats.get('num_people', 0)}\n"
            f"  🚦 Señales/semáforos: {det_stats.get('num_traffic_objects', 0)}\n"
            f"  📈 Confianza promedio: {det_stats['avg_confidence']:.2%}"
        )
        with open(seg_path, 'rb') as f:
            await update.message.reply_photo(
                photo=f,
                caption=caption,
                parse_mode='Markdown',
            )

        # 3. Clip corto solicitado por la guía
        with open(clip_path, 'rb') as f:
            await update.message.reply_video(
                video=f,
                caption="🎞️ *Clip corto con segmentación aplicada*",
                parse_mode='Markdown',
            )

        await status_msg.delete()
        logger.info(f"Procesamiento completado: {num_vehicles} vehículo(s), {num_trucks} camión(es)")

    except Exception as e:
        logger.error(f"Error procesando imagen: {e}", exc_info=True)
        await status_msg.edit_text(MESSAGES['error'].format(str(e)))


async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler para videos enviados manualmente al bot."""
    user = update.effective_user
    logger.info(f"Video recibido de {user.username}")

    status_msg = await update.message.reply_text(MESSAGES['detection_received'])

    try:
        video = update.message.video
        file = await video.get_file()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = DETECTIONS_DIR / f"video_{timestamp}.mp4"
        await file.download_to_drive(str(video_path))

        await status_msg.edit_text("🎬 Procesando video con HOG+SVM + YOLOv8...")

        # Procesar video
        original_path, seg_video_path = video_processor.process_video_file(str(video_path))

        # Estadísticas del primer frame
        cap = cv2.VideoCapture(str(video_path))
        ret, first_frame = cap.read()
        cap.release()

        if ret:
            detection_result = truck_detector.detect(first_frame)
            num_vehicles = detection_result['num_vehicles']
            num_trucks = detection_result['num_trucks']
            num_objects = detection_result.get('num_objects', num_vehicles)
            num_hog = detection_result.get('num_hog_triggers', 0)
            stats['total_vehicles_detected'] += num_vehicles
            stats['total_trucks_detected'] += num_trucks
            frame_hog = truck_detector.draw_hog_detections(first_frame, detection_result)
            frame_path = SEGMENTATIONS_DIR / f"video_hog_frame_{timestamp}.jpg"
            cv2.imwrite(str(frame_path), frame_hog)
        else:
            num_vehicles = 0
            num_trucks = 0
            num_objects = 0
            num_hog = 0
            frame_path = None

        stats['total_videos'] += 1

        video_info = video_processor.get_video_info(seg_video_path)

        await status_msg.edit_text(MESSAGES['sending_results'])

        # 1. Video original
        with open(original_path, 'rb') as f:
            await update.message.reply_video(
                video=f,
                caption="📹 *Video Original*",
                parse_mode='Markdown',
            )

        # 2. Frame clave del disparador HOG+SVM
        if frame_path:
            with open(frame_path, 'rb') as f:
                await update.message.reply_photo(
                    photo=f,
                    caption="🖼️ *Frame clave con detección HOG+SVM*",
                    parse_mode='Markdown',
                )

        # 3. Video segmentado solo con YOLO
        caption = (
            f"🎯 *Video procesado con YOLOv8 (Segmentación)*\n\n"
            f"🟨 Candidatos HOG+SVM en frame inicial: {num_hog}\n"
            f"🔹 Objetos YOLO en frame inicial: {num_objects}\n"
            f"🚗 Vehículos detectados: {num_vehicles}\n"
            f"🚛 Camiones YOLO: {num_trucks}\n"
            f"⏱️ Duración: {video_info['duration']:.1f}s\n"
            f"🎞️ Frames: {video_info['total_frames']}\n"
            f"📊 FPS: {video_info['fps']}"
        )
        with open(seg_video_path, 'rb') as f:
            await update.message.reply_video(
                video=f,
                caption=caption,
                parse_mode='Markdown',
            )

        await status_msg.delete()
        logger.info(f"Video procesado: {num_vehicles} vehículo(s)")

    except Exception as e:
        logger.error(f"Error procesando video: {e}", exc_info=True)
        await status_msg.edit_text(MESSAGES['error'].format(str(e)))


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler para documentos."""
    await update.message.reply_text(
        "📎 He recibido un documento. Por favor, envía imágenes como fotos o videos como archivos de video."
    )


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler para botones inline."""
    query = update.callback_query
    await query.answer()

    if query.data == 'stats':
        await stats_command(update, context)
    elif query.data == 'help':
        await help_command(update, context)


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler global de errores."""
    logger.error(f"Error: {context.error}", exc_info=context.error)
    if update and update.effective_message:
        await update.effective_message.reply_text(
            "❌ Ha ocurrido un error. Por favor, intenta nuevamente."
        )


async def post_init(application: Application):
    """Ejecutar después de inicializar la aplicación."""
    logger.info("Bot de Monitoreo Vial iniciado correctamente")
    logger.info(f"Detecciones: {DETECTIONS_DIR}")
    logger.info(f"Segmentaciones: {SEGMENTATIONS_DIR}")
    logger.info(f"Videos: {VIDEOS_DIR}")


def main():
    """Función principal."""
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN no configurado!")
        logger.error("Configura tu token en el archivo .env")
        return

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Handlers de comandos
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("stats", stats_command))
    application.add_handler(CommandHandler("subscribe", subscribe))
    application.add_handler(CommandHandler("unsubscribe", unsubscribe))

    # Handlers para multimedia
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.VIDEO, handle_video))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    # Botones y errores
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_error_handler(error_handler)

    application.post_init = post_init

    logger.info("Iniciando bot de Telegram...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == '__main__':
    main()

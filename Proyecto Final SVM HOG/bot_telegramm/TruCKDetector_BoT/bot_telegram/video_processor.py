"""
Módulo de Procesamiento de Video para Detección de Vehículos
Adaptado de pedrestrian_detector/bot_telegram/video_processor.py
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
import logging
from datetime import datetime

import os
import subprocess
from config import VIDEO_CONFIG, VIDEOS_DIR

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Procesador de videos con detección de vehículos"""

    def __init__(self, truck_detector):
        self.truck_detector = truck_detector
        self.fps = VIDEO_CONFIG['fps']
        self.duration = VIDEO_CONFIG['duration']
        self.video_format = VIDEO_CONFIG['format']

    def process_video_file(self, video_path: str, output_dir: Path = None) -> Tuple[str, str]:
        """
        Procesar archivo de video completo con detección YOLO.

        Returns:
            (path_original_video, path_segmented_video)
        """
        if output_dir is None:
            output_dir = VIDEOS_DIR

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"Procesando video: {total_frames} frames @ {fps} FPS")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"segmented_{timestamp}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frames_processed = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detection_result = self.truck_detector.detect(frame)
            frame_annotated = self.truck_detector.draw_detections(frame, detection_result)
            out.write(frame_annotated)
            frames_processed += 1

            if frames_processed % 30 == 0:
                logger.info(f"Procesados {frames_processed}/{total_frames} frames")

        cap.release()
        out.release()
        logger.info(f"Video procesado en OpenCV: {output_path}")

        # Convertir a H.264 (yuv420p) usando ffmpeg para compatibilidad con Telegram
        temp_path = str(output_path) + ".temp.mp4"
        try:
            os.rename(str(output_path), temp_path)
            cmd = [
                'ffmpeg', '-y', '-i', temp_path,
                '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
                str(output_path)
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            if os.path.exists(temp_path):
                os.remove(temp_path)
            logger.info(f"Video convertido exitosamente a H.264: {output_path}")
        except Exception as e:
            logger.error(f"Error al convertir video con ffmpeg: {e}")
            if os.path.exists(temp_path):
                if os.path.exists(str(output_path)):
                    os.remove(temp_path)
                else:
                    os.rename(temp_path, str(output_path))

        return video_path, str(output_path)

    def create_short_clip(
        self,
        image: np.ndarray,
        detection_result: dict,
        duration: float = None,
    ) -> str:
        """
        Crear clip corto animado desde una imagen con detección.
        Simula movimiento con zoom/desplazamiento suave.
        """
        if duration is None:
            duration = self.duration

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(VIDEOS_DIR / f"clip_{timestamp}.mp4")

        img_annotated = self.truck_detector.draw_detections(image, detection_result)

        h, w = image.shape[:2]
        num_frames = int(duration * self.fps)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (w, h))

        for i in range(num_frames):
            scale = 1.0 + 0.1 * np.sin(2 * np.pi * i / num_frames)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(img_annotated, (new_w, new_h))

            if new_w > w or new_h > h:
                sx = (new_w - w) // 2
                sy = (new_h - h) // 2
                frame = resized[sy:sy + h, sx:sx + w]
            else:
                frame = cv2.resize(resized, (w, h))

            out.write(frame)

        out.release()
        logger.info(f"Clip corto creado: {output_path}")
        return output_path

    def get_video_info(self, video_path: str) -> dict:
        """Obtener información del video."""
        cap = cv2.VideoCapture(video_path)
        fps_val = cap.get(cv2.CAP_PROP_FPS)
        info = {
            'fps': int(fps_val),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(1, fps_val),
        }
        cap.release()
        return info

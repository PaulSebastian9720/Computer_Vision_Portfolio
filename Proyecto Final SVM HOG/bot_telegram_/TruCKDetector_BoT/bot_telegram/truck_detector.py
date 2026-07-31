"""
Módulo de Detección y Segmentación de Vehículos usando YOLOv8
Reemplaza pose_detector.py de la plantilla pedrestrian_detector
"""
import cv2
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    logger.warning("Ultralytics no disponible. Instalar con: pip install ultralytics")
    YOLO_AVAILABLE = False

from config import HOG_CONFIG, YOLO_CONFIG, VISUALIZATION_CONFIG


class TruckDetector:
    """Detector HOG+SVM y segmentador de vehículos usando YOLOv8."""

    def __init__(self):
        self.conf_threshold = YOLO_CONFIG['conf_threshold']
        self.iou_threshold = YOLO_CONFIG['iou_threshold']
        self.yolo_class_ids = YOLO_CONFIG.get('class_ids')
        self.vehicle_classes = YOLO_CONFIG['vehicle_classes']
        self.traffic_classes = YOLO_CONFIG.get('traffic_classes', [])
        self.target_class = YOLO_CONFIG['target_class']
        self.hog = None
        self.hog_threshold = 0.0
        self._load_hog_detector()

        if YOLO_AVAILABLE:
            logger.info("Cargando modelo YOLOv8 de segmentación...")
            self.model = YOLO(YOLO_CONFIG['model_name'])
            logger.info("Modelo YOLOv8 cargado exitosamente.")
        else:
            logger.warning("YOLOv8 no disponible. Modo simulado.")
            self.model = None

    def _load_hog_detector(self):
        """Cargar detector HOG+SVM de OpenCV exportado en YAML."""
        if not HOG_CONFIG.get('enabled', True):
            logger.info("HOG+SVM deshabilitado por configuración.")
            return

        model_path = HOG_CONFIG['model_path']
        fs = cv2.FileStorage(str(model_path), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            logger.warning(f"No se pudo abrir modelo HOG+SVM: {model_path}")
            return

        win_width = int(fs.getNode("win_width").real())
        win_height = int(fs.getNode("win_height").real())
        detector = fs.getNode("svm_detector").mat()
        file_threshold = fs.getNode("threshold").real()
        fs.release()

        if detector is None or win_width <= 0 or win_height <= 0:
            logger.warning(f"Modelo HOG+SVM inválido: {model_path}")
            return

        threshold_arg = str(HOG_CONFIG.get('threshold', 'auto')).lower()
        self.hog_threshold = float(file_threshold) if threshold_arg == 'auto' else float(threshold_arg)
        self.hog = cv2.HOGDescriptor((win_width, win_height), (16, 16), (8, 8), (8, 8), 9)
        self.hog.setSVMDetector(detector.reshape(-1).astype(np.float32))
        logger.info(f"HOG+SVM cargado: {model_path} threshold={self.hog_threshold:.3f}")

    def _passes_hog_filters(self, bbox, image_shape, filters=None) -> bool:
        if not filters:
            return True

        height, width = image_shape[:2]
        x1, y1, x2, y2 = bbox
        box_width = max(0, x2 - x1)
        box_height = max(0, y2 - y1)
        frame_area = max(1, width * height)
        area_fraction = (box_width * box_height) / frame_area
        center_x = (x1 + box_width * 0.5) / max(1, width)
        center_y = (y1 + box_height * 0.5) / max(1, height)

        return (
            center_x >= filters.get('roi_min_x', 0.0)
            and center_x <= filters.get('roi_max_x', 1.0)
            and center_y >= filters.get('min_center_y', 0.0)
            and center_y <= filters.get('max_center_y', 1.0)
            and area_fraction >= filters.get('min_area', 0.0)
            and area_fraction <= filters.get('max_area', 1.0)
        )

    def _detect_hog(self, image: np.ndarray, threshold_override=None, filters=None) -> list:
        if self.hog is None:
            return []

        frame = image
        scale_back = 1.0
        max_width = HOG_CONFIG.get('max_width', 640)
        if frame.shape[1] > max_width:
            scale = max_width / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            scale_back = 1.0 / scale

        hit_threshold = self.hog_threshold if threshold_override is None else float(threshold_override)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes, weights = self.hog.detectMultiScale(
            gray,
            hit_threshold,
            HOG_CONFIG.get('win_stride', (8, 8)),
            HOG_CONFIG.get('padding', (32, 32)),
            HOG_CONFIG.get('scale', 1.1),
            HOG_CONFIG.get('final_threshold', 4.0),
            False,
        )

        detections = []
        for box, weight in zip(boxes, weights):
            x, y, w, h = [int(v * scale_back) for v in box]
            bbox = [x, y, x + w, y + h]
            if not self._passes_hog_filters(bbox, image.shape, filters):
                continue
            detections.append({
                'class': 'camion_hog_svm',
                'confidence': float(weight),
                'bbox': bbox,
            })
        return detections

    def detect(self, image: np.ndarray, hog_threshold=None, hog_filters=None) -> Dict:
        """
        Detectar y segmentar vehículos en la imagen.

        Returns:
            dict con claves: detections, num_vehicles, num_trucks, image_shape, raw_results
        """
        hog_detections = self._detect_hog(image, hog_threshold, hog_filters)

        if self.model is None:
            result = self._empty_result(image)
            result['hog_detections'] = hog_detections
            result['num_hog_triggers'] = len(hog_detections)
            return result

        try:
            results = self.model(
                image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                classes=self.yolo_class_ids,
                verbose=False,
            )
            r = results[0]

            detections = []
            num_trucks = 0
            num_vehicles = 0
            num_people = 0
            num_traffic_objects = 0

            if r.boxes is not None:
                for i, box in enumerate(r.boxes):
                    cls_id = int(box.cls[0])
                    cls_name = self.model.names[cls_id]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    det = {
                        'class': cls_name,
                        'confidence': conf,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    }

                    if cls_name == self.target_class:
                        num_trucks += 1
                    if cls_name in self.vehicle_classes:
                        num_vehicles += 1
                    if cls_name == 'person':
                        num_people += 1
                    if cls_name in self.traffic_classes:
                        num_traffic_objects += 1

                    detections.append(det)

            return {
                'detections': detections,
                'num_objects': len(detections),
                'num_vehicles': num_vehicles,
                'num_trucks': num_trucks,
                'num_people': num_people,
                'num_traffic_objects': num_traffic_objects,
                'hog_detections': hog_detections,
                'num_hog_triggers': len(hog_detections),
                'image_shape': image.shape[:2],
                'raw_results': r,
            }

        except Exception as e:
            logger.error(f"Error en detección YOLO: {e}")
            result = self._empty_result(image)
            result['hog_detections'] = hog_detections
            result['num_hog_triggers'] = len(hog_detections)
            return result

    def draw_hog_detections(self, image: np.ndarray, detection_result: Dict) -> np.ndarray:
        """Dibujar solo los candidatos del disparador HOG+SVM."""
        annotated = image.copy()

        for det in detection_result.get('hog_detections', []):
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                annotated,
                f"HOG+SVM {det['confidence']:.2f}",
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        num_hog = detection_result.get('num_hog_triggers', 0)
        cv2.putText(
            annotated,
            f"HOG+SVM candidatos: {num_hog}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        return annotated

    def draw_yolo_detections(self, image: np.ndarray, detection_result: Dict) -> np.ndarray:
        """Dibujar solo bboxes, máscaras y etiquetas de YOLOv8."""
        if 'raw_results' not in detection_result or detection_result['raw_results'] is None:
            annotated = image.copy()
        else:
            r = detection_result['raw_results']
            # Usar el ploteo integrado de Ultralytics (incluye máscaras de segmentación)
            annotated = r.plot()

        # Agregar resumen en la esquina superior
        num_o = detection_result.get('num_objects', detection_result.get('num_vehicles', 0))
        num_v = detection_result['num_vehicles']
        num_t = detection_result['num_trucks']
        label = f"Objetos: {num_o} | Vehiculos: {num_v} | Camiones: {num_t}"
        cv2.putText(
            annotated, label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )

        return annotated

    def draw_detections(self, image: np.ndarray, detection_result: Dict) -> np.ndarray:
        """Mantener la salida anterior del bot: segmentación YOLO sin HOG encima."""
        return self.draw_yolo_detections(image, detection_result)

    def get_hog_summary_text(self, detection_result: Dict) -> str:
        """Generar texto de resumen solo del disparador HOG+SVM."""
        hog_detections = detection_result.get('hog_detections', [])
        if not hog_detections:
            return "  HOG+SVM camión: 0 candidato(s)"

        best_score = max(det['confidence'] for det in hog_detections)
        return (
            f"  HOG+SVM camión: {len(hog_detections)} candidato(s)\n"
            f"  Mejor score HOG+SVM: {best_score:.2f}"
        )

    def get_summary_text(self, detection_result: Dict) -> str:
        """Generar texto de resumen YOLO para enviar a Telegram."""
        dets = detection_result.get('detections', [])
        if not dets:
            return "No se detectaron vehículos en la imagen."

        # Contar por clase
        class_counts = {}
        for d in dets:
            cls = d['class']
            class_counts[cls] = class_counts.get(cls, 0) + 1

        lines = []
        for cls, count in sorted(class_counts.items()):
            emoji = (
                '🚛' if cls == 'truck' else
                '🚗' if cls == 'car' else
                '🚌' if cls == 'bus' else
                '🏍️' if cls == 'motorcycle' else
                '🚲' if cls == 'bicycle' else
                '🚶' if cls == 'person' else
                '🚦' if cls == 'traffic light' else
                '🛑' if cls == 'stop sign' else
                '🔹'
            )
            lines.append(f"  {emoji} {cls}: {count}")

        return "\n".join(lines)

    def get_statistics(self, detection_result: Dict) -> Dict:
        """Obtener estadísticas de la detección."""
        dets = detection_result.get('detections', [])
        confs = [d['confidence'] for d in dets] if dets else [0]
        return {
            'num_objects': detection_result.get('num_objects', detection_result.get('num_vehicles', 0)),
            'num_vehicles': detection_result.get('num_vehicles', 0),
            'num_trucks': detection_result.get('num_trucks', 0),
            'num_people': detection_result.get('num_people', 0),
            'num_traffic_objects': detection_result.get('num_traffic_objects', 0),
            'num_hog_triggers': detection_result.get('num_hog_triggers', 0),
            'avg_confidence': sum(confs) / len(confs),
            'max_confidence': max(confs) if confs else 0,
        }

    def _empty_result(self, image: np.ndarray) -> Dict:
        return {
            'detections': [],
            'num_objects': 0,
            'num_vehicles': 0,
            'num_trucks': 0,
            'num_people': 0,
            'num_traffic_objects': 0,
            'hog_detections': [],
            'num_hog_triggers': 0,
            'image_shape': image.shape[:2],
            'raw_results': None,
        }

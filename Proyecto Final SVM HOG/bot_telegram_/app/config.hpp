/**
 * Configuración del Detector de Personas C++
 * Proyecto Integrador - Visión Artificial
 */

#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace Config {

// ============= SERVIDOR PYTHON =============
const std::string PYTHON_SERVER_URL = "http://localhost:8000/detect";

// ============= MODELO SVM (HOG) =============
const std::string SVM_MODEL_PATH = "../models/svm_weights1.txt";

// Parámetros HOG (Deben coincidir con entrenamiento)
const cv::Size WIN_SIZE(64, 128);
const cv::Size BLOCK_SIZE(16, 16);
const cv::Size BLOCK_STRIDE(8, 8);
const cv::Size CELL_SIZE(8, 8);
const int NBINS = 9;

// Parámetros de Detección
// Parámetros de Detección
const double HIT_THRESHOLD = 0.3; // Según script usuario
const cv::Size WIN_STRIDE(8, 8);
const cv::Size PADDING(8, 8);
const double SCALE = 1.05;
const double SCORE_THRESHOLD =
    1.2; // Filtro final (Alineado con HIGH_CONFIDENCE_THRESHOLD)
const double NMS_THRESHOLD = 0.4;  // Solapamiento permitido (usuario usa 0.4)
const bool SEND_TO_SERVER = true; // Enviar a servidor Python

// ============= CÁMARA =============
const int CAMERA_INDEX = 0;
const int CAMERA_WIDTH = 1280;
const int CAMERA_HEIGHT = 720;
const int CAMERA_FPS = 30;

// ============= DETECCIÓN =============
const int DETECTION_COOLDOWN_MS = 20000; // Esperar 7s entre detecciones
const int MIN_PERSON_AREA = 15000;      // Área mínima para considerar detección
const bool SHOW_PREVIEW = true;         // Mostrar preview en tiempo real

// ============= VIDEO RECORDING =============
const bool RECORD_VIDEO = true;
const int VIDEO_DURATION_SECONDS = 5;
const std::string VIDEO_CODEC = "avc1"; // H.264
const std::string TEMP_VIDEO_PATH = "../../outputs/temp_video.mp4";

// ============= RUTAS =============
const std::string OUTPUTS_DIR = "../../outputs/";
const std::string DETECTIONS_DIR = "../../outputs/detections/";
const std::string LOGS_DIR = "../../outputs/logs/";

// ============= VISUALIZACIÓN =============
// ============= VISUALIZACIÓN =============
const cv::Scalar BBOX_COLOR = cv::Scalar(0, 255, 0); // Verde (Alta confianza)
const cv::Scalar BBOX_COLOR_LOW =
    cv::Scalar(0, 165, 255);                 // Naranja (Media confianza)
const float HIGH_CONFIDENCE_THRESHOLD = 1.2; // Umbral para color verde
const int BBOX_THICKNESS = 2;
const float FONT_SCALE = 0.7f;
const cv::Scalar TEXT_COLOR = cv::Scalar(255, 255, 255); // Blanco
const cv::Scalar TEXT_BG_COLOR = cv::Scalar(0, 0, 0);    // Negro

// ============= ESTADÍSTICAS =============
struct Stats {
  int total_detections = 0;
  int persons_detected = 0;
  int images_sent = 0;
  int videos_sent = 0;
  double avg_fps = 0.0;
  double avg_inference_time_ms = 0.0;
};

// ============= CLASES COCO =============
const std::vector<std::string> COCO_CLASSES = {
    "person",        "bicycle",      "car",
    "motorcycle",    "airplane",     "bus",
    "train",         "truck",        "boat",
    "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench",        "bird",
    "cat",           "dog",          "horse",
    "sheep",         "cow",          "elephant",
    "bear",          "zebra",        "giraffe",
    "backpack",      "umbrella",     "handbag",
    "tie",           "suitcase",     "frisbee",
    "skis",          "snowboard",    "sports ball",
    "kite",          "baseball bat", "baseball glove",
    "skateboard",    "surfboard",    "tennis racket",
    "bottle",        "wine glass",   "cup",
    "fork",          "knife",        "spoon",
    "bowl",          "banana",       "apple",
    "sandwich",      "orange",       "broccoli",
    "carrot",        "hot dog",      "pizza",
    "donut",         "cake",         "chair",
    "couch",         "potted plant", "bed",
    "dining table",  "toilet",       "tv",
    "laptop",        "mouse",        "remote",
    "keyboard",      "cell phone",   "microwave",
    "oven",          "toaster",      "sink",
    "refrigerator",  "book",         "clock",
    "vase",          "scissors",     "teddy bear",
    "hair drier",    "toothbrush"};

} // namespace Config

#endif // CONFIG_HPP

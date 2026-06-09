#include "HandTracker.hpp"
#include "ARFilter.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

// ─── Config ──────────────────────────────────────────────────────────────────
// Model paths (run  cd "Proyecto final V2" && bash models/download_models.sh)
static constexpr char PALM_MODEL[] = "models/palm_detection.onnx";
static constexpr char LM_MODEL[]   = "models/hand_landmark.onnx";

// Camera: 0 = webcam, or pass a stream URL as first argument
static constexpr int DEFAULT_CAM_IDX = 0;

int main(int argc, char** argv) {
    // ── Load models ──────────────────────────────────────────────────────────
    HandTracker tracker;
    if (!tracker.load(PALM_MODEL, LM_MODEL)) {
        std::cerr << "\n[Error] No se pudieron cargar los modelos ONNX.\n"
                  << "Ejecuta primero:\n"
                  << "  bash models/download_models.sh\n\n";
        return 1;
    }

    // ── Open camera ──────────────────────────────────────────────────────────
    cv::VideoCapture cap;
    if (argc > 1) {
        cap.open(argv[1]);           // stream URL passed as argument
        std::cout << "[AR] Usando stream: " << argv[1] << "\n";
    } else {
        cap.open(DEFAULT_CAM_IDX);
        cap.set(cv::CAP_PROP_FRAME_WIDTH,  1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT,  720);
        std::cout << "[AR] Usando webcam " << DEFAULT_CAM_IDX << "\n";
    }

    if (!cap.isOpened()) {
        std::cerr << "[Error] No se pudo abrir la cámara.\n";
        return 1;
    }

    std::cout << "\nTeclas: ESC = salir\n\n";

    // ── State ─────────────────────────────────────────────────────────────────
    float angleDeg = 0.0f;

    // zCm = 0 → derivado del tamaño de mano en píxeles (sin estéreo)
    // Puedes cambiar este valor desde el exterior si integras el estéreo.
    float zCm = 0.0f;

    cv::namedWindow("Filtro AR - Manos", cv::WINDOW_NORMAL);
    cv::resizeWindow("Filtro AR - Manos", 1280, 720);

    // ── FPS ───────────────────────────────────────────────────────────────────
    double prev = 0;

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "[AR] Frame vacío, reintentando...\n";
            continue;
        }

        cv::flip(frame, frame, 1);  // mirror like Python version

        // Detect hands
        auto hands = tracker.update(frame, 2);

        // Render effects (zCm=0 → usa tamaño de mano en píxeles)
        angleDeg = ARFilter::render(frame, hands, angleDeg, zCm);

        // FPS overlay
        double now = cv::getTickCount() / cv::getTickFrequency();
        double fps = (prev > 0) ? 1.0 / (now - prev) : 0;
        prev = now;

        char buf[32];
        std::snprintf(buf, sizeof(buf), "FPS: %.0f", fps);
        cv::putText(frame, buf, {20, 40},
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Filtro AR - Manos", frame);

        if ((cv::waitKey(1) & 0xFF) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}

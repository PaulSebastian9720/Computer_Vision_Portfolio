#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/xobjdetect.hpp>
#include <algorithm>
#include <deque>
#include <iostream>
#include <vector>

static const char* kWindow = "Detector Camion Grande - Video (Dual Frontal+Lateral)";

static bool g_ignoreTrackbar = false;
static bool g_seekRequested  = false;
static int  g_seekTarget     = 0;

static void onTrackbar(int pos, void*) {
    if (g_ignoreTrackbar) return;
    g_seekTarget    = pos;
    g_seekRequested = true;
}

static double rectIoU(const cv::Rect& a, const cv::Rect& b) {
    const int inter = (a & b).area();
    const int uni   = a.area() + b.area() - inter;
    return uni > 0 ? static_cast<double>(inter) / uni : 0.0;
}

struct ViewDetector {
    cv::HOGDescriptor hog;
    std::string name;
    float threshold;
    cv::Size stride;
    cv::Size padding;
};

static bool loadDetector(const std::string& path, const std::string& name, float overrideThreshold,
                          std::vector<ViewDetector>& out) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Error: No se pudo abrir " << path << " para la vista '" << name << "'\n";
        return false;
    }
    int winWidth = 0, winHeight = 0;
    cv::Mat detectorMat;
    fs["win_width"] >> winWidth;
    fs["win_height"] >> winHeight;
    fs["svm_detector"] >> detectorMat;
    float fileThreshold = 0.0f;
    fs["threshold"] >> fileThreshold;
    fs.release();

    std::vector<float> svmDetector;
    detectorMat.reshape(1, 1).copyTo(svmDetector);

    ViewDetector vd;
    vd.name = name;
    vd.threshold = (overrideThreshold > -1000.0f) ? overrideThreshold : fileThreshold;
    vd.hog = cv::HOGDescriptor(cv::Size(winWidth, winHeight), cv::Size(16, 16),
                                cv::Size(8, 8), cv::Size(8, 8), 9);
    vd.hog.setSVMDetector(svmDetector);
    
    // Configurar stride y padding optimos para velocidad de procesamiento
    if (name == "frontal") {
        vd.stride = cv::Size(16, 16);  // Igual al stride de minado en entrenamiento
        vd.padding = cv::Size(8, 8);   // Reducir de (32,32) para evitar escaneos innecesarios
    } else {
        vd.stride = cv::Size(16, 16);  // Balance de velocidad y recall lateral
        vd.padding = cv::Size(8, 8);   // Reducir de (32,32)
    }
    
    out.push_back(std::move(vd));
    std::cout << "Vista '" << name << "' cargada. Stride=(" << vd.stride.width << "," << vd.stride.height 
              << ") | Padding=(" << vd.padding.width << "," << vd.padding.height << ") | Umbral=" << vd.threshold << "\n";
    return true;
}

int main(int argc, char** argv) {
    const std::string videoPath    = (argc > 1) ? argv[1] : "../targets/camiones_nuevovision/videos/CamionesColombia.mkv";
    const float frontalThreshold   = (argc > 2) ? std::stof(argv[2]) : -1001.0f;
    const float lateralThreshold   = (argc > 3) ? std::stof(argv[3]) : -1001.0f;
    const std::string frontalPath  = (argc > 4) ? argv[4] : "../entrenamiento camión grande/dataset/detector_frontal.yml";
    const std::string lateralPath  = (argc > 5) ? argv[5] : "../entrenamiento camión grande/dataset/detector_lateral.yml";

    std::cout << "Iniciando detector dual con video: " << videoPath << "\n";

    std::vector<ViewDetector> detectors;
    loadDetector(frontalPath, "frontal", frontalThreshold, detectors);
    loadDetector(lateralPath, "lateral", lateralThreshold, detectors);

    if (detectors.empty()) {
        std::cerr << "Error: No se cargo ningun detector.\n";
        return 1;
    }

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Error: No se pudo abrir el video: " << videoPath << "\n";
        return 1;
    }

    const int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double videoFps = cap.get(cv::CAP_PROP_FPS);
    if (videoFps <= 0) videoFps = 30.0;
    const int frameTimeMs = static_cast<int>(1000.0 / videoFps);

    cv::namedWindow(kWindow);
    if (totalFrames > 0)
        cv::createTrackbar("posicion", kWindow, nullptr, totalFrames - 1, onTrackbar);

    bool paused = false;
    cv::Mat frame, gray, display;
    
    // Historial para validacion temporal (evitar falsos positivos esporadicos)
    const int    HISTORY_LEN = 6;
    const int    MIN_HITS    = 3;
    const double IOU_MIN     = 0.3;
    std::deque<std::vector<cv::Rect>> history;

    double processingFps = 0.0;

    while (true) {
        if (g_seekRequested) {
            cap.set(cv::CAP_PROP_POS_FRAMES, g_seekTarget);
            g_seekRequested = false;
            display.release();
            history.clear();
        }

        const int64 t0 = cv::getTickCount();

        if (!paused || display.empty()) {
            cap >> frame;
            if (frame.empty()) break; // Fin del video

            // Redimensionar para reproducir fluido
            if (frame.cols > 640) {
                double s = 640.0 / frame.cols;
                cv::resize(frame, frame, cv::Size(), s, s);
            }

            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

            std::vector<cv::Rect> boxes;
            std::vector<double> weights;

            for (const auto& vd : detectors) {
                std::vector<cv::Rect> b;
                std::vector<double> w;
                // Deslizamiento de escala HOG con stride y padding optimizados
                vd.hog.detectMultiScale(gray, b, w, vd.threshold, vd.stride,
                                         vd.padding, 1.1, 4.0, false);
                boxes.insert(boxes.end(), b.begin(), b.end());
                weights.insert(weights.end(), w.begin(), w.end());
            }

            // Dibujar cajas y aplicar confirmacion temporal
            for (size_t i = 0; i < boxes.size(); ++i) {
                int hits = 0;
                for (const auto& past : history) {
                    for (const auto& pb : past) {
                        if (rectIoU(boxes[i], pb) > IOU_MIN) { ++hits; break; }
                    }
                }
                if (hits >= MIN_HITS) {
                    // Caja confirmada: verde grueso
                    cv::rectangle(frame, boxes[i], cv::Scalar(0, 255, 0), 2);
                    char label[64];
                    std::snprintf(label, sizeof(label), "camion %.2f", weights[i]);
                    cv::putText(frame, label, boxes[i].tl() + cv::Point(0, -6),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
                } else {
                    // Caja candidata (sin confirmar): amarillo delgado
                    cv::rectangle(frame, boxes[i], cv::Scalar(0, 255, 255), 1);
                }
            }

            history.push_back(boxes);
            if (static_cast<int>(history.size()) > HISTORY_LEN) history.pop_front();

            const int curFrame = static_cast<int>(cap.get(cv::CAP_PROP_POS_FRAMES));
            if (totalFrames > 0) {
                g_ignoreTrackbar = true;
                cv::setTrackbarPos("posicion", kWindow, curFrame);
                g_ignoreTrackbar = false;
            }

            display = frame.clone();
        }

        const int64 t1 = cv::getTickCount();
        const double procTimeMs = (t1 - t0) * 1000.0 / cv::getTickFrequency();
        processingFps = cv::getTickFrequency() / (t1 - t0);

        // Overlay con informacion de estado en tiempo real
        cv::Mat hudView = display.clone();
        char info[128];
        const int curFrame = static_cast<int>(cap.get(cv::CAP_PROP_POS_FRAMES));
        std::snprintf(info, sizeof(info), "HUD FPS: %.1f | %.1fs / %.1fs | U: F=%.2f L=%.2f",
                      processingFps, curFrame / videoFps, totalFrames / videoFps,
                      detectors[0].threshold, detectors[1].threshold);
        cv::putText(hudView, info, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(0, 0, 255), 2);

        if (paused) {
            cv::putText(hudView, "PAUSA", cv::Point(10, 55), cv::FONT_HERSHEY_SIMPLEX,
                        0.8, cv::Scalar(0, 255, 255), 2);
        }

        cv::imshow(kWindow, hudView);

        // Ajustar waitKey para reproduccion en tiempo real y no acelerada
        int waitTime = paused ? 30 : std::max(1, frameTimeMs - static_cast<int>(procTimeMs));
        const int key = cv::waitKey(waitTime);

        if (key == 'q') break;
        if (key == ' ') paused = !paused;
        if (key == 'a' || key == 'd') {
            const double cur   = cap.get(cv::CAP_PROP_POS_FRAMES);
            const double delta = 5.0 * videoFps * (key == 'd' ? 1.0 : -1.0);
            const double maxF  = (totalFrames > 0) ? totalFrames - 1.0 : cur + delta;
            cap.set(cv::CAP_PROP_POS_FRAMES, std::clamp(cur + delta, 0.0, maxF));
            display.release();
            history.clear();
        }
        if (key == 'r') {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            display.release();
            history.clear();
        }
        if (key == '.' && paused) display.release(); 
    }

    return 0;
}

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <deque>
#include <iostream>
#include <vector>

// Reproductor con detección HOG+SVM (frontal + lateral) sobre un video precargado.
// Uso: ./truck_trigger_video [video] [umbral] [detector_frontal.yml] [detector_lateral.yml]
// Formatos: cualquiera soportado por OpenCV/FFmpeg (mp4, webm, avi, mkv, mov, ...).
//
// Controles (con la ventana en foco):
//   espacio  pausa / reanuda
//   d        adelantar 5 s        a  retroceder 5 s
//   .        avanzar 1 frame (en pausa)
//   r        reiniciar desde el inicio
//   q        salir
//   barra    saltar a cualquier punto del video

static const char* kWindow = "Detector Camion - Video (HOG+SVM, frontal+lateral)";

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
    float threshold;  // propio de cada vista: frontal y lateral no calibran igual
};

// overrideThreshold >= -1000 fuerza el mismo umbral para todas las vistas (CLI); si no,
// cada detector usa el que trae su propio .yml (calculado en el entrenamiento).
static bool loadDetector(const std::string& path, const std::string& name, float overrideThreshold,
                          std::vector<ViewDetector>& out) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "No se pudo abrir " << path << " — se omite la vista '" << name << "'\n";
        return false;
    }
    int winWidth = 0, winHeight = 0;
    cv::Mat detectorMat;
    fs["win_width"] >> winWidth;
    fs["win_height"] >> winHeight;
    fs["svm_detector"] >> detectorMat;
    float fileThreshold = 0.0f;
    fs["threshold"] >> fileThreshold;  // 0.0 si el yml no lo trae (compatibilidad con detectores viejos)
    fs.release();

    std::vector<float> svmDetector;
    detectorMat.reshape(1, 1).copyTo(svmDetector);

    ViewDetector vd;
    vd.name = name;
    vd.threshold = (overrideThreshold > -1000.0f) ? overrideThreshold : fileThreshold;
    vd.hog = cv::HOGDescriptor(cv::Size(winWidth, winHeight), cv::Size(16, 16),
                                cv::Size(8, 8), cv::Size(8, 8), 9);
    vd.hog.setSVMDetector(svmDetector);
    out.push_back(std::move(vd));
    std::cout << "vista '" << name << "' cargada, umbral=" << vd.threshold << "\n";
    return true;
}

int main(int argc, char** argv) {
    const std::string videoPath    = (argc > 1) ? argv[1] : "/home/paul/universidad/septimo_v2/Vison_Computador/Practicas/Proyecto Final SVM HOG/artifcats/videos/CamionesFrontal.mp4";
    const float thresholdOverride  = (argc > 2) ? std::stof(argv[2]) : -1001.0f;
    const std::string lateralPath  = (argc > 3) ? argv[3] : "/home/paul/universidad/septimo_v2/Vison_Computador/Practicas/Proyecto Final SVM HOG/entrenamiento camión grande/dataset/detector_frontal.yml";

    std::vector<ViewDetector> detectors;
    loadDetector(lateralPath, "lateral", thresholdOverride, detectors);
    if (detectors.empty()) {
        std::cerr << "No se cargó ningún detector\n";
        return 1;
    }

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "No se pudo abrir el video: " << videoPath << "\n";
        return 1;
    }
    const int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double videoFps = cap.get(cv::CAP_PROP_FPS);
    if (videoFps <= 0) videoFps = 30.0;

    cv::namedWindow(kWindow);
    if (totalFrames > 0)
        cv::createTrackbar("posicion", kWindow, nullptr, totalFrames - 1, onTrackbar);

    bool paused = false;
    cv::Mat frame, gray, display;
    int64 lastTick = cv::getTickCount();

    // Confirmación temporal: una caja se confirma solo si persiste en la misma zona en
    // varios frames del video (un cartel/señal dispara frames sueltos; un camión persiste).
    const int    HISTORY_LEN = 6;
    const int    MIN_HITS    = 3;
    const double IOU_MIN     = 0.3;
    std::deque<std::vector<cv::Rect>> history;

    while (true) {
        if (g_seekRequested) {
            cap.set(cv::CAP_PROP_POS_FRAMES, g_seekTarget);
            g_seekRequested = false;
            display.release();  // fuerza a procesar el frame nuevo aunque esté en pausa
            history.clear();    // el historial de otro tramo del video ya no aplica
        }

        if (!paused || display.empty()) {
            cap >> frame;
            if (frame.empty()) break;  // fin del video

            // normalizar a 640 de ancho para que la inferencia sea fluida
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
                vd.hog.detectMultiScale(gray, b, w, vd.threshold, cv::Size(8, 8),
                                         cv::Size(32, 32), 1.1, 4.0, false);
                boxes.insert(boxes.end(), b.begin(), b.end());
                weights.insert(weights.end(), w.begin(), w.end());
            }

            for (size_t i = 0; i < boxes.size(); ++i) {
                int hits = 0;
                for (const auto& past : history) {
                    for (const auto& pb : past) {
                        if (rectIoU(boxes[i], pb) > IOU_MIN) { ++hits; break; }
                    }
                }
                if (hits >= MIN_HITS) {
                    cv::rectangle(frame, boxes[i], cv::Scalar(0, 255, 0), 2);
                    char label[32];
                    std::snprintf(label, sizeof(label), "camion %.2f", weights[i]);
                    cv::putText(frame, label, boxes[i].tl() + cv::Point(0, -6),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
                } else {
                    // candidata aún sin confirmar: trazo fino amarillo
                    cv::rectangle(frame, boxes[i], cv::Scalar(0, 255, 255), 1);
                }
            }
            history.push_back(boxes);
            if (static_cast<int>(history.size()) > HISTORY_LEN) history.pop_front();

            int64 now = cv::getTickCount();
            double fps = cv::getTickFrequency() / static_cast<double>(now - lastTick);
            lastTick = now;

            const int curFrame = static_cast<int>(cap.get(cv::CAP_PROP_POS_FRAMES));
            char info[96];
            std::snprintf(info, sizeof(info), "FPS: %.1f  |  %.1fs / %.1fs  [esp:pausa a/d:+-5s q:salir]",
                          fps, curFrame / videoFps, totalFrames / videoFps);
            cv::putText(frame, info, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX,
                        0.5, cv::Scalar(0, 0, 255), 2);

            display = frame.clone();

            if (totalFrames > 0) {
                g_ignoreTrackbar = true;
                cv::setTrackbarPos("posicion", kWindow, curFrame);
                g_ignoreTrackbar = false;
            }
        }

        if (paused) {
            cv::Mat pausedView = display.clone();
            cv::putText(pausedView, "PAUSA", cv::Point(10, 55), cv::FONT_HERSHEY_SIMPLEX,
                        0.8, cv::Scalar(0, 255, 255), 2);
            cv::imshow(kWindow, pausedView);
        } else {
            cv::imshow(kWindow, display);
        }

        const int key = cv::waitKey(paused ? 30 : 1);
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
        if (key == '.' && paused) display.release();  // avanza un solo frame
    }

    return 0;
}

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/xobjdetect.hpp>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/resource.h>
#include <vector>

// Reproductor con detección HOG+SVM (frontal + lateral) sobre video o cámara.
// Uso: ./truck_trigger_video [video|camera:0|0] [umbral] [detector.yml] ...
// Formatos: cualquiera soportado por OpenCV/FFmpeg (mp4, webm, avi, mkv, mov, ...)
// o cámara local usando camera:0, camera:1, etc.
//
// Controles (con la ventana en foco):
//   espacio  pausa / reanuda
//   d        adelantar 5 s        a  retroceder 5 s
//   .        avanzar 1 frame (en pausa)
//   r        reiniciar desde el inicio
//   q        salir
//   barra    saltar a cualquier punto del video (solo archivos)

static const char* kWindow = "Detector Camion - Video (HOG+SVM, frontal+lateral)";
namespace fs = std::filesystem;

static bool g_ignoreTrackbar = false;
static bool g_seekRequested  = false;
static int  g_seekTarget     = 0;

static double currentMemoryMb() {
    rusage usage{};
    getrusage(RUSAGE_SELF, &usage);
#ifdef __APPLE__
    return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);
#else
    return static_cast<double>(usage.ru_maxrss) / 1024.0;
#endif
}

static fs::path appLogsDir() {
    fs::path logsDir = fs::absolute("../logs");
    fs::create_directories(logsDir);
    return logsDir;
}

static void appendLog(const std::string& fileName, const std::string& message) {
    std::ofstream log(appLogsDir() / fileName, std::ios::app);
    log << message << "\n";
}

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

static bool shouldSendAlerts(const std::string& serverUrl) {
    return !serverUrl.empty() && serverUrl != "none" && serverUrl != "off";
}

static bool parseCameraSource(const std::string& source, int& cameraIndex) {
    std::string value = source;
    const std::string cameraPrefix = "camera:";
    const std::string camPrefix = "cam:";

    if (value.rfind(cameraPrefix, 0) == 0) {
        value = value.substr(cameraPrefix.size());
    } else if (value.rfind(camPrefix, 0) == 0) {
        value = value.substr(camPrefix.size());
    } else {
        const bool numericOnly = !value.empty() &&
            std::all_of(value.begin(), value.end(), [](unsigned char c) {
                return std::isdigit(c) != 0;
            });
        if (!numericOnly) return false;
    }

    if (value.empty()) value = "0";
    cameraIndex = std::stoi(value);
    return true;
}

static fs::path writeAlertClip(const std::deque<cv::Mat>& recentFrames,
                               double fps,
                               const fs::path& outDir,
                               int64 tick) {
    const int targetFrames = std::max(1, static_cast<int>(fps * 5.0 + 0.5));
    fs::path clipPath = outDir / ("hog_trigger_clip_" + std::to_string(tick) + ".mp4");

    if (recentFrames.empty()) return clipPath;

    const cv::Size frameSize(recentFrames.front().cols, recentFrames.front().rows);
    cv::VideoWriter writer(
        clipPath.string(),
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        fps,
        frameSize
    );

    if (!writer.isOpened()) return clipPath;

    const int missingFrames = targetFrames - static_cast<int>(recentFrames.size());
    for (int i = 0; i < missingFrames; ++i) {
        writer.write(recentFrames.front());
    }
    for (const auto& bufferedFrame : recentFrames) {
        writer.write(bufferedFrame);
    }
    writer.release();

    return clipPath;
}

static void sendAlertToServer(const cv::Mat& cleanFrame,
                              const cv::Mat& hogFrame,
                              const std::deque<cv::Mat>& recentFrames,
                              double fps,
                              double hitThreshold,
                              int hogCount,
                              double minCenterY,
                              double maxCenterY,
                              double minAreaFraction,
                              double maxAreaFraction,
                              double roiMinX,
                              double roiMaxX,
                              const std::string& serverUrl) {
    fs::path outDir = fs::absolute("../outputs/detections");
    fs::path logsDir = appLogsDir();
    fs::create_directories(outDir);

    const int64 tick = cv::getTickCount();
    fs::path imagePath = outDir / ("hog_trigger_" + std::to_string(tick) + ".jpg");
    fs::path hogImagePath = outDir / ("hog_trigger_annotated_" + std::to_string(tick) + ".jpg");
    fs::path clipPath = writeAlertClip(recentFrames, fps, outDir, tick);
    cv::imwrite(imagePath.string(), cleanFrame);
    cv::imwrite(hogImagePath.string(), hogFrame);

    std::string command = "curl -s -X POST \"" + serverUrl + "\" -F \"file=@" +
                          imagePath.string() + "\"";
    if (fs::exists(hogImagePath)) {
        command += " -F \"hog_file=@" + hogImagePath.string() + "\"";
    }
    if (fs::exists(clipPath)) {
        command += " -F \"clip=@" + clipPath.string() + "\"";
    }
    command += " -F \"hog_threshold=" + std::to_string(hitThreshold) + "\"";
    command += " -F \"hog_count=" + std::to_string(hogCount) + "\"";
    command += " -F \"min_center_y=" + std::to_string(minCenterY) + "\"";
    command += " -F \"max_center_y=" + std::to_string(maxCenterY) + "\"";
    command += " -F \"min_area=" + std::to_string(minAreaFraction) + "\"";
    command += " -F \"max_area=" + std::to_string(maxAreaFraction) + "\"";
    command += " -F \"roi_min_x=" + std::to_string(roiMinX) + "\"";
    command += " -F \"roi_max_x=" + std::to_string(roiMaxX) + "\"";
    command += " -w \"http_code=%{http_code} total_time=%{time_total}\\n\"";
    command += " -o /dev/null >> \"" + (logsDir / "api_streams.log").string() + "\" 2>&1 &";
    std::system(command.c_str());

    std::ostringstream log;
    log << "[API STREAM] multipart/form-data"
        << " server=" << serverUrl
        << " file=" << imagePath
        << " hog_file=" << hogImagePath
        << " clip=" << clipPath
        << " hog_count=" << hogCount
        << " threshold=" << hitThreshold;
    appendLog("cpp_metrics.log", log.str());

    std::cout << "Alerta enviada al servidor por API multipart: " << imagePath;
    if (fs::exists(clipPath)) std::cout << " + " << clipPath;
    std::cout << " | log=" << (logsDir / "api_streams.log") << "\n";
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
    const std::string videoPath    = (argc > 1) ? argv[1] : "../targets/camiones_nuevovision/videos/CamionesFrontal.mp4";
    const std::string thresholdArg = (argc > 2) ? argv[2] : "auto";
    const float thresholdOverride  = (thresholdArg == "auto") ? -1001.0f : std::stof(thresholdArg);
    const std::string lateralPath  = (argc > 3) ? argv[3] : "../targets/camiones_nuevovision/models/detector_lateral_latest.yml";
    const double minCenterY        = (argc > 4) ? std::stod(argv[4]) : 0.0;
    const double maxAreaFraction   = (argc > 5) ? std::stod(argv[5]) : 1.0;
    const double roiMinX           = (argc > 7) ? std::stod(argv[7]) : 0.0;
    const double roiMaxX           = (argc > 8) ? std::stod(argv[8]) : 1.0;
    const double minAreaFraction   = (argc > 10) ? std::stod(argv[10]) : 0.0;
    const double maxCenterY        = (argc > 12) ? std::stod(argv[12]) : 1.0;
    const int historyLen           = (argc > 13) ? std::stoi(argv[13]) : 6;
    const int confirmHits          = (argc > 14) ? std::stoi(argv[14]) : 3;
    const double iouMin            = (argc > 15) ? std::stod(argv[15]) : 0.3;
    const std::string serverUrl    = (argc > 16) ? argv[16] : "none";
    const double alertCooldownSec  = (argc > 17) ? std::stod(argv[17]) : 1.0;
    const std::string frontalPath  = (argc > 18) ? argv[18] : "";
    const std::string frontalThresholdArg = (argc > 19) ? argv[19] : "auto";
    const float frontalThresholdOverride =
        (frontalThresholdArg == "auto") ? -1001.0f : std::stof(frontalThresholdArg);

    std::cout << "Configuracion alerta: threshold=" << thresholdOverride
              << " cooldown=" << alertCooldownSec << "s"
              << " server=" << serverUrl << "\n";

    std::vector<ViewDetector> detectors;
    loadDetector(lateralPath, "lateral", thresholdOverride, detectors);
    if (!frontalPath.empty() && frontalPath != "none" && frontalPath != "off") {
        loadDetector(frontalPath, "frontal", frontalThresholdOverride, detectors);
    }
    if (detectors.empty()) {
        std::cerr << "No se cargó ningún detector\n";
        return 1;
    }

    int cameraIndex = 0;
    const bool useCamera = parseCameraSource(videoPath, cameraIndex);
    cv::VideoCapture cap;
    if (useCamera) {
        cap.open(cameraIndex);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        cap.set(cv::CAP_PROP_FPS, 30);
    } else {
        cap.open(videoPath);
    }
    if (!cap.isOpened()) {
        if (useCamera) {
            std::cerr << "No se pudo abrir la camara con indice: " << cameraIndex << "\n";
        } else {
            std::cerr << "No se pudo abrir el video: " << videoPath << "\n";
        }
        return 1;
    }
    const int totalFrames = useCamera ? 0 : static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double videoFps = cap.get(cv::CAP_PROP_FPS);
    if (videoFps <= 0) videoFps = 30.0;
    std::cout << "Fuente de entrada: "
              << (useCamera ? ("camara " + std::to_string(cameraIndex)) : videoPath)
              << " | fps=" << videoFps << "\n";

    cv::namedWindow(kWindow);
    if (!useCamera && totalFrames > 0)
        cv::createTrackbar("posicion", kWindow, nullptr, totalFrames - 1, onTrackbar);

    bool paused = false;
    cv::Mat frame, gray, display;
    int64 lastTick = cv::getTickCount();

    // Confirmación temporal: una caja se confirma solo si persiste en la misma zona en
    // varios frames del video (un cartel/señal dispara frames sueltos; un camión persiste).
    std::deque<std::vector<cv::Rect>> history;
    std::deque<cv::Mat> recentCleanFrames;
    const int alertClipFrames = std::max(1, static_cast<int>(videoFps * 5.0 + 0.5));
    int64 lastAlertTick = 0;
    int64 lastMetricsTick = 0;

    while (true) {
        if (g_seekRequested) {
            if (!useCamera) cap.set(cv::CAP_PROP_POS_FRAMES, g_seekTarget);
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
            cv::Mat cleanFrame = frame.clone();
            recentCleanFrames.push_back(cleanFrame);
            while (static_cast<int>(recentCleanFrames.size()) > alertClipFrames) {
                recentCleanFrames.pop_front();
            }

            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

            std::vector<cv::Rect> boxes;
            std::vector<double> weights;
            std::vector<std::string> views;
            for (const auto& vd : detectors) {
                std::vector<cv::Rect> b;
                std::vector<double> w;
                vd.hog.detectMultiScale(gray, b, w, vd.threshold, cv::Size(8, 8),
                                         cv::Size(32, 32), 1.1, 4.0, false);
                for (size_t j = 0; j < b.size(); ++j) {
                    boxes.push_back(b[j]);
                    weights.push_back(w[j]);
                    views.push_back(vd.name);
                }
            }

            int confirmedBoxes = 0;
            double bestConfidence = 0.0;
            for (size_t i = 0; i < boxes.size(); ++i) {
                const double frameArea = static_cast<double>(frame.cols) * frame.rows;
                const double boxArea = static_cast<double>(boxes[i].width) * boxes[i].height;
                const double areaFraction = boxArea / frameArea;
                const double centerX = (boxes[i].x + boxes[i].width * 0.5) / frame.cols;
                const double centerY = (boxes[i].y + boxes[i].height * 0.5) / frame.rows;
                if (centerX < roiMinX || centerX > roiMaxX ||
                    centerY < minCenterY || centerY > maxCenterY ||
                    areaFraction < minAreaFraction || areaFraction > maxAreaFraction) {
                    continue;
                }

                int hits = 0;
                for (const auto& past : history) {
                    for (const auto& pb : past) {
                        if (rectIoU(boxes[i], pb) > iouMin) { ++hits; break; }
                    }
                }
                if (confirmHits <= 0 || hits >= confirmHits) {
                    ++confirmedBoxes;
                    bestConfidence = std::max(bestConfidence, weights[i]);
                    cv::rectangle(frame, boxes[i], cv::Scalar(0, 255, 0), 2);
                    char label[64];
                    std::snprintf(label, sizeof(label), "camion %.2f", weights[i]);
                    cv::putText(frame, label, boxes[i].tl() + cv::Point(0, -6),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
                } else {
                    // candidata aún sin confirmar: trazo fino amarillo
                    cv::rectangle(frame, boxes[i], cv::Scalar(0, 255, 255), 1);
                }
            }
            history.push_back(boxes);
            if (historyLen > 0 && static_cast<int>(history.size()) > historyLen) history.pop_front();
            if (historyLen <= 0) history.clear();

            const int64 nowForAlert = cv::getTickCount();
            const double secondsSinceAlert =
                lastAlertTick == 0
                    ? alertCooldownSec
                    : (nowForAlert - lastAlertTick) / cv::getTickFrequency();
            if (confirmedBoxes > 0 && shouldSendAlerts(serverUrl) && secondsSinceAlert >= alertCooldownSec) {
                sendAlertToServer(cleanFrame, frame, recentCleanFrames, videoFps,
                                  thresholdOverride, confirmedBoxes,
                                  minCenterY, maxCenterY, minAreaFraction, maxAreaFraction,
                                  roiMinX, roiMaxX, serverUrl);
                lastAlertTick = nowForAlert;
            }

            int64 now = cv::getTickCount();
            double fps = cv::getTickFrequency() / static_cast<double>(now - lastTick);
            lastTick = now;
            const double ramMb = currentMemoryMb();

            const int curFrame = static_cast<int>(cap.get(cv::CAP_PROP_POS_FRAMES));
            char info[96];
            if (useCamera) {
                std::snprintf(info, sizeof(info), "FPS: %.1f  |  CAMARA %d  [esp:pausa q:salir]",
                              fps, cameraIndex);
            } else {
                std::snprintf(info, sizeof(info), "FPS: %.1f  |  %.1fs / %.1fs  [esp:pausa a/d:+-5s q:salir]",
                              fps, curFrame / videoFps, totalFrames / videoFps);
            }
            cv::putText(frame, info, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX,
                        0.5, cv::Scalar(0, 0, 255), 2);
            char metrics[128];
            std::snprintf(metrics, sizeof(metrics), "RAM: %.1f MB | HOG conf: %.2f | candidatos: %d",
                          ramMb, bestConfidence, confirmedBoxes);
            cv::putText(frame, metrics, cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX,
                        0.5, cv::Scalar(0, 0, 255), 2);

            if (lastMetricsTick == 0 ||
                (now - lastMetricsTick) / cv::getTickFrequency() >= 1.0) {
                std::ostringstream metricLog;
                metricLog << std::fixed << std::setprecision(2)
                          << "[C++ METRICS] frame=" << curFrame
                          << " fps=" << fps
                          << " ram_mb=" << ramMb
                          << " hog_candidates=" << confirmedBoxes
                          << " best_hog_confidence=" << bestConfidence
                          << " api_streams=" << (shouldSendAlerts(serverUrl) ? "enabled" : "disabled");
                std::cout << metricLog.str() << "\n";
                appendLog("cpp_metrics.log", metricLog.str());
                lastMetricsTick = now;
            }

            display = frame.clone();

            if (!useCamera && totalFrames > 0) {
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
        if (!useCamera && (key == 'a' || key == 'd')) {
            const double cur   = cap.get(cv::CAP_PROP_POS_FRAMES);
            const double delta = 5.0 * videoFps * (key == 'd' ? 1.0 : -1.0);
            const double maxF  = (totalFrames > 0) ? totalFrames - 1.0 : cur + delta;
            cap.set(cv::CAP_PROP_POS_FRAMES, std::clamp(cur + delta, 0.0, maxF));
            display.release();
            history.clear();
        }
        if (!useCamera && key == 'r') {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            display.release();
            history.clear();
        }
        if (key == '.' && paused) display.release();  // avanza un solo frame
    }

    return 0;
}

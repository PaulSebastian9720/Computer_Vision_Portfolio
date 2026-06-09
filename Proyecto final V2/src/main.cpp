#include "CameraStream.hpp"
#include "StereoProcessor.hpp"
#include "DepthEstimator.hpp"
#include "ObjectDetector.hpp"
#include "HandTracker.hpp"
#include "ARFilter.hpp"

#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <string>
#include <atomic>
#include <memory>

// ─── Config ──────────────────────────────────────────────────────────────────
static constexpr char CAM_LEFT[]   = "http://192.168.0.120:81/stream";
static constexpr char CAM_RIGHT[]  = "http://192.168.0.121:81/stream";
static constexpr char CALIB_YAML[] = "config/stereo_calib.yml";
static constexpr char YOLO_MODEL[] = "config/yolov8n.onnx";
static constexpr char YOLO_NAMES[] = "config/coco.names";
static constexpr char PALM_MODEL[] = "models/palm_detection.onnx";
static constexpr char LM_MODEL[]   = "models/hand_landmark.onnx";

static constexpr int  SYNC_MS       = 250;   // max timestamp diff for stereo pair
static constexpr int  DISP_W        = 1280;
static constexpr int  DISP_H        = 480;

// ─── Trackbar state ──────────────────────────────────────────────────────────
struct Controls {
    int brightness   = 100;  // 0–200, 100 = no change
    int contrast     = 10;   // ×0.1 → 1.0
    int claheOn      = 1;
    int minArea      = 5;    // ×100 px²
    int numDispIdx   = 1;    // 0=64, 1=128, 2=256
    int blockSize    = 9;    // 3–21 odd
    int maxDepthCm   = 300;
    int confPct      = 45;   // confidence % (×0.01 → float)
};

static Controls g_ctrl;

static int numDispFromIdx(int idx) {
    const int vals[] = {64, 128, 256};
    return vals[std::max(0, std::min(idx, 2))];
}

// ─── Helpers ─────────────────────────────────────────────────────────────────
static cv::Mat applyBrightnessContrast(const cv::Mat& src, int bright, int cont) {
    double alpha = cont * 0.1;
    int    beta  = bright - 100;
    cv::Mat out;
    src.convertTo(out, -1, alpha, beta);
    return out;
}

static cv::Mat makeDisparityHeatmap(const cv::Mat& disp16, int numDisp) {
    // Convert CV_16S disparity (×16) to float pixels
    cv::Mat dispF;
    disp16.convertTo(dispF, CV_32F, 1.0 / 16.0);

    // Percentile clip: 2nd–98th percentile of valid pixels
    std::vector<float> vals;
    vals.reserve(dispF.rows * dispF.cols);
    for (int r = 0; r < dispF.rows; ++r) {
        const float* row = dispF.ptr<float>(r);
        for (int c = 0; c < dispF.cols; ++c) {
            if (row[c] > 0.5f) vals.push_back(row[c]);
        }
    }

    float lo = 1.0f, hi = static_cast<float>(numDisp);
    if (vals.size() > 20) {
        std::sort(vals.begin(), vals.end());
        lo = vals[static_cast<size_t>(vals.size() * 0.02f)];
        hi = vals[static_cast<size_t>(vals.size() * 0.98f)];
        if (hi <= lo) hi = lo + 1.0f;
    }

    cv::Mat norm(dispF.size(), CV_8U);
    for (int r = 0; r < dispF.rows; ++r) {
        const float* src = dispF.ptr<float>(r);
        uchar*       dst = norm.ptr<uchar>(r);
        for (int c = 0; c < dispF.cols; ++c) {
            if (src[c] <= 0.5f) { dst[c] = 0; continue; }
            float v = (src[c] - lo) / (hi - lo);
            v = std::max(0.0f, std::min(1.0f, v));
            dst[c] = static_cast<uchar>(v * 255);
        }
    }

    cv::Mat heat;
    cv::applyColorMap(norm, heat, cv::COLORMAP_TURBO);
    // Zero-disparity pixels → black
    cv::Mat mask;
    cv::threshold(norm, mask, 0, 255, cv::THRESH_BINARY_INV);
    heat.setTo(cv::Scalar(0, 0, 0), mask);
    return heat;
}

static float depthAtCenter(const cv::Mat& disp16, double focal, double baseline) {
    int cx = disp16.cols / 2, cy = disp16.rows / 2;
    int patch = 10;
    cv::Rect roi(cx - patch, cy - patch, patch * 2, patch * 2);
    roi &= cv::Rect(0, 0, disp16.cols, disp16.rows);
    cv::Mat region = disp16(roi);

    std::vector<float> vals;
    for (int r = 0; r < region.rows; ++r) {
        const int16_t* row = region.ptr<int16_t>(r);
        for (int c = 0; c < region.cols; ++c) {
            float d = row[c] / 16.0f;
            if (d > 0.5f) vals.push_back(d);
        }
    }
    if (vals.empty()) return 0.0f;
    std::sort(vals.begin(), vals.end());
    float med = vals[vals.size() / 2];
    return static_cast<float>(focal * baseline / med);
}

// ─── Detection overlay ───────────────────────────────────────────────────────
static void drawDetections(cv::Mat& canvas,
                            const std::vector<Detection>& dets,
                            const cv::Mat& disp16,
                            double focal, double baseline,
                            int maxDepthMm) {
    for (const auto& d : dets) {
        cv::Rect b = d.box;
        // Depth from disparity at box center
        cv::Point ctr(b.x + b.width / 2, b.y + b.height / 2);
        float depthMm = 0.0f;
        if (!disp16.empty() &&
            ctr.x >= 0 && ctr.x < disp16.cols &&
            ctr.y >= 0 && ctr.y < disp16.rows) {
            int16_t dv = disp16.at<int16_t>(ctr.y, ctr.x);
            if (dv > 8) depthMm = static_cast<float>(focal * baseline / (dv / 16.0));
        }
        if (maxDepthMm > 0 && depthMm > 0 && depthMm > maxDepthMm) continue;

        cv::Scalar color(0, 255, 0);
        cv::rectangle(canvas, b, color, 2);

        char txt[128];
        if (depthMm > 0)
            std::snprintf(txt, sizeof(txt), "%s %.0fcm %.0f%%",
                          d.label.c_str(), depthMm / 10.0f, d.confidence * 100.0f);
        else
            std::snprintf(txt, sizeof(txt), "%s %.0f%%",
                          d.label.c_str(), d.confidence * 100.0f);

        int baseline_ = 0;
        cv::Size ts = cv::getTextSize(txt, cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline_);
        cv::Rect bg(b.x, b.y - ts.height - 4, ts.width + 4, ts.height + 4);
        bg &= cv::Rect(0, 0, canvas.cols, canvas.rows);
        cv::rectangle(canvas, bg, color, cv::FILLED);
        cv::putText(canvas, txt, {b.x + 2, b.y - 2},
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1);
    }
}

// ─── Main ────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    (void)argc; (void)argv;

    // Stereo processor
    StereoProcessor stereo(CALIB_YAML);
    if (!stereo.isReady()) {
        std::cerr << "Calibration failed to load. Exiting.\n";
        return 1;
    }

    // Depth estimator
    DepthEstimator depth;

    // Hand tracker + AR filter (optional — silent if models not found)
    HandTracker handTracker;
    bool arEnabled = handTracker.load(PALM_MODEL, LM_MODEL);
    if (!arEnabled)
        std::cout << "[AR] Modelos no encontrados. Ejecuta: bash models/download_models.sh\n";
    float arAngle = 0.0f;

    // Object detector (optional)
    ObjectDetector detector;
    detector.load(YOLO_MODEL, YOLO_NAMES, 0.45f, 0.45f);

    // Cameras
    CameraStream camL(CAM_LEFT,  0);
    CameraStream camR(CAM_RIGHT, 1);
    camL.start();
    camR.start();

    // ── Ventana de controles ──────────────────────────────────────────────────
    const std::string winCtrl = "Controles";
    cv::namedWindow(winCtrl, cv::WINDOW_NORMAL);
    cv::resizeWindow(winCtrl, 400, 380);

    cv::createTrackbar("Brillo (0-200)",     winCtrl, &g_ctrl.brightness,  200);
    cv::createTrackbar("Contraste (x0.1)",   winCtrl, &g_ctrl.contrast,    30);
    cv::createTrackbar("CLAHE ON/OFF",        winCtrl, &g_ctrl.claheOn,     1);
    cv::createTrackbar("Area min (x100 px)", winCtrl, &g_ctrl.minArea,     50);
    cv::createTrackbar("numDisp 0=64 1=128", winCtrl, &g_ctrl.numDispIdx,  2);
    cv::createTrackbar("BlockSize (impar)",  winCtrl, &g_ctrl.blockSize,   21);
    cv::createTrackbar("Depth max (cm)",     winCtrl, &g_ctrl.maxDepthCm,  600);
    cv::createTrackbar("Conf % YOLO",        winCtrl, &g_ctrl.confPct,     100);

    // ── Ventana principal ─────────────────────────────────────────────────────
    const std::string winMain = "Stereo V2  |  C=calibrar  S=grabar  ESC=salir";
    cv::namedWindow(winMain, cv::WINDOW_NORMAL);
    cv::resizeWindow(winMain, DISP_W, DISP_H);

    // AVI recording
    cv::VideoWriter writer;
    bool recording = false;

    // Persistent state
    cv::Mat frameL, frameR, dispColor, lRectLast;
    float   zRawMm = 0, zFiltMm = 0;
    std::chrono::steady_clock::time_point tsL, tsR;
    bool hasDisp = false;
    std::vector<HandTracker::Hand> hands;

    // Manual calibration scale
    std::cout << "\nTeclas: C = calibrar escala  |  S = grabar/detener AVI  |  ESC = salir\n\n";

    for (;;) {
        cv::Mat newL, newR;
        std::chrono::steady_clock::time_point ntL, ntR;
        bool gotL = camL.getFrame(newL, ntL);
        bool gotR = camR.getFrame(newR, ntR);

        if (gotL) { frameL = newL; tsL = ntL; }
        if (gotR) { frameR = newR; tsR = ntR; }

        bool renderNeeded = gotL || gotR;

        // Stereo pair
        if (!frameL.empty() && !frameR.empty()) {
            auto dtMs = std::abs(
                            std::chrono::duration_cast<std::chrono::milliseconds>(
                                tsL - tsR).count());
            if (dtMs <= SYNC_MS) {
                // Build params from trackbars
                StereoParams sp;
                sp.numDisp   = numDispFromIdx(g_ctrl.numDispIdx);
                int bs = g_ctrl.blockSize | 1;  // force odd
                sp.blockSize = std::max(3, std::min(bs, 21));
                sp.clahe     = g_ctrl.claheOn != 0;

                cv::Mat lRect = stereo.rectifyLeft(frameL);
                cv::Mat rRect = stereo.rectifyRight(frameR);

                cv::Mat lGray, rGray;
                cv::cvtColor(lRect, lGray, cv::COLOR_BGR2GRAY);
                cv::cvtColor(rRect, rGray, cv::COLOR_BGR2GRAY);

                cv::Mat disp = stereo.computeDisparity(lGray, rGray, sp);

                // Depth at center
                float raw = depthAtCenter(disp, stereo.focalPx(), stereo.baselineMm());
                zRawMm  = raw;
                zFiltMm = depth.update(raw);

                // Disparity heatmap
                dispColor = makeDisparityHeatmap(disp, sp.numDisp);

                // YOLO detection on rectified left frame (scaled to 320×240)
                if (detector.isLoaded()) {
                    float conf = g_ctrl.confPct * 0.01f;
                    auto dets  = detector.detect(lRect, conf);
                    int maxMm  = g_ctrl.maxDepthCm * 10;
                    if (!dispColor.empty())
                        drawDetections(dispColor, dets, disp,
                                       stereo.focalPx(), stereo.baselineMm(), maxMm);
                }

                lRectLast = lRect;

                hasDisp = true;
                renderNeeded = true;
            }
        }

        // Render
        if (renderNeeded && !frameL.empty()) {
            cv::Mat left = applyBrightnessContrast(frameL,
                                                    g_ctrl.brightness,
                                                    g_ctrl.contrast);

            // Build 1280×480 canvas: left | disparity
            cv::Mat canvas(DISP_H, DISP_W, CV_8UC3, cv::Scalar(20, 20, 20));

            cv::Mat lScaled, dScaled;
            cv::resize(left, lScaled, cv::Size(DISP_W / 2, DISP_H));

            // Hand detection on display-resolution frame so coords match render target
            if (arEnabled)
                hands = handTracker.update(lScaled, 2);

            // AR filter overlay — usa el Z filtrado del pipeline estéreo
            if (arEnabled && !hands.empty())
                arAngle = ARFilter::render(lScaled, hands, arAngle,
                                           zFiltMm / 10.0f);  // mm → cm

            lScaled.copyTo(canvas(cv::Rect(0, 0, DISP_W / 2, DISP_H)));

            if (hasDisp && !dispColor.empty()) {
                cv::resize(dispColor, dScaled, cv::Size(DISP_W / 2, DISP_H));
                dScaled.copyTo(canvas(cv::Rect(DISP_W / 2, 0, DISP_W / 2, DISP_H)));
            }

            // Depth HUD
            char hud[128];
            float zRawCm  = zRawMm  / 10.0f;
            float zFiltCm = zFiltMm / 10.0f;
            if (zFiltCm > 1.0f)
                std::snprintf(hud, sizeof(hud),
                              "Z raw: %.1f cm   Z filt: %.1f cm", zRawCm, zFiltCm);
            else
                std::snprintf(hud, sizeof(hud), "Z: ---");

            cv::putText(canvas, hud,
                        {10, DISP_H - 12},
                        cv::FONT_HERSHEY_SIMPLEX, 0.65,
                        cv::Scalar(0, 0, 0), 3);
            cv::putText(canvas, hud,
                        {10, DISP_H - 12},
                        cv::FONT_HERSHEY_SIMPLEX, 0.65,
                        cv::Scalar(0, 255, 0), 1);

            // Crosshair center
            cv::Point ctr(DISP_W / 4, DISP_H / 2);
            cv::drawMarker(canvas, ctr, cv::Scalar(0, 0, 0),
                           cv::MARKER_CROSS, 18, 3);
            cv::drawMarker(canvas, ctr, cv::Scalar(0, 255, 0),
                           cv::MARKER_CROSS, 18, 1);

            // Recording indicator
            if (recording) {
                cv::circle(canvas, {DISP_W - 20, 20}, 8, cv::Scalar(0, 0, 255), cv::FILLED);
                writer.write(canvas);
            }

            cv::imshow(winMain, canvas);
        }

        // Keys
        int key = cv::waitKey(1) & 0xFF;
        if (key == 27) break;  // ESC

        if (key == 's' || key == 'S') {
            if (!recording) {
                std::string fname = "stereo_rec_" +
                    std::to_string(
                        std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::system_clock::now().time_since_epoch()).count()) +
                    ".avi";
                writer.open(fname,
                            cv::VideoWriter::fourcc('M','J','P','G'),
                            20, cv::Size(DISP_W, DISP_H));
                recording = writer.isOpened();
                std::cout << (recording ? "[Rec] Grabando: " + fname
                                        : "[Rec] Error al abrir archivo") << "\n";
            } else {
                writer.release();
                recording = false;
                std::cout << "[Rec] Detenido.\n";
            }
        }

        if (key == 'c' || key == 'C') {
            if (zFiltMm > 100.0f) {
                std::cout << "Ingresa la distancia real al objeto (cm): ";
                float real_cm;
                std::cin >> real_cm;
                if (real_cm > 5.0f) {
                    float prev = depth.scale;
                    depth.scale = (real_cm * 10.0f) / zFiltMm;
                    std::cout << "Escala: " << prev << " → " << depth.scale << "\n";
                }
            } else {
                std::cout << "[Calib] Apunta al objeto primero (Z debe ser >10cm)\n";
            }
        }
    }

    // Cleanup
    if (recording) writer.release();
    camL.stop();
    camR.stop();
    cv::destroyAllWindows();
    return 0;
}

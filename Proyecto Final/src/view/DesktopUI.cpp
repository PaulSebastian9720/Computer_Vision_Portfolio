#include "DesktopUI.hpp"
#include <cstdio>

namespace DesktopUI {
namespace {

static constexpr int PW = 640;
static constexpr int PH = 480;

cv::Mat toBGR(const cv::Mat& src) {
    if (src.channels() == 1) {
        cv::Mat out;
        cv::cvtColor(src, out, cv::COLOR_GRAY2BGR);
        return out;
    }
    return src.clone();
}

void drawCrosshair(cv::Mat& img, cv::Point c, int radius, int armLen,
                   cv::Scalar color, int thickness) {
    cv::circle(img, c, radius, color, thickness);
    cv::line(img, {c.x - armLen, c.y}, {c.x - radius, c.y}, color, thickness);
    cv::line(img, {c.x + radius, c.y}, {c.x + armLen, c.y}, color, thickness);
    cv::line(img, {c.x, c.y - armLen}, {c.x, c.y - radius}, color, thickness);
    cv::line(img, {c.x, c.y + radius}, {c.x, c.y + armLen}, color, thickness);
}

void overlayText(cv::Mat& img, const std::string& text, cv::Point pos,
                 double scale, cv::Scalar color, int thickness = 2) {
    cv::putText(img, text, pos, cv::FONT_HERSHEY_DUPLEX, scale,
                cv::Scalar(0, 0, 0), thickness + 2);
    cv::putText(img, text, pos, cv::FONT_HERSHEY_DUPLEX, scale,
                color, thickness);
}

} // anonymous namespace

void render(const cv::Mat& leftRect, const cv::Mat& dispColor,
            float zRawCm, float zKalmanCm) {
    cv::Mat canvas(PH, PW * 2, CV_8UC3, cv::Scalar(30, 30, 30));

    // --- Left panel: rectified camera con CLAHE para mejor visibilidad ---
    if (!leftRect.empty()) {
        cv::Mat cam;
        cv::resize(leftRect, cam, cv::Size(PW, PH));
        cam = toBGR(cam);
        // CLAHE por canal LAB: mejora contraste sin saturar colores
        cv::Mat lab;
        cv::cvtColor(cam, lab, cv::COLOR_BGR2Lab);
        std::vector<cv::Mat> channels;
        cv::split(lab, channels);
        auto clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        clahe->apply(channels[0], channels[0]);
        cv::merge(channels, lab);
        cv::cvtColor(lab, cam, cv::COLOR_Lab2BGR);
        // Unsharp mask: realza bordes sin aumentar ruido de fondo
        cv::Mat blurred;
        cv::GaussianBlur(cam, blurred, cv::Size(0, 0), 1.5);
        cv::addWeighted(cam, 1.4, blurred, -0.4, 0, cam);
        cam.copyTo(canvas(cv::Rect(0, 0, PW, PH)));
    }

    // Green crosshair at centre of left panel
    const cv::Point center{PW / 2, PH / 2};
    drawCrosshair(canvas, center, 10, 22, cv::Scalar(0, 0, 0),   3);
    drawCrosshair(canvas, center, 10, 22, cv::Scalar(0, 255, 0), 1);

    // Z Raw (red) — just above centre-bottom
    {
        char buf[64];
        if (zRawCm > 0.0f && zRawCm < 500.0f)
            std::snprintf(buf, sizeof(buf), "Z Raw: %.2f cm.", zRawCm);
        else
            std::snprintf(buf, sizeof(buf), "Z Raw: ---");
        overlayText(canvas, buf, {12, PH - 40}, 0.85, cv::Scalar(50, 50, 255));
    }

    // Z Kalman (green) — bottom row
    {
        char buf[64];
        if (zKalmanCm > 0.0f && zKalmanCm < 500.0f)
            std::snprintf(buf, sizeof(buf), "Z Kalman: %.2f cm.", zKalmanCm);
        else
            std::snprintf(buf, sizeof(buf), "Z Kalman: ---");
        overlayText(canvas, buf, {12, PH - 12}, 0.85, cv::Scalar(0, 255, 80));
    }

    overlayText(canvas, "CAMARA", {8, 26}, 0.65, cv::Scalar(180, 255, 180), 1);

    // --- Right panel: disparity map ---
    if (!dispColor.empty()) {
        cv::Mat disp;
        cv::resize(dispColor, disp, cv::Size(PW, PH));
        disp.copyTo(canvas(cv::Rect(PW, 0, PW, PH)));
    } else {
        overlayText(canvas, "calculando...", {PW + 20, PH / 2},
                    0.8, cv::Scalar(140, 140, 140), 1);
    }
    overlayText(canvas, "DISPARIDAD", {PW + 8, 26}, 0.65, cv::Scalar(180, 180, 255), 1);

    cv::line(canvas, {PW, 0}, {PW, PH}, cv::Scalar(80, 80, 80), 1);

    cv::imshow("Vision Estereo", canvas);
}

} // namespace DesktopUI

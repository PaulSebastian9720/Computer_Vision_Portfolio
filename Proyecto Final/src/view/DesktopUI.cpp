#include "DesktopUI.hpp"
#include <cstdio>
#include <vector>

namespace DesktopUI {

namespace {

cv::Mat toBGR(const cv::Mat& src) {
    if (src.channels() == 1) {
        cv::Mat out;
        cv::cvtColor(src, out, cv::COLOR_GRAY2BGR);
        return out;
    }
    return src;
}

cv::Mat resizeTo(const cv::Mat& src, cv::Size target) {
    cv::Mat out;
    cv::resize(src, out, target);
    return out;
}

} // anonymous namespace

namespace {

// Draws a crosshair + circle at the given point to mark the measurement target.
void drawCrosshair(cv::Mat& img, cv::Point center, int radius, int armLen,
                   cv::Scalar color, int thickness) {
    cv::circle(img, center, radius, color, thickness);
    cv::line(img, {center.x - armLen, center.y}, {center.x - radius, center.y}, color, thickness);
    cv::line(img, {center.x + radius, center.y}, {center.x + armLen, center.y}, color, thickness);
    cv::line(img, {center.x, center.y - armLen}, {center.x, center.y - radius}, color, thickness);
    cv::line(img, {center.x, center.y + radius}, {center.x, center.y + armLen}, color, thickness);
}

} // inner anonymous namespace

void renderMainWindow(const cv::Mat& leftRect,
                      const cv::Mat& rightRect,
                      const cv::Mat& dispColor) {
    if (leftRect.empty() || rightRect.empty() || dispColor.empty()) return;

    const cv::Size panel(320, 240);
    cv::Mat l = toBGR(resizeTo(leftRect,  panel));
    cv::Mat r = toBGR(resizeTo(rightRect, panel));
    cv::Mat d = toBGR(resizeTo(dispColor, panel));

    // Small crosshair on the left panel so the user can align the object
    drawCrosshair(l, {panel.width / 2, panel.height / 2}, 10, 20,
                  cv::Scalar(0, 255, 0), 1);

    cv::Mat row;
    cv::hconcat(std::vector<cv::Mat>{l, r, d}, row);

    const cv::Scalar red(0, 0, 220);
    cv::putText(row, "cam iz",  {10,  22}, cv::FONT_HERSHEY_SIMPLEX, 0.55, red, 1);
    cv::putText(row, "cam de",  {330, 22}, cv::FONT_HERSHEY_SIMPLEX, 0.55, red, 1);
    cv::putText(row, "mapa de", {650, 22}, cv::FONT_HERSHEY_SIMPLEX, 0.55, red, 1);

    cv::imshow("Monitor", row);
}

void renderOutputWindow(const cv::Mat& colorRect, float distanceCm) {
    if (colorRect.empty()) return;

    cv::Mat out;
    cv::resize(colorRect, out, cv::Size(640, 480));
    out = toBGR(out);

    // Large crosshair at image centre — marks the exact pixel used for Z
    const cv::Point center{out.cols / 2, out.rows / 2};
    drawCrosshair(out, center, 20, 40, cv::Scalar(0, 0, 0),   3); // shadow
    drawCrosshair(out, center, 20, 40, cv::Scalar(0, 255, 0), 1); // green

    if (distanceCm > 0.0f && distanceCm < 500.0f) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "Z: %.1f cm", distanceCm);
        // Shadow for readability on any background
        cv::putText(out, buf, {18, 458}, cv::FONT_HERSHEY_DUPLEX, 1.1,
                    cv::Scalar(0, 0, 0), 4);
        cv::putText(out, buf, {18, 458}, cv::FONT_HERSHEY_DUPLEX, 1.1,
                    cv::Scalar(0, 255, 80), 2);
    }

    cv::imshow("salida", out);
}

} // namespace DesktopUI

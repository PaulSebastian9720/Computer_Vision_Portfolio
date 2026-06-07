#pragma once
#include <opencv2/opencv.hpp>

namespace DesktopUI {

// Single combined window: left panel = camera + crosshair + Z Raw + Z Kalman,
// right panel = disparity map (JET colormap). dispColor may be empty.
// zRawCm / zKalmanCm <= 0 → show "---".
void render(const cv::Mat& leftRect, const cv::Mat& dispColor,
            float zRawCm, float zKalmanCm);

} // namespace DesktopUI

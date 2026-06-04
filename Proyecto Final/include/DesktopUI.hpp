#pragma once
#include <opencv2/opencv.hpp>

namespace DesktopUI {

// Main monitoring window: 3 side-by-side panels (left cam, right cam, disparity).
void renderMainWindow(const cv::Mat& leftRect,
                      const cv::Mat& rightRect,
                      const cv::Mat& dispColor);

// Secondary output window: color rectified frame with distance overlay.
void renderOutputWindow(const cv::Mat& colorRect, float distanceCm);

} // namespace DesktopUI

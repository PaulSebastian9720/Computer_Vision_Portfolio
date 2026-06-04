#include "StereoService.hpp"
#include <iostream>

StereoService::StereoService(const std::string& calibYamlPath) {
    cv::FileStorage fs(calibYamlPath, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "[StereoService] Cannot open: " << calibYamlPath
                  << " — running without rectification.\n";
        return;
    }

    fs["K_l"]    >> K_l_;
    fs["K_r"]    >> K_r_;
    fs["dist_l"] >> dist_l_;
    fs["dist_r"] >> dist_r_;
    fs["R"]      >> R_;
    fs["T"]      >> T_;
    fs.release();

    if (K_l_.empty() || K_r_.empty() || R_.empty() || T_.empty()) {
        std::cerr << "[StereoService] Calibration file is incomplete.\n";
        return;
    }

    // Compute stereo rectification for VGA frames
    cv::Size imgSize(640, 480);
    cv::Rect validRoiL, validRoiR;
    cv::stereoRectify(K_l_, dist_l_, K_r_, dist_r_, imgSize,
                      R_, T_, R1_, R2_, P1_, P2_, Q_,
                      cv::CALIB_ZERO_DISPARITY, -1, imgSize,
                      &validRoiL, &validRoiR);

    cv::initUndistortRectifyMap(K_l_, dist_l_, R1_, P1_,
                                imgSize, CV_32FC1, mapLx_, mapLy_);
    cv::initUndistortRectifyMap(K_r_, dist_r_, R2_, P2_,
                                imgSize, CV_32FC1, mapRx_, mapRy_);

    // CLAHE for adaptive illumination before block matching
    clahe_ = cv::createCLAHE(2.0, cv::Size(8, 8));

    // StereoSGBM — tunable starting parameters for VGA + ESP32-CAM
    const int blockSize = 7;    // smaller block = más detalle, menos costo computacional
    const int numDisp   = 64;   // 64 vs 96 = ~33% más rápido; cubre hasta ~1.5 m
    const int P1        = 8  * 3 * blockSize * blockSize;
    const int P2        = 32 * 3 * blockSize * blockSize;

    sgbm_ = cv::StereoSGBM::create(
        0,          // minDisparity
        numDisp,
        blockSize,
        P1,
        P2,
        1,          // disp12MaxDiff
        10,         // preFilterCap (5–63)
        8,          // uniquenessRatio (5–15)
        100,        // speckleWindowSize (50–200)
        32,         // speckleRange
        cv::StereoSGBM::MODE_SGBM_3WAY
    );

    calibrated_ = true;
    std::cout << "[StereoService] Calibration loaded. Q matrix ready.\n";
}

cv::Mat StereoService::process(const cv::Mat& leftGray,
                                const cv::Mat& rightGray) {
    // CLAHE + mild blur to handle JPEG block artifacts and variable lighting
    cv::Mat l, r;
    clahe_->apply(leftGray,  l);
    clahe_->apply(rightGray, r);
    cv::GaussianBlur(l, l, cv::Size(3, 3), 0.8);
    cv::GaussianBlur(r, r, cv::Size(3, 3), 0.8);

    cv::Mat disp;
    sgbm_->compute(l, r, disp);
    return disp;  // CV_16S, scale ×16
}

cv::Mat StereoService::rectifyLeft(const cv::Mat& frame) const {
    cv::Mat out;
    cv::remap(frame, out, mapLx_, mapLy_, cv::INTER_LINEAR);
    return out;
}

cv::Mat StereoService::rectifyRight(const cv::Mat& frame) const {
    cv::Mat out;
    cv::remap(frame, out, mapRx_, mapRy_, cv::INTER_LINEAR);
    return out;
}

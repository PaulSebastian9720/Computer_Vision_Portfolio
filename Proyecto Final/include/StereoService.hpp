#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <string>

class StereoService {
public:
    explicit StereoService(const std::string& calibYamlPath);

    // Input: rectified grayscale frames.
    // Returns raw disparity (CV_16S, internal scale ×16).
    cv::Mat process(const cv::Mat& leftGray, const cv::Mat& rightGray);

    // Rectify a full-color frame with the loaded calibration maps.
    cv::Mat rectifyLeft(const cv::Mat& frame)  const;
    cv::Mat rectifyRight(const cv::Mat& frame) const;

    const cv::Mat& getQ()    const { return Q_; }
    bool           isCalibrated() const { return calibrated_; }

private:
    // Intrinsics & extrinsics loaded from YAML
    cv::Mat K_l_, K_r_, dist_l_, dist_r_, R_, T_;

    // Rectification outputs
    cv::Mat R1_, R2_, P1_, P2_, Q_;
    cv::Mat mapLx_, mapLy_, mapRx_, mapRy_;

    cv::Ptr<cv::StereoSGBM> sgbm_;
    cv::Ptr<cv::CLAHE>      clahe_;

    bool calibrated_{false};
};

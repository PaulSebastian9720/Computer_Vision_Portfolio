#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/ximgproc.hpp>
#include <string>

class StereoService {
public:
    explicit StereoService(const std::string& calibYamlPath);

    // Input: rectified grayscale frames.
    // Returns WLS-filtered disparity (CV_16S, scale ×16).
    cv::Mat process(const cv::Mat& leftGray, const cv::Mat& rightGray);

    cv::Mat rectifyLeft(const cv::Mat& frame)  const;
    cv::Mat rectifyRight(const cv::Mat& frame) const;

    const cv::Mat& getQ()         const { return Q_; }
    bool           isCalibrated() const { return calibrated_; }
    int            numDisp()      const { return numDisp_; }

private:
    cv::Mat K_l_, K_r_, dist_l_, dist_r_, R_, T_;
    cv::Mat R1_, R2_, P1_, P2_, Q_;
    cv::Mat mapLx_, mapLy_, mapRx_, mapRy_;

    cv::Ptr<cv::StereoSGBM>                       sgbm_;
    cv::Ptr<cv::StereoMatcher>                     rightMatcher_;
    cv::Ptr<cv::ximgproc::DisparityWLSFilter>      wlsFilter_;
    cv::Ptr<cv::CLAHE>                             clahe_;

    cv::Rect validRoiL_, validRoiR_;  // valid pixel regions after rectification
    int      numDisp_{96};

    bool calibrated_{false};
};

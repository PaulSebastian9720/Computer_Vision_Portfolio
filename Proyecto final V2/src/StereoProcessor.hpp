#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <string>

struct StereoParams {
    int  numDisp   = 128;  // must be multiple of 16
    int  blockSize = 9;    // must be odd ≥ 3
    bool clahe     = true;
};

class StereoProcessor {
public:
    explicit StereoProcessor(const std::string& calibYaml);

    bool           isReady()     const { return ready_; }
    const cv::Mat& Q()           const { return Q_; }
    double         focalPx()     const { return focal_; }
    double         baselineMm()  const { return baseline_; }
    int            numDisp()     const { return built_.numDisp; }

    cv::Mat rectifyLeft (const cv::Mat& frame) const;
    cv::Mat rectifyRight(const cv::Mat& frame) const;

    // Returns WLS-filtered disparity CV_16S (×16 internal scale)
    cv::Mat computeDisparity(const cv::Mat& leftGray,
                              const cv::Mat& rightGray,
                              const StereoParams& params);

private:
    void rebuildMatcher(const StereoParams& p);

    cv::Mat K_l_, K_r_, dist_l_, dist_r_, R_, T_;
    cv::Mat R1_, R2_, P1_, P2_, Q_;
    cv::Mat mapLx_, mapLy_, mapRx_, mapRy_;
    cv::Rect roiL_, roiR_;

    cv::Ptr<cv::StereoSGBM>                   sgbm_;
    cv::Ptr<cv::StereoMatcher>                rightMatcher_;
    cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_;
    cv::Ptr<cv::CLAHE>                        clahe_;

    double focal_    = 0;
    double baseline_ = 0;
    bool   ready_    = false;
    StereoParams built_;
};

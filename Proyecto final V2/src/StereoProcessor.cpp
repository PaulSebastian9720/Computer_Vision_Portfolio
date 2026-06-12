#include "StereoProcessor.hpp"
#include <iostream>

static void destripe(cv::Mat& g) {
    double gm = cv::mean(g)[0];
    if (gm < 1.0) return;
    for (int r = 0; r < g.rows; ++r) {
        auto* row = g.ptr<uchar>(r);
        double s = 0;
        for (int c = 0; c < g.cols; ++c) s += row[c];
        double rm = s / g.cols;
        if (rm < 5.0) continue;
        double k = std::max(0.70, std::min(1.43, gm / rm));
        for (int c = 0; c < g.cols; ++c)
            row[c] = cv::saturate_cast<uchar>(row[c] * k);
    }
}

StereoProcessor::StereoProcessor(const std::string& calibYaml) {
    cv::FileStorage fs(calibYaml, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "[Stereo] Cannot open: " << calibYaml << "\n";
        return;
    }
    fs["K_l"] >> K_l_; fs["K_r"] >> K_r_;
    fs["dist_l"] >> dist_l_; fs["dist_r"] >> dist_r_;
    fs["R"] >> R_; fs["T"] >> T_;
    fs.release();

    if (K_l_.empty() || T_.empty()) {
        std::cerr << "[Stereo] Incomplete calibration file.\n";
        return;
    }

    cv::Size sz(320, 240);
    cv::stereoRectify(K_l_, dist_l_, K_r_, dist_r_, sz, R_, T_,
                      R1_, R2_, P1_, P2_, Q_,
                      cv::CALIB_ZERO_DISPARITY, 0, sz, &roiL_, &roiR_);
    cv::initUndistortRectifyMap(K_l_, dist_l_, R1_, P1_, sz, CV_32FC1, mapLx_, mapLy_);
    cv::initUndistortRectifyMap(K_r_, dist_r_, R2_, P2_, sz, CV_32FC1, mapRx_, mapRy_);

    clahe_    = cv::createCLAHE(3.0, cv::Size(8, 8));
    focal_    = K_l_.at<double>(0, 0);
    baseline_ = cv::norm(T_);

    rebuildMatcher(StereoParams{});
    ready_ = true;

    std::cout << "[Stereo] baseline=" << baseline_ << "mm  focal=" << focal_ << "px\n"
              << "[Stereo] range: " << focal_ * baseline_ / built_.numDisp
              << " – " << focal_ * baseline_ / 2 << " mm\n";
}

void StereoProcessor::rebuildMatcher(const StereoParams& p) {
    int nd = ((p.numDisp + 15) / 16) * 16;
    if (nd < 16) nd = 16;
    int bs = (p.blockSize % 2 == 0) ? p.blockSize + 1 : p.blockSize;
    if (bs < 3) bs = 3;

    const int P1 = 8  * 3 * bs * bs;
    const int P2 = 32 * 3 * bs * bs;

    sgbm_ = cv::StereoSGBM::create(0, nd, bs, P1, P2,
                                    5, 30, 12, 200, 2,
                                    cv::StereoSGBM::MODE_SGBM);
    rightMatcher_ = cv::ximgproc::createRightMatcher(sgbm_);
    wls_ = cv::ximgproc::createDisparityWLSFilter(sgbm_);
    wls_->setLambda(18000.0);
    wls_->setSigmaColor(2.0);

    built_ = p;
    built_.numDisp   = nd;
    built_.blockSize = bs;
}

cv::Mat StereoProcessor::computeDisparity(const cv::Mat& lg, const cv::Mat& rg,
                                           const StereoParams& params) {
    if (params.numDisp   != built_.numDisp ||
        params.blockSize != built_.blockSize)
        rebuildMatcher(params);

    cv::Mat l, r;
    if (params.clahe) {
        clahe_->apply(lg, l);
        clahe_->apply(rg, r);
    } else {
        l = lg.clone(); r = rg.clone();
    }

    // Brightness equalization between cameras
    cv::Scalar ml = cv::mean(l), mr = cv::mean(r);
    if (mr[0] > 1.0) r.convertTo(r, CV_8U, ml[0] / mr[0]);

    destripe(l);
    destripe(r);
    // Blur horizontal (5×1): atenúa rayas horizontales del OV2640
    // sin borrar los bordes verticales que SGBM necesita para matching
    cv::GaussianBlur(l, l, {5, 1}, 1.2);
    cv::GaussianBlur(r, r, {5, 1}, 1.2);

    cv::Mat dL, dR;
    sgbm_->compute(l, r, dL);
    rightMatcher_->compute(r, l, dR);

    cv::Rect roi = roiL_ & roiR_;
    cv::Mat filtered;
    if (roi.area() > 0)
        wls_->filter(dL, l, filtered, dR, roi);
    else
        wls_->filter(dL, l, filtered, dR);

    return filtered;
}

cv::Mat StereoProcessor::rectifyLeft(const cv::Mat& f) const {
    cv::Mat out;
    cv::remap(f, out, mapLx_, mapLy_, cv::INTER_LINEAR);
    return out;
}

cv::Mat StereoProcessor::rectifyRight(const cv::Mat& f) const {
    cv::Mat out;
    cv::remap(f, out, mapRx_, mapRy_, cv::INTER_LINEAR);
    return out;
}

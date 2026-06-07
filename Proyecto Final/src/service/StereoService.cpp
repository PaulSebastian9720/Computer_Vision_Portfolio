#include "StereoService.hpp"
#include <iostream>

// Removes horizontal banding caused by AC-light flicker and rolling shutter.
// Normalises each row to the global mean so the SGBM block matcher doesn't
// confuse brightness stripes with real depth edges.
static void destripe(cv::Mat& gray) {
    double gMean = cv::mean(gray)[0];
    if (gMean < 1.0) return;
    for (int r = 0; r < gray.rows; ++r) {
        uchar* row = gray.ptr<uchar>(r);
        double rSum = 0;
        for (int c = 0; c < gray.cols; ++c) rSum += row[c];
        double rMean = rSum / gray.cols;
        if (rMean < 1.0) continue;
        double gain = gMean / rMean;
        // Clamp to ±30% to avoid overcorrecting very dark/bright rows
        gain = std::max(0.70, std::min(1.43, gain));
        for (int c = 0; c < gray.cols; ++c)
            row[c] = cv::saturate_cast<uchar>(row[c] * gain);
    }
}

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

    cv::Size imgSize(320, 240);

    // alpha=0: solo píxeles válidos en la imagen rectificada → sin bordes negros
    // que confunden al SGBM con matches falsos.
    cv::stereoRectify(K_l_, dist_l_, K_r_, dist_r_, imgSize,
                      R_, T_, R1_, R2_, P1_, P2_, Q_,
                      cv::CALIB_ZERO_DISPARITY, 0, imgSize,
                      &validRoiL_, &validRoiR_);

    cv::initUndistortRectifyMap(K_l_, dist_l_, R1_, P1_,
                                imgSize, CV_32FC1, mapLx_, mapLy_);
    cv::initUndistortRectifyMap(K_r_, dist_r_, R2_, P2_,
                                imgSize, CV_32FC1, mapRx_, mapRy_);

    clahe_ = cv::createCLAHE(3.0, cv::Size(8, 8));

    double baseline_mm = cv::norm(T_);
    double focal_px    = K_l_.at<double>(0, 0);
    std::cout << "[StereoService] Baseline: " << baseline_mm << " mm\n";
    std::cout << "[StereoService] ValidROI L: " << validRoiL_
              << "  R: " << validRoiR_ << "\n";

    numDisp_ = 96;
    const int blockSize = 9;
    const int P1        = 8  * 3 * blockSize * blockSize;
    const int P2        = 32 * 3 * blockSize * blockSize;

    std::cout << "[StereoService] Measurable range: "
              << (focal_px * baseline_mm / numDisp_) << " – "
              << (focal_px * baseline_mm / 2) << " mm\n";

    sgbm_ = cv::StereoSGBM::create(
        0,          // minDisparity
        numDisp_,
        blockSize,
        P1,
        P2,
        2,          // disp12MaxDiff: más tolerante a inconsistencias L/R
        15,         // preFilterCap: mejor para escenas de interior normales
        8,          // uniquenessRatio
        150,        // speckleWindowSize
        2,          // speckleRange
        cv::StereoSGBM::MODE_SGBM_3WAY
    );

    rightMatcher_ = cv::ximgproc::createRightMatcher(sgbm_);
    wlsFilter_    = cv::ximgproc::createDisparityWLSFilter(sgbm_);
    wlsFilter_->setLambda(12000.0);   // menos agresivo → mejor detalle de bordes
    wlsFilter_->setSigmaColor(1.5);   // más tolerante a variaciones de color

    calibrated_ = true;
    std::cout << "[StereoService] Calibration loaded. Q matrix ready.\n";
}

cv::Mat StereoService::process(const cv::Mat& leftGray,
                                const cv::Mat& rightGray) {
    cv::Mat l, r;
    clahe_->apply(leftGray,  l);
    clahe_->apply(rightGray, r);

    // Normalización de brillo L/R: compensa diferencias de exposición entre cámaras.
    cv::Scalar meanL = cv::mean(l);
    cv::Scalar meanR = cv::mean(r);
    if (meanR[0] > 1.0) {
        double gain = meanL[0] / meanR[0];
        r.convertTo(r, CV_8U, gain);
    }

    // Destripe: normaliza el brillo por fila para eliminar las bandas horizontales
    // del parpadeo AC. Imprescindible para que SGBM no confunda rayas con bordes reales.
    destripe(l);
    destripe(r);

    // Solo blur vertical 1×3: el downscale VGA→QVGA ya actúa como anti-alias horizontal.
    // El 3×3 adicional borraba detalle sin beneficio real.
    cv::GaussianBlur(l, l, cv::Size(1, 3), 0, 0.8);
    cv::GaussianBlur(r, r, cv::Size(1, 3), 0, 0.8);

    cv::Mat dispL, dispR;
    sgbm_->compute(l, r, dispL);
    rightMatcher_->compute(r, l, dispR);

    // Pasar ROI válido al WLS: el filtro sabe qué zonas son confiables.
    cv::Rect roi = validRoiL_ & validRoiR_;
    cv::Mat filtered;
    if (roi.area() > 0)
        wlsFilter_->filter(dispL, l, filtered, dispR, roi);
    else
        wlsFilter_->filter(dispL, l, filtered, dispR);

    return filtered;
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

#include "HandTracker.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>

// ─── Blob helper ─────────────────────────────────────────────────────────────
cv::Mat HandTracker::nhwcBlob(const cv::Mat& bgr, int W, int H) {
    cv::Mat rgb, resized, f;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, resized, cv::Size(W, H));
    resized.convertTo(f, CV_32F, 1.0 / 255.0);
    // f is [H, W, 3] HWC → add batch dim → [1, H, W, 3]
    std::vector<int> shape = {1, H, W, 3};
    return f.reshape(1, shape);
}

// ─── Anchor generation ───────────────────────────────────────────────────────
// Palm detection 128×128: strides [8,16], anchors_per_cell [2,6]
// Total: 16*16*2 + 8*8*6 = 512 + 384 = 896
void HandTracker::genAnchors() {
    const int strides[] = {8, 16};
    const int apc[]     = {2, 6};
    for (int s = 0; s < 2; ++s) {
        int fm = 128 / strides[s];
        for (int y = 0; y < fm; ++y)
            for (int x = 0; x < fm; ++x)
                for (int a = 0; a < apc[s]; ++a)
                    anchors_.push_back({(x + 0.5f) / fm, (y + 0.5f) / fm});
    }
}

// ─── Load ────────────────────────────────────────────────────────────────────
bool HandTracker::load(const std::string& palmOnnx,
                        const std::string& lmOnnx,
                        float det, float nms) {
    detThr_ = det; nmsThr_ = nms;
    try {
        palmNet_ = cv::dnn::readNetFromONNX(palmOnnx);
        lmNet_   = cv::dnn::readNetFromONNX(lmOnnx);
    } catch (const cv::Exception& e) {
        std::cerr << "[Hand] Load error: " << e.what() << "\n";
        return false;
    }
    if (palmNet_.empty() || lmNet_.empty()) {
        std::cerr << "[Hand] Empty model after load.\n";
        return false;
    }
    palmNet_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    palmNet_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    lmNet_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    lmNet_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    genAnchors();
    loaded_ = true;
    std::cout << "[Hand] Models loaded. Anchors: " << anchors_.size() << "\n";
    return true;
}

// ─── Palm detection postprocessing ───────────────────────────────────────────
std::vector<HandTracker::RawDet> HandTracker::decodePalms(
    const std::vector<cv::Mat>& outs, int imgW, int imgH) {

    // Find scores [N,1] and regressors [N,18] by output size
    const cv::Mat* scoresMat = nullptr;
    const cv::Mat* regsMat   = nullptr;

    for (const auto& o : outs) {
        cv::Mat flat = o.reshape(1, (int)o.total() > 0 ? -1 : 1);
        int total = (int)o.total();
        if (total % 18 == 0 && total / 18 == (int)anchors_.size())
            regsMat   = &o;
        else if (total == (int)anchors_.size())
            scoresMat = &o;
    }

    if (!scoresMat || !regsMat) {
        std::cerr << "[Hand] Unexpected palm detection output shapes.\n";
        for (size_t i = 0; i < outs.size(); ++i) {
            std::cerr << "  out[" << i << "] total=" << outs[i].total() << "\n";
        }
        return {};
    }

    cv::Mat scores = scoresMat->reshape(1, (int)anchors_.size()); // [896, 1]
    cv::Mat regs   = regsMat->reshape(1, (int)anchors_.size());   // [896, 18]

    std::vector<RawDet> out;
    int N = (int)anchors_.size();

    for (int i = 0; i < N; ++i) {
        float raw = scores.at<float>(i, 0);
        float sc  = 1.0f / (1.0f + std::exp(-raw));
        if (sc < detThr_) continue;

        const float* r  = regs.ptr<float>(i);
        const Anchor& a = anchors_[i];

        RawDet d;
        d.score = sc;
        // Decode: offset / 128.0 + anchor center (in [0,1]), then scale to pixels
        d.cx = (a.cx + r[0] / 128.0f) * imgW;
        d.cy = (a.cy + r[1] / 128.0f) * imgH;
        d.w  = (r[2] / 128.0f) * imgW;
        d.h  = (r[3] / 128.0f) * imgH;
        for (int k = 0; k < 7; ++k) {
            d.kp[2*k]   = (a.cx + r[4 + 2*k]   / 128.0f) * imgW;
            d.kp[2*k+1] = (a.cy + r[4 + 2*k+1] / 128.0f) * imgH;
        }
        out.push_back(d);
    }
    return out;
}

// ─── NMS ─────────────────────────────────────────────────────────────────────
std::vector<HandTracker::RawDet> HandTracker::nms(std::vector<RawDet>& dets) {
    if (dets.empty()) return {};
    std::sort(dets.begin(), dets.end(),
              [](const RawDet& a, const RawDet& b){ return a.score > b.score; });

    auto iou = [](const RawDet& a, const RawDet& b) {
        float ax1 = a.cx - a.w * 0.5f, ay1 = a.cy - a.h * 0.5f;
        float ax2 = a.cx + a.w * 0.5f, ay2 = a.cy + a.h * 0.5f;
        float bx1 = b.cx - b.w * 0.5f, by1 = b.cy - b.h * 0.5f;
        float bx2 = b.cx + b.w * 0.5f, by2 = b.cy + b.h * 0.5f;
        float ix1 = std::max(ax1, bx1), iy1 = std::max(ay1, by1);
        float ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);
        if (ix2 <= ix1 || iy2 <= iy1) return 0.0f;
        float inter = (ix2 - ix1) * (iy2 - iy1);
        float ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter;
        return (ua > 0) ? inter / ua : 0.0f;
    };

    std::vector<bool> sup(dets.size(), false);
    std::vector<RawDet> result;
    for (size_t i = 0; i < dets.size(); ++i) {
        if (sup[i]) continue;
        result.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j)
            if (!sup[j] && iou(dets[i], dets[j]) > nmsThr_)
                sup[j] = true;
    }
    return result;
}

// ─── Landmark detection ──────────────────────────────────────────────────────
HandTracker::Hand HandTracker::runLandmark(const cv::Mat& bgr,
                                             const RawDet& d) {
    Hand h;

    // Expand ROI around detected palm
    float expand = 1.8f;
    int side = (int)(std::max(d.w, d.h) * expand * 0.5f);
    int cx   = (int)d.cx, cy = (int)d.cy;

    cv::Rect roi(cx - side, cy - side, side * 2, side * 2);
    roi &= cv::Rect(0, 0, bgr.cols, bgr.rows);
    if (roi.area() < 100) return h;

    cv::Mat blob = nhwcBlob(bgr(roi), 224, 224);
    lmNet_.setInput(blob);

    std::vector<cv::Mat> outs;
    lmNet_.forward(outs, lmNet_.getUnconnectedOutLayersNames());

    // Find the 63-value landmark output (21 × xyz)
    const cv::Mat* lmOut = nullptr;
    for (const auto& o : outs) {
        if ((int)o.total() == 63) { lmOut = &o; break; }
    }
    if (!lmOut) {
        // Some exports have 21*3=63 in different shapes; try best match
        for (const auto& o : outs) {
            if ((int)o.total() >= 63) { lmOut = &o; break; }
        }
    }
    if (!lmOut) return h;

    const float* lm = reinterpret_cast<const float*>(lmOut->data);

    // Landmark values are normalized [0,1] in the 224×224 crop space
    // Some PINTO0309 models output pixel coords in 0-224 range; divide by 224
    // Detect scale: if max(x,y) > 1.5 assume pixel coords
    float maxVal = 0;
    for (int i = 0; i < 21; ++i) {
        maxVal = std::max(maxVal, std::max(lm[3*i], lm[3*i+1]));
    }
    float scale = (maxVal > 1.5f) ? (1.0f / 224.0f) : 1.0f;

    h.conf = d.score;
    for (int i = 0; i < 21; ++i) {
        float nx = lm[3*i]   * scale;  // normalized [0,1] in crop
        float ny = lm[3*i+1] * scale;
        h.lm[i].x = roi.x + nx * roi.width;
        h.lm[i].y = roi.y + ny * roi.height;
    }

    // Handedness from largest non-63 output
    for (const auto& o : outs) {
        if ((int)o.total() == 1) {
            h.isRight = (o.at<float>(0) > 0.5f);
            break;
        }
    }

    // Palm center (landmarks 0, 5, 9, 13, 17)
    int pidx[] = {0, 5, 9, 13, 17};
    float px = 0, py = 0;
    for (int i : pidx) { px += h.lm[i].x; py += h.lm[i].y; }
    h.palmCenter = {(int)(px / 5), (int)(py / 5)};

    // Radius: palm center to middle fingertip (landmark 12)
    float dx = h.lm[12].x - h.palmCenter.x;
    float dy = h.lm[12].y - h.palmCenter.y;
    h.radiusPx = std::sqrt(dx*dx + dy*dy) * 0.90f;
    h.radiusPx = std::max(45.0f, std::min(h.radiusPx, 210.0f));

    return h;
}

// ─── Main update ─────────────────────────────────────────────────────────────
std::vector<HandTracker::Hand> HandTracker::update(const cv::Mat& bgr,
                                                     int maxHands) {
    if (!loaded_ || bgr.empty()) return {};

    // Run palm detection on 128×128
    cv::Mat blob = nhwcBlob(bgr, 128, 128);
    palmNet_.setInput(blob);

    std::vector<cv::Mat> outs;
    palmNet_.forward(outs, palmNet_.getUnconnectedOutLayersNames());

    auto rawDets = decodePalms(outs, bgr.cols, bgr.rows);
    auto filtered = nms(rawDets);

    int limit = std::min((int)filtered.size(), maxHands);
    std::vector<Hand> hands;
    hands.reserve(limit);

    for (int i = 0; i < limit; ++i) {
        Hand h = runLandmark(bgr, filtered[i]);
        if (h.conf > 0) hands.push_back(h);
    }
    return hands;
}

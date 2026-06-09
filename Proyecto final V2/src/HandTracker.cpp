#include "HandTracker.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>

// ─── Blob helper ─────────────────────────────────────────────────────────────
cv::Mat HandTracker::nhwcBlob(const cv::Mat& bgr, int W, int H) {
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(W, H));
    // blobFromImage: BGR→RGB swap, normalize [0,1], NCHW [1,3,H,W]
    return cv::dnn::blobFromImage(resized, 1.0 / 255.0, cv::Size(W, H),
                                  cv::Scalar(), true, false, CV_32F);
}

// ─── Anchor generation ───────────────────────────────────────────────────────
// Palm detection 192×192: strides [8,16], anchors_per_cell [2,6]
// Total: 24*24*2 + 12*12*6 = 1152 + 864 = 2016
void HandTracker::genAnchors() {
    const int strides[] = {8, 16};
    const int apc[]     = {2, 6};
    for (int s = 0; s < 2; ++s) {
        int fm = 192 / strides[s];
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

    // Use raw pointer access to avoid reshape() on 3D tensors [1, N, D]
    const float* scoresPtr = reinterpret_cast<const float*>(scoresMat->data);
    const float* regsPtr   = reinterpret_cast<const float*>(regsMat->data);

    std::vector<RawDet> out;
    int N = (int)anchors_.size();

    for (int i = 0; i < N; ++i) {
        float sc = 1.0f / (1.0f + std::exp(-scoresPtr[i]));  // sigmoid on raw logits
        if (sc < detThr_) continue;

        const float* r  = regsPtr + i * 18;
        const Anchor& a = anchors_[i];

        RawDet d;
        d.score = sc;
        // Decode: offset / 192.0 + anchor center (in [0,1]), then scale to pixels
        d.cx = (a.cx + r[0] / 192.0f) * imgW;
        d.cy = (a.cy + r[1] / 192.0f) * imgH;
        d.w  = (r[2] / 192.0f) * imgW;
        d.h  = (r[3] / 192.0f) * imgH;
        for (int k = 0; k < 7; ++k) {
            d.kp[2*k]   = (a.cx + r[4 + 2*k]   / 192.0f) * imgW;
            d.kp[2*k+1] = (a.cy + r[4 + 2*k+1] / 192.0f) * imgH;
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
HandTracker::Hand HandTracker::runLandmark(const cv::Mat& /*bgr*/,
                                             const RawDet& d) {
    Hand h;
    h.conf       = d.score;
    h.palmCenter = { (int)d.cx, (int)d.cy };
    float r      = std::max(d.w, d.h) * 0.55f;
    h.radiusPx   = std::max(40.0f, std::min(r, 200.0f));
    // Fill lm[] with palmCenter so any code that reads landmarks doesn't crash
    for (auto& lm : h.lm) { lm.x = (float)h.palmCenter.x; lm.y = (float)h.palmCenter.y; }
    return h;
}

// ─── Main update ─────────────────────────────────────────────────────────────
std::vector<HandTracker::Hand> HandTracker::update(const cv::Mat& bgr,
                                                     int maxHands) {
    if (!loaded_ || bgr.empty()) return {};

    // Run palm detection on 192×192
    cv::Mat blob = nhwcBlob(bgr, 192, 192);
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

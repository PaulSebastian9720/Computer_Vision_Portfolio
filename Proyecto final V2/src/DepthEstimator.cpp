#include "DepthEstimator.hpp"
#include <algorithm>
#include <cmath>

DepthEstimator::DepthEstimator(float q, float r)
    : q_(q), r_(r), xEst_(0), pEst_(1000.0f) {}

void DepthEstimator::reset() {
    xEst_  = 0;
    pEst_  = 1000.0f;
    head_  = 0;
    count_ = 0;
    lastRaw_ = 0;
}

float DepthEstimator::ringMedian() {
    if (count_ == 0) return 0.0f;
    std::array<float, RING> tmp;
    std::copy_n(ring_.begin(), count_, tmp.begin());
    std::sort(tmp.begin(), tmp.begin() + count_);
    int mid = count_ / 2;
    return (count_ % 2 == 0) ? 0.5f * (tmp[mid - 1] + tmp[mid]) : tmp[mid];
}

float DepthEstimator::update(float rawMm) {
    if (!std::isfinite(rawMm) || rawMm <= 0 || rawMm > 8000.0f)
        return xEst_ * scale;

    // Large jump → reset
    if (xEst_ > 0 && std::fabs(rawMm - xEst_) / xEst_ > 0.35f)
        reset();

    // Ring buffer
    ring_[head_] = rawMm;
    head_        = (head_ + 1) % RING;
    if (count_ < RING) ++count_;
    lastRaw_     = rawMm;

    float med = ringMedian();

    // Kalman predict
    pEst_ += q_;

    // Kalman update
    float kg  = pEst_ / (pEst_ + r_);
    xEst_     = xEst_ + kg * (med - xEst_);
    pEst_     = (1.0f - kg) * pEst_;

    return xEst_ * scale;
}

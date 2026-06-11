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
    // Media del rango intercuartílico (Q1..Q3) — rechaza outliers sin bloquear movimiento
    int q1 = count_ / 4;
    int q3 = count_ * 3 / 4;
    float sum = 0.0f;
    for (int i = q1; i <= q3; ++i) sum += tmp[i];
    return sum / (q3 - q1 + 1);
}

float DepthEstimator::update(float rawMm) {
    // Rechazar medidas fuera del rango útil (0–350 cm)
    if (!std::isfinite(rawMm) || rawMm <= 0 || rawMm > 3500.0f)
        return xEst_ * scale;

    // Reset duro solo si la divergencia es extrema (> 200 cm)
    // — sin gate multiplicativo para no bloquear movimientos rápidos
    if (xEst_ > 0 && std::fabs(rawMm - xEst_) > 2000.0f)
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

#include "Stabilizer.hpp"

Stabilizer::Stabilizer(float processNoise, float measureNoise)
    : x_(0.0f), P_(1000.0f), Q_(processNoise), R_(measureNoise),
      initialized_(false) {}

float Stabilizer::update(float measurementMm) {
    if (!initialized_) {
        x_ = measurementMm;
        P_ = R_;
        initialized_ = true;
        return x_;
    }

    // Predict (constant-position model: no motion assumed between frames)
    float P_pred = P_ + Q_;

    // Update
    float K = P_pred / (P_pred + R_);
    x_ = x_ + K * (measurementMm - x_);
    P_ = (1.0f - K) * P_pred;

    return x_;
}

void Stabilizer::reset() {
    initialized_ = false;
    P_ = 1000.0f;
}

#pragma once

// 1D Kalman filter for depth (Z) stabilisation.
// Models depth as a slowly-changing position with noisy measurements.
class Stabilizer {
public:
    // processNoise : Q — how fast depth can change (mm²). Low = trust model.
    // measureNoise : R — measurement noisiness (mm²). High = trust model more.
    // Q=10, R=500 → converges in ~2-3 seconds, much smoother for noisy calibration
    explicit Stabilizer(float processNoise = 10.0f, float measureNoise = 500.0f);

    // Feed a new raw measurement (mm). Returns the filtered estimate (mm).
    float update(float measurementMm);

    // Reset the filter (e.g. when the scene changes completely).
    void  reset();

    bool  isInitialized() const { return initialized_; }

private:
    float x_;            // state estimate (depth mm)
    float P_;            // error covariance
    float Q_;            // process noise variance
    float R_;            // measurement noise variance
    bool  initialized_;
};

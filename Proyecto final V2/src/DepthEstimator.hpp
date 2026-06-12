#pragma once
#include <array>
#include <cmath>
#include <limits>

// 1-D Kalman filter + ring-buffer moving average for depth stabilization
class DepthEstimator {
public:
    static constexpr int RING = 24;

    explicit DepthEstimator(float processNoise = 10.0f,
                             float measureNoise = 500.0f);

    // Feed a new raw measurement (mm). Returns filtered estimate (mm).
    float update(float rawMm);

    // Force-reset (e.g. on large jumps)
    void reset();

    float lastRaw()      const { return lastRaw_; }
    float lastFiltered() const { return xEst_; }

    // Scale factor for manual calibration (multiplied on output)
    float scale = 1.0f;

private:
    float q_, r_;           // process / measurement noise
    float xEst_, pEst_;    // Kalman state

    std::array<float, RING> ring_{};
    int   head_  = 0;
    int   count_ = 0;
    float lastRaw_ = 0;

    float ringMedian();
};

#pragma once
#include "HandTracker.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

// Direct C++ port of prueba_manos_f3.py
// Z depth (cm) controls scale and rotation speed, clamped to [Z_MIN, Z_MAX]
namespace ARFilter {

static constexpr float Z_MIN_CM = 30.0f;   // closer than this = maximum effect
static constexpr float Z_MAX_CM = 130.0f;  // farther than this = minimum effect

// Normalizes Z to [0,1]: Z=Z_MIN→1.0 (close/big), Z=Z_MAX→0.0 (far/small)
float zToIntensity(float zCm);

// Returns true if hand has >= 4 fingers extended
bool isHandOpen(const HandTracker::Hand& h, bool isRight);

// Draws the rotating sigil on a single open hand
// intensity drives scale and opacity
// angleDeg is the current rotation angle (caller increments per frame)
void drawSigil(cv::Mat& frame, cv::Point center, int radius,
               float intensity, float angleDeg);

// Draws the portal effect between two open hands
void drawPortal(cv::Mat& frame, cv::Point center, int radius,
                float intensity, float angleDeg);

// Full frame render: detects open hands, draws effects, returns updated angleDeg
float render(cv::Mat& frame,
             const std::vector<HandTracker::Hand>& hands,
             float angleDeg,
             float zCm);   // Z from stereo (or 0 = use hand pixel size)

} // namespace ARFilter

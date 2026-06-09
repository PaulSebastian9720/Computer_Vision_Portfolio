#include "ARFilter.hpp"
#include <chrono>
#include <cmath>
#include <algorithm>

static double nowSec() {
    using namespace std::chrono;
    return duration_cast<microseconds>(
               steady_clock::now().time_since_epoch()).count() * 1e-6;
}

static float normalize(float v, float lo, float hi) {
    if (hi <= lo) return 0.0f;
    return std::max(0.0f, std::min(1.0f, (v - lo) / (hi - lo)));
}

namespace ARFilter {

// ─── Z → intensity ───────────────────────────────────────────────────────────
float zToIntensity(float zCm) {
    // Clamp to [Z_MIN, Z_MAX]
    float z = std::max(Z_MIN_CM, std::min(Z_MAX_CM, zCm));
    // Z=30cm → intensity=1.0 (close, big, fast)
    // Z=130cm → intensity=0.0 (far, small, slow)
    float i = 1.0f - normalize(z, Z_MIN_CM, Z_MAX_CM);
    return std::max(0.50f, i);  // minimum 0.5 like Python version
}

// ─── Finger check ────────────────────────────────────────────────────────────
bool isHandOpen(const HandTracker::Hand& h, bool isRight) {
    // Fingers: tip/pip pairs
    const int tips[] = {8, 12, 16, 20};
    const int pips[] = {6, 10, 14, 18};
    int extended = 0;

    for (int i = 0; i < 4; ++i)
        if (h.lm[tips[i]].y < h.lm[pips[i]].y)
            ++extended;

    // Thumb (uses X axis)
    if (isRight) {
        if (h.lm[4].x < h.lm[3].x) ++extended;
    } else {
        if (h.lm[4].x > h.lm[3].x) ++extended;
    }
    return extended >= 4;
}

// ─── Sigil ────────────────────────────────────────────────────────────────────
void drawSigil(cv::Mat& frame, cv::Point center, int radius,
               float intensity, float angleDeg) {
    double t   = nowSec();
    int cx = center.x, cy = center.y;

    float alpha   = 0.22f + intensity * 0.58f;
    int thickness = (int)(2 + intensity * 4);
    int bright    = (int)(150 + intensity * 105);

    cv::Scalar base(0, bright, 255);
    cv::Scalar sec(0, (int)(bright * 0.70), 255);

    cv::Mat overlay = frame.clone();

    // Concentric circles
    cv::circle(overlay, center, radius,               base, thickness);
    cv::circle(overlay, center, (int)(radius * 0.72), sec,  std::max(1, thickness - 1));
    cv::circle(overlay, center, (int)(radius * 0.35), base, std::max(1, thickness - 1));

    // 12 radial lines
    for (int i = 0; i < 12; ++i) {
        double theta = (angleDeg + i * 30.0) * CV_PI / 180.0;
        cv::Point p1{(int)(cx + std::cos(theta) * radius * 0.38),
                     (int)(cy + std::sin(theta) * radius * 0.38)};
        cv::Point p2{(int)(cx + std::cos(theta) * radius * 0.95),
                     (int)(cy + std::sin(theta) * radius * 0.95)};
        cv::line(overlay, p1, p2, base, 1);
    }

    // Triangle
    std::vector<cv::Point> tri;
    for (int i = 0; i < 3; ++i) {
        double theta = (angleDeg + 90.0 + i * 120.0) * CV_PI / 180.0;
        tri.push_back({(int)(cx + std::cos(theta) * radius * 0.62),
                       (int)(cy + std::sin(theta) * radius * 0.62)});
    }
    cv::polylines(overlay, tri, true, sec, thickness);

    // Hexagon
    std::vector<cv::Point> hex;
    for (int i = 0; i < 6; ++i) {
        double theta = (-angleDeg * 0.8 + i * 60.0) * CV_PI / 180.0;
        hex.push_back({(int)(cx + std::cos(theta) * radius * 0.48),
                       (int)(cy + std::sin(theta) * radius * 0.48)});
    }
    cv::polylines(overlay, hex, true, base, 1);

    // Orbital particles
    int nPart = (int)(20 + intensity * 50);
    for (int i = 0; i < nPart; ++i) {
        double theta = (angleDeg * 2.0 + i * 360.0 / nPart) * CV_PI / 180.0;
        double wave  = std::sin(t * 3.0 + i) * 8.0;
        double dist  = radius + 8.0 + wave;
        cv::Point p{(int)(cx + std::cos(theta) * dist),
                    (int)(cy + std::sin(theta) * dist)};
        cv::circle(overlay, p, 2, base, -1);
    }

    // Center glow dot
    cv::circle(overlay, center, (int)(6 + intensity * 8),
               cv::Scalar(0, 255, 255), -1);

    cv::addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame);

    // Outer glow ring
    cv::Mat glow = frame.clone();
    cv::circle(glow, center, (int)(radius * 1.08), base, 2);
    float ga = 0.15f + intensity * 0.20f;
    cv::addWeighted(glow, ga, frame, 1.0 - ga, 0, frame);
}

// ─── Portal ───────────────────────────────────────────────────────────────────
void drawPortal(cv::Mat& frame, cv::Point center, int radius,
                float intensity, float angleDeg) {
    double t   = nowSec();
    int cx = center.x, cy = center.y;
    radius = std::max(80, std::min(radius, 260));

    float  alpha  = 0.28f + intensity * 0.45f;
    int    bright = (int)(170 + intensity * 85);

    cv::Scalar base   (0, bright, 255);
    cv::Scalar intense(0, 255, 255);
    cv::Scalar shadow (0, (int)(bright * 0.45), 180);

    cv::Mat overlay = frame.clone();

    // Main rings
    float scales[] = {1.0f, 0.82f, 0.62f};
    for (int i = 0; i < 3; ++i)
        cv::circle(overlay, center, (int)(radius * scales[i]),
                   base, std::max(1, 5 - i));

    // Spiral
    const int steps = 120;
    std::vector<cv::Point> spiral;
    spiral.reserve(steps);
    for (int i = 0; i < steps; ++i) {
        float t2 = (float)i / steps;
        double theta = angleDeg * 2.0 * CV_PI / 180.0
                       + 3.2 * 2.0 * CV_PI * t2;
        float r = radius * (0.10f + 0.72f * t2);
        spiral.push_back({(int)(cx + std::cos(theta) * r),
                           (int)(cy + std::sin(theta) * r)});
    }
    for (int i = 1; i < (int)spiral.size(); ++i)
        cv::line(overlay, spiral[i-1], spiral[i], shadow, 2);

    // Radial rays
    for (int i = 0; i < 18; ++i) {
        double theta = (-angleDeg + i * 20.0) * CV_PI / 180.0;
        float  r1 = radius * 0.82f;
        float  r2 = radius * (1.05f + 0.08f * (float)std::sin(t * 4.0 + i));
        cv::Point p1{(int)(cx + std::cos(theta) * r1),
                     (int)(cy + std::sin(theta) * r1)};
        cv::Point p2{(int)(cx + std::cos(theta) * r2),
                     (int)(cy + std::sin(theta) * r2)};
        cv::line(overlay, p1, p2, base, 2);
    }

    // Orbital particles
    int nPart = (int)(35 + intensity * 60);
    for (int i = 0; i < nPart; ++i) {
        double theta = (angleDeg * 3.0 + i * 360.0 / nPart) * CV_PI / 180.0;
        double wave  = std::sin(t * 5.0 + i * 0.7) * 12.0;
        float  dist  = radius * 1.05f + (float)wave;
        cv::Point p{(int)(cx + std::cos(theta) * dist),
                    (int)(cy + std::sin(theta) * dist)};
        cv::circle(overlay, p, 2, intense, -1);
    }

    // Inner nucleus
    cv::circle(overlay, center, (int)(radius * 0.28), shadow, -1);

    cv::addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame);

    // Outer glow
    cv::Mat glow = frame.clone();
    cv::circle(glow, center, (int)(radius * 1.12), base, 4);
    float ga = 0.20f + intensity * 0.18f;
    cv::addWeighted(glow, ga, frame, 1.0 - ga, 0, frame);
}

// ─── Full render ──────────────────────────────────────────────────────────────
float render(cv::Mat& frame,
             const std::vector<HandTracker::Hand>& hands,
             float angleDeg, float zCm) {

    // Build per-hand info
    struct Info {
        cv::Point center;
        int       radius;
        float     intensity;
        bool      open;
        bool      isRight;
    };
    std::vector<Info> infos;

    for (const auto& h : hands) {
        if (h.conf <= 0) continue;

        // Intensity: prefer stereo Z if available, fall back to pixel size
        float intensity;
        if (zCm > 1.0f) {
            intensity = zToIntensity(zCm);
        } else {
            // Pixel-based proxy: big hand = close = high intensity
            float cercania = normalize(h.radiusPx, 35.0f, 150.0f);
            intensity = std::max(0.50f, 1.0f - cercania);
        }

        Info inf;
        inf.center    = h.palmCenter;
        inf.isRight   = h.isRight;
        inf.open      = isHandOpen(h, h.isRight);
        inf.intensity = intensity;
        // Effective radius: expand when hand is open and closer
        inf.radius = (inf.open)
            ? std::max(75, (int)(h.radiusPx * (1.15f + intensity * 0.50f)))
            : (int)h.radiusPx;

        infos.push_back(inf);
    }

    // Draw skeleton landmarks
    for (const auto& h : hands) {
        // Draw all 21 landmarks as dots
        for (int i = 0; i < 21; ++i) {
            cv::circle(frame,
                       {(int)h.lm[i].x, (int)h.lm[i].y},
                       3, cv::Scalar(0, 200, 200), -1);
        }
        // Palm center dot
        cv::circle(frame, h.palmCenter, 5, cv::Scalar(0, 255, 255), -1);
    }

    // Sigilos individuales
    for (const auto& inf : infos) {
        if (inf.open) {
            drawSigil(frame, inf.center, inf.radius, inf.intensity, angleDeg);
            cv::putText(frame, "SIGILO ACTIVADO",
                        {inf.center.x - 70, inf.center.y + inf.radius + 22},
                        cv::FONT_HERSHEY_SIMPLEX, 0.55,
                        cv::Scalar(0, 255, 255), 1);
        }
    }

    // Portal entre dos manos abiertas
    std::vector<const Info*> open_hands;
    for (const auto& inf : infos)
        if (inf.open) open_hands.push_back(&inf);

    if (open_hands.size() >= 2) {
        const Info& m1 = *open_hands[0];
        const Info& m2 = *open_hands[1];

        float dx = (float)(m2.center.x - m1.center.x);
        float dy = (float)(m2.center.y - m1.center.y);
        float sep = std::sqrt(dx*dx + dy*dy);

        cv::Point portalCenter{(int)((m1.center.x + m2.center.x) / 2),
                                (int)((m1.center.y + m2.center.y) / 2)};
        int portalR = (int)(sep * 0.45f);

        float portalI = std::max(0.55f, normalize(sep, 180.0f, 600.0f));

        // Connection line
        cv::line(frame, m1.center, m2.center, cv::Scalar(0, 200, 255), 2);

        drawPortal(frame, portalCenter, portalR, portalI, -angleDeg);
    }

    // Z info text
    if (zCm > 1.0f) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "Z=%.0f cm", zCm);
        cv::putText(frame, buf, {10, frame.rows - 12},
                    cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 0, 0), 3);
        cv::putText(frame, buf, {10, frame.rows - 12},
                    cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 0), 1);
    }

    // Advance angle: faster when closer (higher intensity)
    float avgIntensity = 0.5f;
    if (!infos.empty()) {
        for (const auto& inf : infos) avgIntensity += inf.intensity;
        avgIntensity /= infos.size();
    }
    angleDeg += 2.5f + avgIntensity * 5.0f;
    if (angleDeg >= 360.0f) angleDeg -= 360.0f;

    return angleDeg;
}

} // namespace ARFilter

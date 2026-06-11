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
bool isHandOpen(const HandTracker::Hand& /*h*/, bool /*isRight*/) {
    return true;
}

// ─── Sigil ────────────────────────────────────────────────────────────────────
// Dibujado completamente con manipulación de píxeles:
// cada forma se calcula por distancia/ángulo por píxel y el blend
// se hace en un loop explícito de compositing alfa.
void drawSigil(cv::Mat& frame, cv::Point center, int radius,
               float intensity, float angleDeg) {
    double t  = nowSec();
    int cx = center.x, cy = center.y;

    float alpha  = 0.22f + intensity * 0.58f;
    int   bright = (int)(150 + intensity * 105);
    float ringW  = std::max(2.0f, 2.0f + intensity * 4.0f);

    // Capa de efecto vacía — se dibuja con acceso directo a píxeles
    cv::Mat layer(frame.size(), CV_8UC3, cv::Scalar(0, 0, 0));

    // Precomputar vértices de triángulo y hexágono
    float triA = (angleDeg + 90.0f) * (float)(CV_PI / 180.0);
    float hexA = -angleDeg * 0.8f   * (float)(CV_PI / 180.0);
    float triV[3][2], hexV[6][2];
    for (int i = 0; i < 3; ++i) {
        float a = triA + i * (float)(CV_PI * 2.0 / 3.0);
        triV[i][0] = std::cos(a) * radius * 0.62f;
        triV[i][1] = std::sin(a) * radius * 0.62f;
    }
    for (int i = 0; i < 6; ++i) {
        float a = hexA + i * (float)(CV_PI / 3.0);
        hexV[i][0] = std::cos(a) * radius * 0.48f;
        hexV[i][1] = std::sin(a) * radius * 0.48f;
    }

    // Colores exactos del original (BGR): base=(0,bright,255), sec=(0,bright*0.7,255)
    // El grosor equivale a thickness/2 a cada lado del radio (igual que cv::circle)
    float hw      = ringW * 0.5f;   // half-width, igual que thickness/2 en cv::circle
    float radii[3]= { (float)radius, radius * 0.72f, radius * 0.35f };
    float glowR   = 6.0f + intensity * 8.0f;

    // Precomputar endpoints de las 12 líneas radiales (igual que el original)
    float lineP1[12][2], lineP2[12][2];
    for (int i = 0; i < 12; ++i) {
        double th = (angleDeg + i * 30.0) * CV_PI / 180.0;
        lineP1[i][0] = (float)std::cos(th) * radius * 0.38f;
        lineP1[i][1] = (float)std::sin(th) * radius * 0.38f;
        lineP2[i][0] = (float)std::cos(th) * radius * 0.95f;
        lineP2[i][1] = (float)std::sin(th) * radius * 0.95f;
    }

    // ROI
    int x0 = std::max(0, cx - radius - 14);
    int x1 = std::min(frame.cols - 1, cx + radius + 14);
    int y0 = std::max(0, cy - radius - 14);
    int y1 = std::min(frame.rows - 1, cy + radius + 14);

    // ── Loop principal de píxeles ────────────────────────────────────────────
    for (int y = y0; y <= y1; ++y) {
        cv::Vec3b* row = layer.ptr<cv::Vec3b>(y);
        for (int x = x0; x <= x1; ++x) {
            float dx = (float)(x - cx);
            float dy = (float)(y - cy);
            float d  = std::sqrt(dx * dx + dy * dy);

            // 1. Círculos concéntricos
            //    Mismo criterio que cv::circle con thickness: |d - r| <= hw
            //    Falloff coseno para suavizar bordes como el AA de OpenCV
            for (int ri = 0; ri < 3; ++ri) {
                float diff = std::abs(d - radii[ri]);
                if (diff <= hw + 1.0f) {
                    float fade = (diff <= hw)
                        ? 1.0f
                        : 1.0f - (diff - hw);          // 1px de anti-alias
                    // base para radio 0 y 2, sec para radio 1
                    float gv = (ri == 1) ? bright * 0.70f : (float)bright;
                    row[x][0] = 0;
                    row[x][1] = cv::saturate_cast<uchar>(
                        std::max((float)row[x][1], gv * fade));
                    row[x][2] = cv::saturate_cast<uchar>(
                        std::max((float)row[x][2], 255.0f * fade));
                }
            }

            // 2. Líneas radiales — distancia de píxel al segmento (igual que cv::line)
            for (int i = 0; i < 12; ++i) {
                float ax = lineP1[i][0], ay = lineP1[i][1];
                float bx = lineP2[i][0], by = lineP2[i][1];
                float lx = bx - ax, ly = by - ay;
                float len2 = lx*lx + ly*ly;
                float tc = (len2 > 0) ? ((dx-ax)*lx + (dy-ay)*ly) / len2 : 0.0f;
                tc = std::max(0.0f, std::min(1.0f, tc));
                float ex = ax + tc*lx, ey = ay + tc*ly;
                float dist = std::sqrt((dx-ex)*(dx-ex) + (dy-ey)*(dy-ey));
                if (dist < 1.5f) {
                    float fade = 1.0f - dist / 1.5f;
                    row[x][0] = 0;
                    row[x][1] = cv::saturate_cast<uchar>(
                        std::max((float)row[x][1], bright * 0.6f * fade));
                    row[x][2] = cv::saturate_cast<uchar>(
                        std::max((float)row[x][2], 255.0f * fade));
                }
            }

            // 3. Triángulo — distancia de píxel a cada segmento (cv::polylines)
            for (int i = 0; i < 3; ++i) {
                int   j  = (i + 1) % 3;
                float ax = triV[i][0], ay = triV[i][1];
                float bx = triV[j][0], by = triV[j][1];
                float lx = bx - ax, ly = by - ay;
                float len2 = lx*lx + ly*ly;
                float tc = (len2 > 0) ? ((dx-ax)*lx + (dy-ay)*ly) / len2 : 0.0f;
                tc = std::max(0.0f, std::min(1.0f, tc));
                float ex = ax + tc*lx, ey = ay + tc*ly;
                float dist = std::sqrt((dx-ex)*(dx-ex) + (dy-ey)*(dy-ey));
                if (dist < hw + 1.0f) {
                    float fade = (dist <= hw) ? 1.0f : 1.0f - (dist - hw);
                    row[x][0] = 0;
                    row[x][1] = cv::saturate_cast<uchar>(
                        std::max((float)row[x][1], bright * 0.70f * fade));
                    row[x][2] = cv::saturate_cast<uchar>(
                        std::max((float)row[x][2], 255.0f * fade));
                }
            }

            // 4. Hexágono — distancia de píxel a cada segmento (cv::polylines, grosor 1)
            for (int i = 0; i < 6; ++i) {
                int   j  = (i + 1) % 6;
                float ax = hexV[i][0], ay = hexV[i][1];
                float bx = hexV[j][0], by = hexV[j][1];
                float lx = bx - ax, ly = by - ay;
                float len2 = lx*lx + ly*ly;
                float tc = (len2 > 0) ? ((dx-ax)*lx + (dy-ay)*ly) / len2 : 0.0f;
                tc = std::max(0.0f, std::min(1.0f, tc));
                float ex = ax + tc*lx, ey = ay + tc*ly;
                float dist = std::sqrt((dx-ex)*(dx-ex) + (dy-ey)*(dy-ey));
                if (dist < 1.5f) {
                    float fade = 1.0f - dist / 1.5f;
                    row[x][0] = 0;
                    row[x][1] = cv::saturate_cast<uchar>(
                        std::max((float)row[x][1], (float)bright * fade));
                    row[x][2] = cv::saturate_cast<uchar>(
                        std::max((float)row[x][2], 255.0f * fade));
                }
            }

            // 5. Glow central — gradiente radial (reemplaza cv::circle relleno)
            if (d < glowR) {
                float g = 1.0f - (d / glowR);
                row[x][0] = cv::saturate_cast<uchar>(row[x][0] + (uchar)(255 * g));
                row[x][1] = cv::saturate_cast<uchar>(row[x][1] + (uchar)(255 * g));
                row[x][2] = cv::saturate_cast<uchar>(row[x][2] + (uchar)(255 * g));
            }
        }
    }

    // 6. Partículas orbitales — asignación directa de píxel (valor RGB)
    int nPart = (int)(20 + intensity * 50);
    for (int i = 0; i < nPart; ++i) {
        double theta = (angleDeg * 2.0 + i * 360.0 / nPart) * CV_PI / 180.0;
        double wave  = std::sin(t * 3.0 + i) * 8.0;
        int    px    = (int)(cx + std::cos(theta) * (radius + 8.0 + wave));
        int    py    = (int)(cy + std::sin(theta) * (radius + 8.0 + wave));
        for (int oy = -2; oy <= 2; ++oy) {
            int ny = py + oy;
            if (ny < 0 || ny >= layer.rows) continue;
            cv::Vec3b* row = layer.ptr<cv::Vec3b>(ny);
            for (int ox = -2; ox <= 2; ++ox) {
                int nx = px + ox;
                if (nx >= 0 && nx < layer.cols)
                    row[nx] = cv::Vec3b(0, (uchar)bright, 255);
            }
        }
    }

    // 7. Compositing alfa píxel a píxel: dst = α·src + (1-α)·dst
    for (int y = y0; y <= y1; ++y) {
        const cv::Vec3b* src = layer.ptr<cv::Vec3b>(y);
        cv::Vec3b*       dst = frame.ptr<cv::Vec3b>(y);
        for (int x = x0; x <= x1; ++x) {
            if (src[x][0] == 0 && src[x][1] == 0 && src[x][2] == 0) continue;
            for (int c = 0; c < 3; ++c)
                dst[x][c] = cv::saturate_cast<uchar>(
                    alpha * src[x][c] + (1.0f - alpha) * dst[x][c]);
        }
    }

    // 8. Anillo de glow exterior — gradiente radial sobre el frame
    float outerR = radius * 1.08f, outerW = radius * 0.12f;
    float gaMax  = 0.15f + intensity * 0.20f;
    int   gr0 = std::max(0, cy - (int)(radius * 1.22f));
    int   gr1 = std::min(frame.rows - 1, cy + (int)(radius * 1.22f));
    int   gc0 = std::max(0, cx - (int)(radius * 1.22f));
    int   gc1 = std::min(frame.cols - 1, cx + (int)(radius * 1.22f));
    for (int y = gr0; y <= gr1; ++y) {
        cv::Vec3b* dst = frame.ptr<cv::Vec3b>(y);
        for (int x = gc0; x <= gc1; ++x) {
            float d    = std::sqrt((float)((x-cx)*(x-cx) + (y-cy)*(y-cy)));
            float diff = std::abs(d - outerR);
            if (diff < outerW) {
                float ga = gaMax * (1.0f - diff / outerW);
                dst[x][1] = cv::saturate_cast<uchar>(dst[x][1] + (uchar)(bright * ga));
                dst[x][2] = cv::saturate_cast<uchar>(dst[x][2] + (uchar)(255  * ga));
            }
        }
    }
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

    // Sigilos individuales
    for (const auto& inf : infos) {
        if (inf.open)
            drawSigil(frame, inf.center, inf.radius, inf.intensity, angleDeg);
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

#include <jni.h>
#include <string>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <android/log.h>
#include <opencv2/opencv.hpp>
#include "training_data.h"

#define LOG_TAG "ShapeSignature"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static const int N_POINTS   = 256;  // puntos de remuestreo del contorno
static const int N_DESC     = 12;   // componentes del descriptor Fourier

// Remuestrea contorno a N puntos equiespaciados por longitud de arco
static std::vector<cv::Point2f> resampleContour(
        const std::vector<cv::Point>& contour, int N) {

    int M = (int)contour.size();
    if (M < 3) return {};

    // Longitudes de arco acumuladas (incluyendo cierre)
    std::vector<float> arc(M + 1, 0.f);
    for (int i = 0; i < M; i++) {
        float dx = contour[(i + 1) % M].x - contour[i].x;
        float dy = contour[(i + 1) % M].y - contour[i].y;
        arc[i + 1] = arc[i] + std::sqrt(dx * dx + dy * dy);
    }
    float totalLen = arc[M];
    if (totalLen < 1.f) return {};

    std::vector<cv::Point2f> result(N);
    int j = 0;
    for (int i = 0; i < N; i++) {
        float target = (float)i / N * totalLen;
        while (j < M - 1 && arc[j + 1] < target) j++;
        float span = arc[j + 1] - arc[j];
        float t = (span > 1e-6f) ? (target - arc[j]) / span : 0.f;
        result[i].x = contour[j].x + t * (contour[(j + 1) % M].x - contour[j].x);
        result[i].y = contour[j].y + t * (contour[(j + 1) % M].y - contour[j].y);
    }
    return result;
}

// Pipeline completo: RGBA → descriptor de Fourier de coordenadas complejas
// Devuelve vector vacío si no se detecta forma válida
static std::vector<float> extractDescriptor(const cv::Mat& rgba) {

    // 1. Convertir a escala de grises
    cv::Mat gray;
    cv::cvtColor(rgba, gray, cv::COLOR_RGBA2GRAY);

    // 2. Filtro Gaussiano para reducción de ruido
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);

    // 3. Umbral Otsu — óptimo para fondos blancos uniformes (Flavia Dataset)
    //    La guía sugiere umbral adaptativo para condiciones variables de luz;
    //    aquí se usa Otsu porque el pipeline de entrenamiento usa el mismo método.
    cv::Mat binary;
    cv::threshold(blurred, binary, 0, 255,
                  cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    // 4. Operaciones morfológicas para limpiar ruido residual
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN,  kernel);

    // 5. Detectar contornos externos (cv::findContours)
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    if (contours.empty()) return {};

    // Seleccionar contorno de mayor área
    int idx = 0;
    double maxArea = 0.0;
    for (int i = 0; i < (int)contours.size(); i++) {
        double a = cv::contourArea(contours[i]);
        if (a > maxArea) { maxArea = a; idx = i; }
    }
    if (maxArea < 200.0) {
        LOGD("Contorno demasiado pequeño: %.1f px²", maxArea);
        return {};
    }

    const auto& contour = contours[idx];

    // 6. Centroide mediante momentos
    cv::Moments mom = cv::moments(contour);
    if (std::abs(mom.m00) < 1.0) return {};
    float xc = (float)(mom.m10 / mom.m00);
    float yc = (float)(mom.m01 / mom.m00);

    // 7. Remuestrear a N_POINTS puntos equiespaciados
    auto pts = resampleContour(contour, N_POINTS);
    if (pts.empty()) return {};

    // 8. Señal compleja s(n) = (x(n)−xc) + j·(y(n)−yc)
    //    Se almacena como Mat de 2 canales: canal 0 = parte real, canal 1 = parte imag.
    cv::Mat input(1, N_POINTS, CV_32FC2);
    for (int i = 0; i < N_POINTS; i++) {
        input.at<cv::Vec2f>(0, i) = {pts[i].x - xc, pts[i].y - yc};
    }

    // 9. DFT de la señal compleja (cv::DFT_COMPLEX_OUTPUT)
    cv::Mat complexOut;
    cv::dft(input, complexOut, cv::DFT_COMPLEX_OUTPUT);

    // 10. Normalización por |F(1)| → invarianza a escala y descarte de fase
    cv::Vec2f F1 = complexOut.at<cv::Vec2f>(0, 1);
    float f1Mag = std::sqrt(F1[0] * F1[0] + F1[1] * F1[1]);
    if (f1Mag < 1e-6f) {
        LOGE("F(1) ≈ 0, forma inválida");
        return {};
    }

    // Descriptor = |F(1)|/|F(1)|, |F(2)|/|F(1)|, ..., |F(12)|/|F(1)|
    std::vector<float> desc(N_DESC);
    for (int k = 1; k <= N_DESC; k++) {
        cv::Vec2f Fk = complexOut.at<cv::Vec2f>(0, k);
        float mag = std::sqrt(Fk[0] * Fk[0] + Fk[1] * Fk[1]);
        desc[k - 1] = mag / f1Mag;
    }

    return desc;
}

// Clasificación KNN-1 por distancia Euclídea contra conjunto de entrenamiento
static int classify(const std::vector<float>& desc) {
    float minDist = std::numeric_limits<float>::max();
    int bestLabel = -1;

    for (int i = 0; i < NUM_TRAIN; i++) {
        float dist = 0.f;
        for (int j = 0; j < N_DESC; j++) {
            float diff = desc[j] - TRAIN_DESCRIPTORS[i][j];
            dist += diff * diff;
        }
        if (dist < minDist) {
            minDist = dist;
            bestLabel = TRAIN_LABELS[i];
        }
    }
    return bestLabel;
}

// ─── JNI ENTRY POINT ────────────────────────────────────────────────────────
// Recibe puntero nativo del Mat RGBA del frame actual.
// Devuelve String con formato:  "label;d1;d2;...;d12"
// Si no hay forma detectada:    "-1;0;0;...;0"
extern "C" JNIEXPORT jstring JNICALL
Java_com_example_shapesignature_MainActivity_processFrame(
        JNIEnv* env, jobject /* thiz */, jlong rgbaAddr) {

    cv::Mat& rgba = *reinterpret_cast<cv::Mat*>(rgbaAddr);

    std::string result;
    auto desc = extractDescriptor(rgba);

    if (desc.empty()) {
        result = "-1";
        for (int i = 0; i < N_DESC; i++) result += ";0.0000";
    } else {
        int label = classify(desc);
        result = std::to_string(label);
        char buf[32];
        for (float v : desc) {
            snprintf(buf, sizeof(buf), ";%.4f", v);
            result += buf;
        }
        LOGD("Clase: %d  |  F1..F4: %.3f %.3f %.3f %.3f",
             label, desc[0], desc[1], desc[2], desc[3]);
    }

    return env->NewStringUTF(result.c_str());
}

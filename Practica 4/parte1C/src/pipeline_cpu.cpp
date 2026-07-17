// Practica 4 - Parte 1C — Pipeline CPU (Tabla 1 de la guia).
//
// Suavizado (Gaussiano) -> Erosion -> Dilatacion -> Canny + Ecualizacion, todo con cv::Mat.
// No requiere CUDA: compila con cualquier OpenCV, incluido el de este equipo.
//
// Fuente: video pregrabado (sin camara — el laboratorio no tiene). Se reinicia solo al llegar
// al final. Ver pipeline_gpu.cpp para el equivalente GPU-only (Tabla 2), que es un programa
// aparte para no mezclar ambas mediciones en un mismo bucle.
//
// Uso: ./preprocesamiento_cpu [ruta_video_opcional]
//   Teclas: 'q' salir | 'r' reset de contadores de tiempo

#include <chrono>
#include <deque>
#include <iostream>
#include <numeric>
#include <string>

#include <opencv2/opencv.hpp>

namespace {

const std::string kVideoPathPorDefecto = "../artifacts/video/La Gran Entrevista _ Mickey Mouse.mp4";

constexpr int kGaussianKernel = 5;
constexpr double kGaussianSigma = 1.5;
constexpr int kMorphKernel = 3;
constexpr double kCannyLow = 50.0;
constexpr double kCannyHigh = 150.0;
constexpr size_t kHistorySize = 60;

class PromedioMovil {
public:
    explicit PromedioMovil(size_t n) : max_size_(n) {}

    void agregar(double ms) {
        valores_.push_back(ms);
        if (valores_.size() > max_size_) valores_.pop_front();
    }

    double media() const {
        if (valores_.empty()) return 0.0;
        return std::accumulate(valores_.begin(), valores_.end(), 0.0) / valores_.size();
    }

    void reset() { valores_.clear(); }

private:
    size_t max_size_;
    std::deque<double> valores_;
};

cv::Mat pipelineCPU(const cv::Mat& frame) {
    cv::Mat gris, suavizado, erosionado, dilatado, bordes, ecualizado;

    cv::cvtColor(frame, gris, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gris, suavizado, cv::Size(kGaussianKernel, kGaussianKernel), kGaussianSigma);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kMorphKernel, kMorphKernel));
    cv::erode(suavizado, erosionado, kernel);
    cv::dilate(erosionado, dilatado, kernel);

    cv::Canny(dilatado, bordes, kCannyLow, kCannyHigh);
    cv::equalizeHist(dilatado, ecualizado);

    cv::Mat resultado;
    cv::hconcat(bordes, ecualizado, resultado);
    return resultado;
}

void dibujarHUD(cv::Mat& img, double ms) {
    std::string texto = "CPU | " + std::to_string(ms).substr(0, 5) + " ms/frame ("
                       + std::to_string(static_cast<int>(1000.0 / std::max(ms, 1e-3))) + " FPS)";
    cv::putText(img, texto, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0), 3);
    cv::putText(img, texto, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255), 1);
}

}  // namespace

int main(int argc, char** argv) {
    std::string videoPath = argc > 1 ? argv[1] : kVideoPathPorDefecto;

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "No se pudo abrir el video: " << videoPath << "\n";
        return 1;
    }
    std::cout << "Video: " << videoPath << "\n";
    std::cout << "Presione 'q' para salir, 'r' para reiniciar el promedio movil." << std::endl;

    PromedioMovil prom(kHistorySize);

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);  // video corto: reinicia en loop
            continue;
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        cv::Mat salida = pipelineCPU(frame);
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        prom.agregar(ms);
        dibujarHUD(salida, prom.media());

        cv::imshow("Pipeline CPU (bordes | ecualizado)", salida);

        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q') break;
        if (key == 'r') prom.reset();
    }

    std::cout << "\nResumen (promedio movil final, " << kHistorySize << " frames): "
              << prom.media() << " ms/frame\n";
    return 0;
}

// Practica 4 - Parte 1C — Pipeline GPU-only (Tabla 2 de la guia).
//
// Un solo upload al inicio y un solo download al final; todo lo intermedio se queda en
// cv::cuda::GpuMat (Suavizado Gaussiano -> Erosion -> Dilatacion -> Canny + Ecualizacion).
//
// Requiere OpenCV compilado con WITH_CUDA=ON — no compila con el OpenCV local de este equipo,
// esta pensado para el laboratorio de Computo 8. Ver pipeline_cpu.cpp para el equivalente CPU
// (Tabla 1), que es un programa aparte para no mezclar ambas mediciones en un mismo bucle.
//
// Fuente: video pregrabado (sin camara — el laboratorio no tiene). Se reinicia solo al llegar
// al final.
//
// Uso: ./preprocesamiento_gpu [ruta_video_opcional]
//   Teclas: 'q' salir | 'r' reset de contadores de tiempo

#include <chrono>
#include <deque>
#include <iostream>
#include <numeric>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>

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

cv::Mat pipelineGPU(const cv::Mat& frame,
                     const cv::Ptr<cv::cuda::Filter>& gaussianFilter,
                     const cv::Ptr<cv::cuda::Filter>& erodeFilter,
                     const cv::Ptr<cv::cuda::Filter>& dilateFilter,
                     const cv::Ptr<cv::cuda::CannyEdgeDetector>& cannyDetector) {
    cv::cuda::GpuMat d_frame, d_gris, d_suavizado, d_erosionado, d_dilatado, d_bordes, d_ecualizado;

    d_frame.upload(frame);                                    // CPU -> GPU (unica subida)
    cv::cuda::cvtColor(d_frame, d_gris, cv::COLOR_BGR2GRAY);   // GPU
    gaussianFilter->apply(d_gris, d_suavizado);                // GPU
    erodeFilter->apply(d_suavizado, d_erosionado);             // GPU
    dilateFilter->apply(d_erosionado, d_dilatado);             // GPU
    cannyDetector->detect(d_dilatado, d_bordes);                // GPU
    cv::cuda::equalizeHist(d_dilatado, d_ecualizado);           // GPU

    cv::cuda::GpuMat d_resultado;
    cv::cuda::hconcat(d_bordes, d_ecualizado, d_resultado);

    cv::Mat resultado;
    d_resultado.download(resultado);                            // GPU -> CPU (unica bajada)
    return resultado;
}

void dibujarHUD(cv::Mat& img, double ms) {
    std::string texto = "GPU-only | " + std::to_string(ms).substr(0, 5) + " ms/frame ("
                       + std::to_string(static_cast<int>(1000.0 / std::max(ms, 1e-3))) + " FPS)";
    cv::putText(img, texto, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0), 3);
    cv::putText(img, texto, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255), 1);
}

}  // namespace

int main(int argc, char** argv) {
    std::string videoPath = argc > 1 ? argv[1] : kVideoPathPorDefecto;

    if (cv::cuda::getCudaEnabledDeviceCount() == 0) {
        std::cerr << "No se detecto ninguna GPU CUDA-capable via OpenCV. "
                     "Verificar que la build tenga WITH_CUDA=ON y que el driver este activo.\n";
        return 1;
    }
    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "No se pudo abrir el video: " << videoPath << "\n";
        return 1;
    }
    std::cout << "Video: " << videoPath << "\n";
    std::cout << "Presione 'q' para salir, 'r' para reiniciar el promedio movil.\n";

    // Filtros GPU creados una sola vez fuera del bucle (evita recompilar el filtro cada frame)
    auto gaussianFilter = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1,
                                                           cv::Size(kGaussianKernel, kGaussianKernel),
                                                           kGaussianSigma);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kMorphKernel, kMorphKernel));
    auto erodeFilter = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, CV_8UC1, kernel);
    auto dilateFilter = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, CV_8UC1, kernel);
    auto cannyDetector = cv::cuda::createCannyEdgeDetector(kCannyLow, kCannyHigh);

    PromedioMovil prom(kHistorySize);

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);  // video corto: reinicia en loop
            continue;
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        cv::Mat salida = pipelineGPU(frame, gaussianFilter, erodeFilter, dilateFilter, cannyDetector);
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        prom.agregar(ms);
        dibujarHUD(salida, prom.media());

        cv::imshow("Pipeline GPU-only (bordes | ecualizado)", salida);

        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q') break;
        if (key == 'r') prom.reset();
    }

    std::cout << "\nResumen (promedio movil final, " << kHistorySize << " frames): "
              << prom.media() << " ms/frame\n";
    return 0;
}

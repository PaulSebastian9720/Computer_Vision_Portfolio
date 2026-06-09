#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "Operaciones.hpp"


using namespace cv;
using namespace std;
namespace fs = std::filesystem;



bool guardarImagen(
    const fs::path &ruta,
    const Mat &imagen
)
{

    bool ok =
        imwrite(
            ruta.string(),
            imagen
        );

    cout
        << (ok ? "[OK] " : "[ERROR] ")
        << ruta.string()
        << endl;

    return ok;

}



string formatoNumero(
    double valor,
    int precision = 2
)
{

    stringstream ss;
    ss
        << fixed
        << setprecision(precision)
        << valor;

    return ss.str();

}



string formatoArchivo(
    double valor
)
{

    string texto =
        formatoNumero(
            valor,
            2
        );

    for(char &c : texto)
    {
        if(c == '.')
        {
            c = '_';
        }
    }

    return texto;

}



int kernelDesdeTrackbar(
    int valor
)
{

    return valor * 2 + 1;

}



void dibujarEstado(
    Mat &imagen,
    const string &texto,
    Point posicion = Point(20, 36)
)
{

    putText(
        imagen,
        texto,
        posicion,
        FONT_HERSHEY_SIMPLEX,
        0.78,
        Scalar(0, 0, 0),
        4,
        LINE_AA
    );

    putText(
        imagen,
        texto,
        posicion,
        FONT_HERSHEY_SIMPLEX,
        0.78,
        Scalar(255, 255, 255),
        2,
        LINE_AA
    );

}



Mat componerChromaKey(
    Operaciones &op,
    const Mat &frame,
    const Scalar &hsvMin,
    const Scalar &hsvMax,
    Mat &mascaraFondo,
    Mat &mascaraPrimerPlano
)
{

    Mat fondo =
        op.crearFondoGradiente(
            frame.size()
        );

    Mat chroma =
        op.chromaKeyHSV(
            frame,
            hsvMin,
            hsvMax,
            fondo,
            mascaraFondo
        );

    bitwise_not(
        mascaraFondo,
        mascaraPrimerPlano
    );

    return chroma;

}



void generarVideoKarlStruss(
    Operaciones &op,
    const Mat &imagen,
    const fs::path &rutaSalida
)
{

    VideoWriter video(
        rutaSalida.string(),
        VideoWriter::fourcc('M', 'J', 'P', 'G'),
        20.0,
        imagen.size(),
        true
    );

    if(!video.isOpened())
    {
        cout
            << "[ERROR] No se pudo crear el video: "
            << rutaSalida.string()
            << endl;

        return;
    }

    for(int i = 0; i < 120; i++)
    {
        double alpha =
            0.5 - 0.5 * cos(
                2.0 * CV_PI * static_cast<double>(i) / 119.0
            );

        Mat mezcla =
            op.mezclaCanales(
                imagen,
                alpha
            );

        Mat frame;
        cvtColor(
            mezcla,
            frame,
            COLOR_GRAY2BGR
        );

        dibujarEstado(
            frame,
            "Karl Struss alpha=" + formatoNumero(alpha)
        );

        video.write(
            frame
        );
    }

    video.release();

    cout
        << "[OK] "
        << rutaSalida.string()
        << endl;

}



void generarVideoChroma(
    const vector<Mat> &etapas,
    const vector<string> &nombres,
    const fs::path &rutaSalida
)
{

    if(etapas.empty())
    {
        return;
    }

    VideoWriter video(
        rutaSalida.string(),
        VideoWriter::fourcc('M', 'J', 'P', 'G'),
        10.0,
        etapas[0].size(),
        true
    );

    if(!video.isOpened())
    {
        cout
            << "[ERROR] No se pudo crear el video: "
            << rutaSalida.string()
            << endl;

        return;
    }

    for(size_t etapa = 0; etapa < etapas.size(); etapa++)
    {
        Mat frame;

        if(etapas[etapa].channels() == 1)
        {
            cvtColor(
                etapas[etapa],
                frame,
                COLOR_GRAY2BGR
            );
        }
        else
        {
            frame =
                etapas[etapa].clone();
        }

        dibujarEstado(
            frame,
            nombres[etapa]
        );

        for(int i = 0; i < 24; i++)
        {
            video.write(
                frame
            );
        }
    }

    video.release();

    cout
        << "[OK] "
        << rutaSalida.string()
        << endl;

}



double medirFPS(
    Operaciones &op,
    const Mat &imagen,
    int iteraciones
)
{

    Mat mascaraFondo;
    Mat mascaraPrimerPlano;

    auto inicio =
        chrono::high_resolution_clock::now();

    for(int i = 0; i < iteraciones; i++)
    {
        double alpha =
            static_cast<double>(i % 100) / 99.0;

        Mat mezcla =
            op.mezclaCanales(
                imagen,
                alpha
            );

        Mat chroma =
            componerChromaKey(
                op,
                imagen,
                Scalar(100, 80, 40),
                Scalar(130, 255, 255),
                mascaraFondo,
                mascaraPrimerPlano
            );

        Mat ruido =
            op.ruidoGaussiano(
                chroma,
                22.0,
                mascaraPrimerPlano
            );

        Mat filtrada =
            op.filtroMediana(
                ruido,
                7
            );

        (void) mezcla;
        (void) filtrada;
    }

    auto fin =
        chrono::high_resolution_clock::now();

    chrono::duration<double> duracion =
        fin - inicio;

    return static_cast<double>(iteraciones) / duracion.count();

}



int ejecutarImagen(
    const string &rutaImagen
)
{

    Mat imagen =
        imread(
            rutaImagen,
            IMREAD_COLOR
        );

    if(imagen.empty())
    {
        cout
            << "No se pudo cargar la imagen: "
            << rutaImagen
            << endl;

        return -1;
    }

    fs::path carpetaSalida =
        "resultados";

    fs::create_directories(
        carpetaSalida
    );

    Operaciones op;

    cout
        << "Procesando practica 2 con imagen: "
        << rutaImagen
        << endl;

    Mat gris =
        op.escalaGris(
            imagen
        );

    vector<Mat> canales =
        op.separarCanales(
            imagen
        );

    guardarImagen(carpetaSalida / "01_original.jpg", imagen);
    guardarImagen(carpetaSalida / "02_escala_grises.jpg", gris);
    guardarImagen(carpetaSalida / "03_canal_azul.jpg", canales[0]);
    guardarImagen(carpetaSalida / "04_canal_verde.jpg", canales[1]);
    guardarImagen(carpetaSalida / "05_canal_rojo.jpg", canales[2]);

    vector<pair<string, Mat>> comparativaCanales =
    {
        {"Original", imagen},
        {"Grises", gris},
        {"Canal azul (B)", canales[0]},
        {"Canal verde (G)", canales[1]},
        {"Canal rojo (R)", canales[2]}
    };

    vector<double> alphas =
    {
        0.00,
        0.25,
        0.50,
        0.75,
        1.00
    };

    for(double alpha : alphas)
    {
        Mat mezcla =
            op.mezclaCanales(
                imagen,
                alpha
            );

        guardarImagen(
            carpetaSalida / ("06_mezcla_alpha_" + formatoArchivo(alpha) + ".jpg"),
            mezcla
        );

        comparativaCanales.push_back(
            {
                "Mezcla alpha=" + formatoNumero(alpha),
                mezcla
            }
        );
    }

    guardarImagen(
        carpetaSalida / "07_mosaico_karl_struss.jpg",
        op.crearMosaico(
            comparativaCanales,
            5
        )
    );

    generarVideoKarlStruss(
        op,
        imagen,
        carpetaSalida / "08_video_karl_struss.avi"
    );

    Mat mascaraPrimerPlano;
    Mat escenaAzul =
        op.crearEscenaConFondoColor(
            imagen,
            Scalar(255, 0, 0),
            mascaraPrimerPlano
        );

    Mat fondoNuevo =
        op.crearFondoGradiente(
            imagen.size()
        );

    Mat mascaraFondo;
    Mat chroma =
        op.chromaKeyHSV(
            escenaAzul,
            Scalar(100, 80, 40),
            Scalar(130, 255, 255),
            fondoNuevo,
            mascaraFondo
        );

    Mat ruidoGaussiano =
        op.ruidoGaussiano(
            chroma,
            24.0,
            mascaraPrimerPlano
        );

    Mat ruidoSpeckle =
        op.ruidoSpeckle(
            chroma,
            0.16,
            mascaraPrimerPlano
        );

    Mat filtroGaussiano =
        op.filtroGaussiano(
            ruidoGaussiano,
            7,
            1.2
        );

    Mat filtroMediana =
        op.filtroMediana(
            ruidoGaussiano,
            7
        );

    guardarImagen(carpetaSalida / "09_mascara_primer_plano.jpg", mascaraPrimerPlano);
    guardarImagen(carpetaSalida / "10_escena_fondo_azul.jpg", escenaAzul);
    guardarImagen(carpetaSalida / "11_mascara_chroma_key.jpg", mascaraFondo);
    guardarImagen(carpetaSalida / "12_chroma_key_fondo_reemplazado.jpg", chroma);
    guardarImagen(carpetaSalida / "13_ruido_gaussiano_primer_plano.jpg", ruidoGaussiano);
    guardarImagen(carpetaSalida / "14_ruido_speckle_primer_plano.jpg", ruidoSpeckle);
    guardarImagen(carpetaSalida / "15_filtro_gaussiano.jpg", filtroGaussiano);
    guardarImagen(carpetaSalida / "16_filtro_mediana.jpg", filtroMediana);

    vector<pair<string, Mat>> comparativaChroma =
    {
        {"Original", imagen},
        {"Fondo azul sintetico", escenaAzul},
        {"Mascara primer plano", mascaraPrimerPlano},
        {"Mascara Chroma Key", mascaraFondo},
        {"Fondo reemplazado", chroma},
        {"Ruido gaussiano", ruidoGaussiano},
        {"Ruido speckle", ruidoSpeckle},
        {"Filtro gaussiano", filtroGaussiano},
        {"Filtro mediana", filtroMediana}
    };

    guardarImagen(
        carpetaSalida / "17_mosaico_chroma_ruido_filtros.jpg",
        op.crearMosaico(
            comparativaChroma,
            3
        )
    );

    generarVideoChroma(
        {escenaAzul, chroma, ruidoGaussiano, filtroMediana},
        {
            "Fondo azul sintetico",
            "Chroma Key HSV",
            "Ruido Gaussiano en primer plano",
            "Filtro Mediana k=7"
        },
        carpetaSalida / "18_video_chroma_ruido_filtro.avi"
    );

    double fps =
        medirFPS(
            op,
            imagen,
            240
        );

    ofstream reporte(
        carpetaSalida / "19_reporte_fps.txt"
    );

    reporte
        << "Practica 2 - Vision por Computador" << endl
        << "Modo: imagen fija para evidencias complementarias" << endl
        << "Imagen de entrada: " << rutaImagen << endl
        << "Resolucion: " << imagen.cols << " x " << imagen.rows << endl
        << fixed << setprecision(2)
        << "FPS aproximados del pipeline C++: " << fps << endl;

    reporte.close();

    cout
        << "[OK] "
        << (carpetaSalida / "19_reporte_fps.txt").string()
        << endl;

    return 0;

}



int ejecutarWebcam(
    int indiceCamara
)
{

    fs::path carpetaSalida =
        "resultados_webcam";

    fs::create_directories(
        carpetaSalida
    );

    VideoCapture camara(
        indiceCamara
    );

    if(!camara.isOpened())
    {
        cout
            << "No se pudo abrir la webcam con indice "
            << indiceCamara
            << ". Revise permisos de camara o pruebe otro indice."
            << endl;

        return -1;
    }

    camara.set(
        CAP_PROP_FRAME_WIDTH,
        1280
    );

    camara.set(
        CAP_PROP_FRAME_HEIGHT,
        720
    );

    Operaciones op;

    int alphaBar =
        50;
    int hMin =
        100;
    int hMax =
        130;
    int sMin =
        70;
    int sMax =
        255;
    int vMin =
        40;
    int vMax =
        255;
    int ruidoGauss =
        20;
    int ruidoSpeckle =
        0;
    int kernelBar =
        3;
    int filtro =
        2;

    namedWindow(
        "Controles",
        WINDOW_AUTOSIZE
    );

    namedWindow(
        "Practica 2 - Webcam",
        WINDOW_NORMAL
    );

    createTrackbar("Alpha Karl", "Controles", &alphaBar, 100);
    createTrackbar("H Min", "Controles", &hMin, 179);
    createTrackbar("H Max", "Controles", &hMax, 179);
    createTrackbar("S Min", "Controles", &sMin, 255);
    createTrackbar("S Max", "Controles", &sMax, 255);
    createTrackbar("V Min", "Controles", &vMin, 255);
    createTrackbar("V Max", "Controles", &vMax, 255);
    createTrackbar("Ruido Gauss", "Controles", &ruidoGauss, 80);
    createTrackbar("Ruido Speckle", "Controles", &ruidoSpeckle, 50);
    createTrackbar("Kernel", "Controles", &kernelBar, 18);
    createTrackbar("Filtro 0-1-2", "Controles", &filtro, 2);

    cout
        << endl
        << "Modo webcam activo." << endl
        << "Controles:" << endl
        << "- Alpha Karl: peso del canal rojo en (1-alpha)B + alpha R." << endl
        << "- H/S/V Min-Max: rango HSV del fondo para Chroma Key." << endl
        << "- Ruido Gauss / Speckle: ruido sintetico sobre primer plano." << endl
        << "- Filtro 0-1-2: 0 ninguno, 1 Gaussiano, 2 Mediana." << endl
        << "Teclas: s guardar captura, r iniciar/detener video, q o ESC salir." << endl
        << endl;

    Mat frame;
    Mat ultimoMosaico;
    VideoWriter grabador;
    bool grabando =
        false;
    int contadorCapturas =
        1;
    int contadorVideos =
        1;
    double fps =
        0.0;

    auto tAnterior =
        chrono::high_resolution_clock::now();

    while(true)
    {
        camara >> frame;

        if(frame.empty())
        {
            cout
                << "Frame capturado con valor nulo."
                << endl;

            break;
        }

        flip(
            frame,
            frame,
            1
        );

        auto tActual =
            chrono::high_resolution_clock::now();

        chrono::duration<double> delta =
            tActual - tAnterior;

        tAnterior =
            tActual;

        if(delta.count() > 0.0)
        {
            fps =
                0.90 * fps + 0.10 * (1.0 / delta.count());
        }

        double alpha =
            static_cast<double>(alphaBar) / 100.0;

        vector<Mat> canales =
            op.separarCanales(
                frame
            );

        Mat mezcla =
            op.mezclaCanales(
                frame,
                alpha
            );

        Mat mascaraFondo;
        Mat mascaraPrimerPlano;

        Mat chroma =
            componerChromaKey(
                op,
                frame,
                Scalar(hMin, sMin, vMin),
                Scalar(hMax, sMax, vMax),
                mascaraFondo,
                mascaraPrimerPlano
            );

        Mat procesado =
            chroma.clone();

        if(ruidoGauss > 0)
        {
            procesado =
                op.ruidoGaussiano(
                    procesado,
                    static_cast<double>(ruidoGauss),
                    mascaraPrimerPlano
                );
        }

        if(ruidoSpeckle > 0)
        {
            procesado =
                op.ruidoSpeckle(
                    procesado,
                    static_cast<double>(ruidoSpeckle) / 100.0,
                    mascaraPrimerPlano
                );
        }

        int kernel =
            kernelDesdeTrackbar(
                kernelBar
            );

        Mat filtrado =
            procesado.clone();

        if(filtro == 1)
        {
            filtrado =
                op.filtroGaussiano(
                    procesado,
                    kernel,
                    1.2
                );
        }
        else if(filtro == 2)
        {
            filtrado =
                op.filtroMediana(
                    procesado,
                    kernel
                );
        }

        vector<pair<string, Mat>> vistas =
        {
            {"Original webcam", frame},
            {"Canal azul (B)", canales[0]},
            {"Canal verde (G)", canales[1]},
            {"Canal rojo (R)", canales[2]},
            {"Karl alpha=" + formatoNumero(alpha), mezcla},
            {"Mascara HSV fondo", mascaraFondo},
            {"Chroma Key", chroma},
            {"Ruido primer plano", procesado},
            {"Filtro final", filtrado}
        };

        ultimoMosaico =
            op.crearMosaico(
                vistas,
                3,
                Size(360, 220)
            );

        string estado =
            "FPS=" + formatoNumero(fps) +
            " | k=" + to_string(kernel) +
            " | filtro=" + to_string(filtro) +
            " | grabando=" + string(grabando ? "si" : "no");

        dibujarEstado(
            ultimoMosaico,
            estado,
            Point(18, ultimoMosaico.rows - 18)
        );

        imshow(
            "Practica 2 - Webcam",
            ultimoMosaico
        );

        if(grabando)
        {
            if(!grabador.isOpened())
            {
                fs::path rutaVideo =
                    carpetaSalida / ("webcam_practica2_" + to_string(contadorVideos) + ".avi");

                grabador.open(
                    rutaVideo.string(),
                    VideoWriter::fourcc('M', 'J', 'P', 'G'),
                    20.0,
                    ultimoMosaico.size(),
                    true
                );

                cout
                    << "[OK] Grabando "
                    << rutaVideo.string()
                    << endl;

                contadorVideos++;
            }

            grabador.write(
                ultimoMosaico
            );
        }

        int tecla =
            waitKey(1);

        if(tecla == 27 || tecla == 'q' || tecla == 'Q')
        {
            break;
        }

        if(tecla == 's' || tecla == 'S')
        {
            guardarImagen(
                carpetaSalida / ("captura_webcam_" + to_string(contadorCapturas) + ".jpg"),
                ultimoMosaico
            );

            contadorCapturas++;
        }

        if(tecla == 'r' || tecla == 'R')
        {
            grabando =
                !grabando;

            if(!grabando && grabador.isOpened())
            {
                grabador.release();
                cout
                    << "[OK] Video cerrado."
                    << endl;
            }
        }
    }

    if(grabador.isOpened())
    {
        grabador.release();
    }

    camara.release();
    destroyAllWindows();

    return 0;

}



void imprimirUso()
{

    cout
        << "Uso:" << endl
        << "  ./build/Practica2                  # webcam indice 0" << endl
        << "  ./build/Practica2 --webcam 1       # webcam indice 1" << endl
        << "  ./build/Practica2 --imagen imagen.jpg" << endl;

}



int main(
    int argc,
    char *argv[]
)
{

    if(argc >= 2)
    {
        string modo =
            argv[1];

        if(modo == "--imagen")
        {
            string ruta =
                argc >= 3 ? argv[2] : "imagen.jpg";

            return ejecutarImagen(
                ruta
            );
        }

        if(modo == "--webcam")
        {
            int indice =
                argc >= 3 ? stoi(argv[2]) : 0;

            return ejecutarWebcam(
                indice
            );
        }

        if(modo == "--help" || modo == "-h")
        {
            imprimirUso();
            return 0;
        }

        return ejecutarImagen(
            modo
        );
    }

    return ejecutarWebcam(
        0
    );

}

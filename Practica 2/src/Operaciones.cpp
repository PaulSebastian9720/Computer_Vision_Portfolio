#include "Operaciones.hpp"



// ===============================
// UTILIDADES
// ===============================


int Operaciones::kernelImpar(
    int kernel
)
{

    if(kernel < 1)
    {
        return 1;
    }

    if(kernel % 2 == 0)
    {
        return kernel + 1;
    }

    return kernel;

}



// ===============================
// KARL STRUSS
// ===============================


Mat Operaciones::mezclaCanales(
    const Mat &imagen,
    double alpha
)
{

    vector<Mat> canales;
    split(
        imagen,
        canales
    );

    alpha = min(
        1.0,
        max(
            0.0,
            alpha
        )
    );

    Mat resultado;
    addWeighted(
        canales[0],
        1.0 - alpha,
        canales[2],
        alpha,
        0.0,
        resultado
    );

    return resultado;

}



// ===============================
// ESCALA GRIS
// ===============================


Mat Operaciones::escalaGris(
    const Mat &imagen
)
{

    Mat gris;

    if(imagen.channels() == 1)
    {
        gris = imagen.clone();
    }
    else
    {
        cvtColor(
            imagen,
            gris,
            COLOR_BGR2GRAY
        );
    }

    return gris;

}



// ===============================
// SEPARAR CANALES
// ===============================


vector<Mat> Operaciones::separarCanales(
    const Mat &imagen
)
{

    vector<Mat> canales;
    split(
        imagen,
        canales
    );

    return canales;

}



// ===============================
// RUIDO GAUSSIANO
// ===============================


Mat Operaciones::ruidoGaussiano(
    const Mat &imagen,
    double sigma,
    const Mat &mascara
)
{

    Mat imagen32F;
    imagen.convertTo(
        imagen32F,
        CV_32F
    );

    Mat ruido(
        imagen.size(),
        imagen32F.type()
    );

    randn(
        ruido,
        Scalar::all(0.0),
        Scalar::all(sigma)
    );

    Mat ruidosa32F =
        imagen32F + ruido;

    Mat ruidosa;
    ruidosa32F.convertTo(
        ruidosa,
        imagen.type()
    );

    if(mascara.empty())
    {
        return ruidosa;
    }

    Mat salida =
        imagen.clone();

    ruidosa.copyTo(
        salida,
        mascara
    );

    return salida;

}



// ===============================
// RUIDO SPECKLE
// ===============================


Mat Operaciones::ruidoSpeckle(
    const Mat &imagen,
    double sigma,
    const Mat &mascara
)
{

    Mat imagen32F;
    imagen.convertTo(
        imagen32F,
        CV_32F
    );

    Mat ruido(
        imagen.size(),
        imagen32F.type()
    );

    randn(
        ruido,
        Scalar::all(0.0),
        Scalar::all(sigma)
    );

    Mat ruidosa32F =
        imagen32F + imagen32F.mul(ruido);

    Mat ruidosa;
    ruidosa32F.convertTo(
        ruidosa,
        imagen.type()
    );

    if(mascara.empty())
    {
        return ruidosa;
    }

    Mat salida =
        imagen.clone();

    ruidosa.copyTo(
        salida,
        mascara
    );

    return salida;

}



// ===============================
// FILTRO GAUSSIANO
// ===============================


Mat Operaciones::filtroGaussiano(
    const Mat &imagen,
    int kernel,
    double sigma
)
{

    Mat salida;
    int k =
        kernelImpar(kernel);

    GaussianBlur(
        imagen,
        salida,
        Size(k, k),
        sigma
    );

    return salida;

}



// ===============================
// FILTRO MEDIANA
// ===============================


Mat Operaciones::filtroMediana(
    const Mat &imagen,
    int kernel
)
{

    Mat salida;
    int k =
        kernelImpar(kernel);

    medianBlur(
        imagen,
        salida,
        k
    );

    return salida;

}



// ===============================
// MASCARA PRIMER PLANO
// ===============================


Mat Operaciones::mascaraPrimerPlanoDesdeBlanco(
    const Mat &imagen,
    int valorMinimo,
    int saturacionMaxima
)
{

    Mat hsv;
    cvtColor(
        imagen,
        hsv,
        COLOR_BGR2HSV
    );

    Mat mascaraFondo;
    inRange(
        hsv,
        Scalar(0, 0, valorMinimo),
        Scalar(179, saturacionMaxima, 255),
        mascaraFondo
    );

    Mat mascaraPrimerPlano;
    bitwise_not(
        mascaraFondo,
        mascaraPrimerPlano
    );

    Mat elemento =
        getStructuringElement(
            MORPH_ELLIPSE,
            Size(5, 5)
        );

    morphologyEx(
        mascaraPrimerPlano,
        mascaraPrimerPlano,
        MORPH_OPEN,
        elemento
    );

    morphologyEx(
        mascaraPrimerPlano,
        mascaraPrimerPlano,
        MORPH_CLOSE,
        elemento
    );

    return mascaraPrimerPlano;

}



// ===============================
// ESCENA CON FONDO COLOR
// ===============================


Mat Operaciones::crearEscenaConFondoColor(
    const Mat &imagen,
    const Scalar &colorFondo,
    Mat &mascaraPrimerPlano
)
{

    mascaraPrimerPlano =
        mascaraPrimerPlanoDesdeBlanco(
            imagen
        );

    Mat escena(
        imagen.size(),
        imagen.type(),
        colorFondo
    );

    imagen.copyTo(
        escena,
        mascaraPrimerPlano
    );

    return escena;

}



// ===============================
// CHROMA KEY HSV
// ===============================


Mat Operaciones::chromaKeyHSV(
    const Mat &imagen,
    const Scalar &hsvMin,
    const Scalar &hsvMax,
    const Mat &nuevoFondo,
    Mat &mascaraFondo
)
{

    Mat hsv;
    cvtColor(
        imagen,
        hsv,
        COLOR_BGR2HSV
    );

    inRange(
        hsv,
        hsvMin,
        hsvMax,
        mascaraFondo
    );

    Mat elemento =
        getStructuringElement(
            MORPH_ELLIPSE,
            Size(7, 7)
        );

    morphologyEx(
        mascaraFondo,
        mascaraFondo,
        MORPH_OPEN,
        elemento
    );

    morphologyEx(
        mascaraFondo,
        mascaraFondo,
        MORPH_CLOSE,
        elemento
    );

    Mat fondo;
    resize(
        nuevoFondo,
        fondo,
        imagen.size()
    );

    Mat salida =
        imagen.clone();

    fondo.copyTo(
        salida,
        mascaraFondo
    );

    return salida;

}



// ===============================
// FONDO GRADIENTE
// ===============================


Mat Operaciones::crearFondoGradiente(
    Size tamano
)
{

    Mat fondo(
        tamano,
        CV_8UC3
    );

    for(int y = 0; y < fondo.rows; y++)
    {
        for(int x = 0; x < fondo.cols; x++)
        {
            double fx =
                static_cast<double>(x) / max(1, fondo.cols - 1);

            double fy =
                static_cast<double>(y) / max(1, fondo.rows - 1);

            Vec3b &pixel =
                fondo.at<Vec3b>(y, x);

            pixel[0] =
                saturate_cast<uchar>(40 + 90 * (1.0 - fy));
            pixel[1] =
                saturate_cast<uchar>(120 + 80 * fx);
            pixel[2] =
                saturate_cast<uchar>(210 - 70 * fy);
        }
    }

    return fondo;

}



// ===============================
// MOSAICO
// ===============================


Mat Operaciones::crearMosaico(
    const vector<pair<string, Mat>> &imagenes,
    int columnas,
    Size tamanoCelda
)
{

    int margenSuperior =
        34;

    int filas =
        static_cast<int>(
            ceil(
                static_cast<double>(imagenes.size()) / columnas
            )
        );

    Mat mosaico(
        Size(
            columnas * tamanoCelda.width,
            filas * (tamanoCelda.height + margenSuperior)
        ),
        CV_8UC3,
        Scalar(245, 245, 245)
    );

    for(size_t i = 0; i < imagenes.size(); i++)
    {
        Mat imagen =
            imagenes[i].second;

        Mat bgr;
        if(imagen.channels() == 1)
        {
            cvtColor(
                imagen,
                bgr,
                COLOR_GRAY2BGR
            );
        }
        else
        {
            bgr =
                imagen.clone();
        }

        Mat redimensionada;
        resize(
            bgr,
            redimensionada,
            tamanoCelda
        );

        int fila =
            static_cast<int>(i) / columnas;
        int columna =
            static_cast<int>(i) % columnas;

        Rect roi(
            columna * tamanoCelda.width,
            fila * (tamanoCelda.height + margenSuperior) + margenSuperior,
            tamanoCelda.width,
            tamanoCelda.height
        );

        redimensionada.copyTo(
            mosaico(roi)
        );

        Point origenTexto(
            columna * tamanoCelda.width + 10,
            fila * (tamanoCelda.height + margenSuperior) + 23
        );

        putText(
            mosaico,
            imagenes[i].first,
            origenTexto,
            FONT_HERSHEY_SIMPLEX,
            0.62,
            Scalar(20, 20, 20),
            2,
            LINE_AA
        );
    }

    return mosaico;

}

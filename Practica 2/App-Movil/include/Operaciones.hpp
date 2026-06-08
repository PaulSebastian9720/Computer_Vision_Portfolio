#ifndef OPERACIONES_HPP
#define OPERACIONES_HPP


#include <opencv2/opencv.hpp>
#include <vector>
#include <string>


using namespace cv;
using namespace std;



class Operaciones
{

public:


    // Mezcla Karl Struss: Salida = (1-alpha) * B + alpha * R
    Mat mezclaCanales(
        const Mat &imagen,
        double alpha
    );


    // Escala de grises
    Mat escalaGris(
        const Mat &imagen
    );


    // Separar canales BGR
    vector<Mat> separarCanales(
        const Mat &imagen
    );


    // Ruido Gaussiano
    Mat ruidoGaussiano(
        const Mat &imagen,
        double sigma,
        const Mat &mascara = Mat()
    );


    // Ruido Speckle
    Mat ruidoSpeckle(
        const Mat &imagen,
        double sigma,
        const Mat &mascara = Mat()
    );


    // Filtro Gaussiano
    Mat filtroGaussiano(
        const Mat &imagen,
        int kernel,
        double sigma = 0.0
    );


    // Filtro Mediana
    Mat filtroMediana(
        const Mat &imagen,
        int kernel
    );


    // Mascara del primer plano cuando el fondo original es blanco
    Mat mascaraPrimerPlanoDesdeBlanco(
        const Mat &imagen,
        int valorMinimo = 245,
        int saturacionMaxima = 70
    );


    // Construir una escena con fondo uniforme para probar Chroma Key
    Mat crearEscenaConFondoColor(
        const Mat &imagen,
        const Scalar &colorFondo,
        Mat &mascaraPrimerPlano
    );


    // Reemplazar un fondo verde/azul detectado en HSV
    Mat chromaKeyHSV(
        const Mat &imagen,
        const Scalar &hsvMin,
        const Scalar &hsvMax,
        const Mat &nuevoFondo,
        Mat &mascaraFondo
    );


    // Fondo sintetico para sustituir el color segmentado
    Mat crearFondoGradiente(
        Size tamano
    );


    // Mosaico comparativo con etiquetas
    Mat crearMosaico(
        const vector<pair<string, Mat>> &imagenes,
        int columnas,
        Size tamanoCelda = Size(360, 240)
    );


private:


    int kernelImpar(
        int kernel
    );


};


#endif

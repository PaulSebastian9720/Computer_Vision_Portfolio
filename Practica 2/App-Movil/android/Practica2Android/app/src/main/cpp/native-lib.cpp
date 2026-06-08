#include <jni.h>
#include <cstdio>
#include <string>
#include <vector>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;

static int kernelImpar(int kernel) {
    if (kernel < 1) return 1;
    return (kernel % 2 == 0) ? kernel + 1 : kernel;
}

static Mat fondoGradiente(Size tamano) {
    Mat fondo(tamano, CV_8UC3);
    for (int y = 0; y < fondo.rows; y++) {
        for (int x = 0; x < fondo.cols; x++) {
            double fx = static_cast<double>(x) / std::max(1, fondo.cols - 1);
            double fy = static_cast<double>(y) / std::max(1, fondo.rows - 1);
            Vec3b &pixel = fondo.at<Vec3b>(y, x);
            pixel[0] = saturate_cast<uchar>(40 + 90 * (1.0 - fy));
            pixel[1] = saturate_cast<uchar>(120 + 80 * fx);
            pixel[2] = saturate_cast<uchar>(210 - 70 * fy);
        }
    }
    return fondo;
}

static void obtenerMascaraPrecisa(const Mat &hsv, int hMin, int hMax, int sMin, int vMin, Mat &mascaraFinal) {
    Mat mascara;
    inRange(hsv, Scalar(hMin, sMin, vMin), Scalar(hMax, 255, 255), mascara);

    // Limpieza agresiva para unir hombros y cara
    Mat element3 = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    Mat element7 = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
    Mat element15 = getStructuringElement(MORPH_ELLIPSE, Size(15, 15));

    // 1. Eliminar ruido (Apertura)
    morphologyEx(mascara, mascara, MORPH_OPEN, element3);

    // 2. Unir partes separadas (Dilatación antes del cierre para conectar hombros)
    dilate(mascara, mascara, element7, Point(-1,-1), 1);

    // 3. Rellenar huecos grandes (Cierre)
    morphologyEx(mascara, mascara, MORPH_CLOSE, element15);

    // 4. Suavizar bordes (Erosión para compensar la dilatación previa)
    erode(mascara, mascara, element7, Point(-1,-1), 1);

    // 5. Filtro de contornos: Mantener solo contornos grandes (Persona)
    std::vector<std::vector<Point>> contornos;
    findContours(mascara.clone(), contornos, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    mascaraFinal = Mat::zeros(mascara.size(), CV_8UC1);
    double maxArea = mascara.rows * mascara.cols * 0.05; // Mínimo 5% de la pantalla

    for (size_t i = 0; i < contornos.size(); i++) {
        if (contourArea(contornos[i]) > maxArea) {
            drawContours(mascaraFinal, contornos, (int)i, Scalar(255), FILLED);
        }
    }

    // Si no hay nada grande, mostrar todo lo detectado para no dejar pantalla negra
    if (countNonZero(mascaraFinal) == 0) {
        mascara.copyTo(mascaraFinal);
    }
}

static Mat chromaKey(
    const Mat &rgba,
    const Mat &fondoImg,
    int hMin, int hMax,
    int sMin, int vMin,
    Mat &mascaraPrimerPlano,
    bool detectarUsuario
) {
    Mat bgr;
    cvtColor(rgba, bgr, COLOR_RGBA2BGR);

    Mat hsv;
    Mat bgrBlur;
    medianBlur(bgr, bgrBlur, 5); // Mejor que Gaussian para bordes
    cvtColor(bgrBlur, hsv, COLOR_BGR2HSV);

    Mat mascaraDeteccion;
    obtenerMascaraPrecisa(hsv, hMin, hMax, sMin, vMin, mascaraDeteccion);

    Mat mascaraFondo;
    if (detectarUsuario) {
        mascaraPrimerPlano = mascaraDeteccion.clone();
        bitwise_not(mascaraPrimerPlano, mascaraFondo);
    } else {
        mascaraFondo = mascaraDeteccion.clone();
        bitwise_not(mascaraFondo, mascaraPrimerPlano);
    }

    Mat salida = bgr.clone();
    Mat fondoFinal;

    if (!fondoImg.empty()) {
        resize(fondoImg, fondoFinal, bgr.size());
        if (fondoFinal.channels() == 4) cvtColor(fondoFinal, fondoFinal, COLOR_RGBA2BGR);
    } else {
        fondoFinal = fondoGradiente(bgr.size());
    }

    fondoFinal.copyTo(salida, mascaraFondo);
    return salida;
}

static void agregarRuido(Mat &imagen, const Mat &mascara, int gauss, int speckle) {
    if (gauss > 0) {
        Mat imagen32F;
        imagen.convertTo(imagen32F, CV_32F);
        Mat ruido(imagen.size(), imagen32F.type());
        randn(ruido, Scalar::all(0.0), Scalar::all(static_cast<double>(gauss)));
        Mat temp = imagen32F + ruido;
        Mat ruidosa;
        temp.convertTo(ruidosa, imagen.type());
        ruidosa.copyTo(imagen, mascara);
    }

    if (speckle > 0) {
        Mat imagen32F;
        imagen.convertTo(imagen32F, CV_32F);
        Mat ruido(imagen.size(), imagen32F.type());
        randn(ruido, Scalar::all(0.0), Scalar::all(static_cast<double>(speckle) / 100.0));
        Mat temp = imagen32F + imagen32F.mul(ruido);
        Mat ruidosa;
        temp.convertTo(ruidosa, imagen.type());
        ruidosa.copyTo(imagen, mascara);
    }
}

static void rotular(Mat &im, const std::string &txt) {
    putText(im, txt, Point(15, 30), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(0, 0, 0, 255), 4, LINE_AA);
    putText(im, txt, Point(15, 30), FONT_HERSHEY_SIMPLEX, 0.55, Scalar(255, 255, 255, 255), 2, LINE_AA);
}

static void copiarSubMosaico(const Mat &src, Mat dst, const std::string &label) {
    Mat temp;
    if (src.channels() == 1) {
        cvtColor(src, temp, COLOR_GRAY2RGBA);
    } else if (src.channels() == 3) {
        cvtColor(src, temp, COLOR_BGR2RGBA);
    } else {
        src.copyTo(temp);
    }
    resize(temp, temp, dst.size());
    rotular(temp, label);
    temp.copyTo(dst);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_practica2_MainActivity_procesarFrame(
    JNIEnv *, jobject, jlong rgbaAddr, jlong bgAddr,
    jint modo, jboolean detectarUsuario,
    jint alphaBlue, jint sMin, jint vMin,
    jint hMin, jint hMax, jint gauss, jint speckle, jint kernel
) {
    Mat &rgba = *reinterpret_cast<Mat *>(rgbaAddr);
    Mat &fondoImg = *reinterpret_cast<Mat *>(bgAddr);

    if (modo == 1) return;

    double alpha = static_cast<double>(alphaBlue) / 100.0;

    Mat bgr;
    cvtColor(rgba, bgr, COLOR_RGBA2BGR);
    std::vector<Mat> canales;
    split(bgr, canales);
    Mat mezclaBlue;
    addWeighted(canales[0], 1.0 - alpha, canales[2], alpha, 0.0, mezclaBlue);

    if (modo == 2) {
        Mat res; cvtColor(mezclaBlue, res, COLOR_GRAY2RGBA);
        res.copyTo(rgba); return;
    }

    Mat mascaraPrimerPlano;
    Mat chroma = chromaKey(rgba, fondoImg, hMin, hMax, sMin, vMin, mascaraPrimerPlano, detectarUsuario);

    if (modo == 3) {
        Mat res; cvtColor(mascaraPrimerPlano, res, COLOR_GRAY2RGBA);
        res.copyTo(rgba); return;
    }

    if (modo == 4) {
        Mat soloUsuario = Mat::zeros(bgr.size(), bgr.type());
        bgr.copyTo(soloUsuario, mascaraPrimerPlano);
        Mat res; cvtColor(soloUsuario, res, COLOR_BGR2RGBA);
        res.copyTo(rgba); return;
    }

    if (modo == 5) {
        Mat res; cvtColor(chroma, res, COLOR_BGR2RGBA);
        res.copyTo(rgba); return;
    }

    Mat ruidosa = chroma.clone();
    agregarRuido(ruidosa, mascaraPrimerPlano, gauss, speckle);
    Mat filtrada;
    medianBlur(ruidosa, filtrada, kernelImpar(kernel));

    if (modo == 6) {
        Mat res; cvtColor(filtrada, res, COLOR_BGR2RGBA);
        res.copyTo(rgba); return;
    }

    if (modo == 0) {
        int w = rgba.cols / 3, h = rgba.rows / 3;
        if (w <= 0 || h <= 0) return;

        Mat mosaico = Mat::zeros(rgba.size(), rgba.type());

        copiarSubMosaico(rgba, mosaico(Rect(0, 0, w, h)), "1. Original");
        copiarSubMosaico(mezclaBlue, mosaico(Rect(w, 0, w, h)), "2. Karl Struss");
        copiarSubMosaico(mascaraPrimerPlano, mosaico(Rect(w * 2, 0, w, h)), "3. Mask (White=Keep)");

        Mat soloUsuario = Mat::zeros(bgr.size(), bgr.type());
        bgr.copyTo(soloUsuario, mascaraPrimerPlano);
        copiarSubMosaico(soloUsuario, mosaico(Rect(0, h, w, h)), "4. Solo Usuario");

        Mat fondoPanel;
        if (!fondoImg.empty()) {
            resize(fondoImg, fondoPanel, Size(w, h));
            if (fondoPanel.channels() == 3) cvtColor(fondoPanel, fondoPanel, COLOR_BGR2RGBA);
        } else {
            Mat grad = fondoGradiente(Size(w, h));
            cvtColor(grad, fondoPanel, COLOR_BGR2RGBA);
        }
        rotular(fondoPanel, "5. Fondo Imagen");
        fondoPanel.copyTo(mosaico(Rect(w, h, w, h)));

        copiarSubMosaico(chroma, mosaico(Rect(w * 2, h, w, h)), "5. Fondo Imagen");

        copiarSubMosaico(ruidosa, mosaico(Rect(0, h * 2, w, h)), "7. Ruido");
        copiarSubMosaico(filtrada, mosaico(Rect(w, h * 2, w, h)), "8. Filtrada");

        Mat maskInversa; bitwise_not(mascaraPrimerPlano, maskInversa);
        copiarSubMosaico(maskInversa, mosaico(Rect(w * 2, h * 2, w, h)), "9. Mask Fondo");

        mosaico.copyTo(rgba);
    }
}

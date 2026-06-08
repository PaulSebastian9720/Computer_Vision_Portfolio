# Practica 2 Android/JNI

Base Android Studio para cubrir el criterio de entorno nativo Android (JNI + OpenCV).

## Requisitos

- Android Studio.
- OpenCV Android SDK descargado.
- Celular Android con camara o emulador con camara configurada.

## Configuracion

1. Abrir esta carpeta en Android Studio:

   `android/Practica2Android`

2. Copiar el OpenCV Android SDK dentro del proyecto con esta estructura:

   `android/Practica2Android/opencv-sdk/sdk/native/jni/OpenCVConfig.cmake`

3. Sincronizar Gradle.

4. Ejecutar en el celular.

## Que hace

- Captura frames de la camara con `JavaCameraView`.
- Envia cada `Mat` RGBA a C++ mediante JNI.
- En `native-lib.cpp` aplica:
  - Chroma Key en HSV.
  - Insercion de fondo sintetico.
  - Ruido Gaussiano/Speckle sobre primer plano.
  - Filtro de Mediana.
  - Mezcla Karl Struss `(1 - alpha) * B + alpha * R`.
- Muestra FPS y parametros en pantalla.

## Controles

- SeekBar Alpha: peso de canal rojo.
- SeekBar H Min/H Max: rango de color del fondo para Chroma Key.
- SeekBar Gauss/Speckle: intensidad de ruido.
- SeekBar Kernel: tamano del filtro.

## Nota

Este proyecto no se compilo en esta maquina porque el entorno actual no tiene `gradle`, Android SDK ni OpenCV Android SDK instalados. La logica C++ nativa esta incluida y lista para integrarse en Android Studio.

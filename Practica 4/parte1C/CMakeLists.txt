cmake_minimum_required(VERSION 3.16)
project(Practica4Parte1C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "")

# Por defecto, el mismo OpenCV local que "Proyecto final V2" (sin CUDA) — asi
# preprocesamiento_cpu ya compila y corre en este equipo. En el laboratorio de Computo 8,
# sobreescribir OpenCV_DIR para que apunte al OpenCV compilado CON CUDA (WITH_CUDA=ON +
# opencv_contrib); ahi tambien se habilita preprocesamiento_gpu.
set(OpenCV_DIR "/home/paul/Aplicaciones/opencv/install/lib/cmake/opencv4"
    CACHE PATH "OpenCV cmake config directory")

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV ${OpenCV_VERSION} (CUDA: ${OpenCV_CUDA_VERSION}) — ${OpenCV_LIBS}")

# ── Pipeline CPU (Tabla 1): no requiere CUDA, compila con cualquier OpenCV ─────────────────────
add_executable(preprocesamiento_cpu src/pipeline_cpu.cpp)
target_link_libraries(preprocesamiento_cpu PRIVATE ${OpenCV_LIBS})
target_compile_options(preprocesamiento_cpu PRIVATE -O2 -Wall -Wextra)

# ── Pipeline GPU-only (Tabla 2): requiere OpenCV con WITH_CUDA=ON ──────────────────────────────
if(OpenCV_CUDA_VERSION)
    add_executable(preprocesamiento_gpu src/pipeline_gpu.cpp)
    target_link_libraries(preprocesamiento_gpu PRIVATE ${OpenCV_LIBS})
    target_compile_options(preprocesamiento_gpu PRIVATE -O2 -Wall -Wextra)
else()
    message(WARNING "OpenCV sin soporte CUDA (OpenCV_DIR=${OpenCV_DIR}): "
                     "preprocesamiento_gpu NO se agrega al build. "
                     "Sobreescribir OpenCV_DIR con una build WITH_CUDA=ON (laboratorio) para compilarlo.")
endif()

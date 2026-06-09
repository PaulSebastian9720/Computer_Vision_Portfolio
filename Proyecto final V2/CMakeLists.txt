cmake_minimum_required(VERSION 3.16)
project(StereoVisionV2 CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "")

# Custom OpenCV install path
set(OpenCV_DIR "/home/paul/Aplicaciones/opencv/install/lib/cmake/opencv4"
    CACHE PATH "OpenCV cmake config directory")

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

set(SOURCES
    src/main.cpp
    src/CameraStream.cpp
    src/StereoProcessor.cpp
    src/DepthEstimator.cpp
    src/ObjectDetector.cpp
    src/HandTracker.cpp
    src/ARFilter.cpp
)

add_executable(stereo_v2 ${SOURCES})

target_include_directories(stereo_v2 PRIVATE src/)

target_link_libraries(stereo_v2 PRIVATE
    ${OpenCV_LIBS}
    Threads::Threads
)

target_compile_options(stereo_v2 PRIVATE -O2 -Wall -Wextra -Wpedantic)

# RPATH so the binary finds shared libs without LD_LIBRARY_PATH
set_target_properties(stereo_v2 PROPERTIES
    BUILD_RPATH  "/home/paul/Aplicaciones/opencv/install/lib"
    INSTALL_RPATH "/home/paul/Aplicaciones/opencv/install/lib"
)

# ── AR Filter (standalone, webcam + hand tracker) ─────────────────────────────
add_executable(ar_filter
    src/main_ar.cpp
    src/HandTracker.cpp
    src/ARFilter.cpp
)

target_include_directories(ar_filter PRIVATE src/)

target_link_libraries(ar_filter PRIVATE
    ${OpenCV_LIBS}
    Threads::Threads
)

target_compile_options(ar_filter PRIVATE -O2 -Wall -Wextra)

set_target_properties(ar_filter PROPERTIES
    BUILD_RPATH  "/home/paul/Aplicaciones/opencv/install/lib"
    INSTALL_RPATH "/home/paul/Aplicaciones/opencv/install/lib"
)

message(STATUS "OpenCV ${OpenCV_VERSION} — ${OpenCV_LIBS}")

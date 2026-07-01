cmake_minimum_required(VERSION 3.22.1)

project(shapesignature)

find_package(OpenCV REQUIRED)

add_library(
    shapesignature
    SHARED
    native-lib.cpp
)

target_link_libraries(
    shapesignature
    ${OpenCV_LIBS}
    android
    log
)

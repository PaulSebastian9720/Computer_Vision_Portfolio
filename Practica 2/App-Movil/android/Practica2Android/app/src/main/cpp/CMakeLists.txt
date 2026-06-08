cmake_minimum_required(VERSION 3.22.1)

project(practica2)

find_package(OpenCV REQUIRED)

add_library(
    practica2
    SHARED
    native-lib.cpp
)

target_link_libraries(
    practica2
    ${OpenCV_LIBS}
    android
    log
)

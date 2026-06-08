cmake_minimum_required(VERSION 3.10)

project(Practica2)


set(CMAKE_CXX_STANDARD 17)


find_package(OpenCV REQUIRED)


include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
)



add_executable(
    Practica2

    main.cpp
    src/Operaciones.cpp
)



target_link_libraries(
    Practica2

    ${OpenCV_LIBS}
)
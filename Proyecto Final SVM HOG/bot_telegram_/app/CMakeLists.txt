cmake_minimum_required(VERSION 3.16)
project(TruckTrigger CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "")

find_package(OpenCV REQUIRED)

add_executable(truck_trigger main.cpp)
add_executable(truck_trigger_video main_video.cpp)
add_executable(truck_detector_dual truck_detector_dual.cpp)

foreach(target truck_trigger truck_trigger_video truck_detector_dual)
    target_link_libraries(${target} PRIVATE ${OpenCV_LIBS})
    target_compile_options(${target} PRIVATE -O2 -Wall -Wextra -Wpedantic)
endforeach()

message(STATUS "OpenCV ${OpenCV_VERSION} — ${OpenCV_LIBS}")

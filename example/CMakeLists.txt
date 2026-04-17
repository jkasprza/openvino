cmake_minimum_required(VERSION 3.21)

project(interop_example LANGUAGES CXX)
set(TARGET_NAME interop_example)

find_package(OpenVINO REQUIRED COMPONENTS Runtime)
find_package(OpenCL REQUIRED)

add_executable(${TARGET_NAME} main.cpp)
target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime OpenCL::OpenCL)
target_compile_definitions(${TARGET_NAME} PRIVATE CL_TARGET_OPENCL_VERSION=300)

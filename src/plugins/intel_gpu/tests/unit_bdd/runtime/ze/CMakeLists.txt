# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if (NOT GPU_RT_TYPE STREQUAL "L0")
    return()
endif()

set(TARGET_NAME "ov_gpu_unit_bdd_runtime_ze")

file(GLOB_RECURSE SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
)

add_library(${TARGET_NAME} OBJECT ${SOURCES})

target_link_libraries(${TARGET_NAME} PRIVATE openvino_intel_gpu_runtime common_test_utils gtest LevelZero::LevelZero)
target_include_directories(${TARGET_NAME} PRIVATE 
    "${CMAKE_HOME_DIRECTORY}/src/plugins/intel_gpu/src/runtime/ze"
    )
set(UNIT_TEST_OBJECTS  "${UNIT_TEST_OBJECTS};$<TARGET_OBJECTS:${TARGET_NAME}>" CACHE INTERNAL "Unit test objects" FORCE)
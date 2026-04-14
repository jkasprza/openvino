// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/ocl_ze_converter.hpp"
#include "ocl_device_detector.hpp"
#include "ocl_common.hpp"

namespace cldnn {
namespace {

static constexpr cl_uint CL_L0_CONTEXT_HANDLE = 0x10021;
static constexpr cl_uint CL_L0_IMMEDIATE_CMD_LIST_HANDLE = 0x10022;
//static constexpr cl_uint CL_L0_EVENT_HANDLE = 0x10023;
static constexpr cl_uint CL_L0_MEM_OBJ_HANDLE = 0x10024;
static constexpr cl_uint CL_L0_DEVICE_HANDLE = 0x10025;

} // namespace

ze_handle ocl_ze_converter::convert_ocl_context_to_ze(ocl_handle ocl_ctx) {
    static_assert(sizeof(ocl_handle) == sizeof(cl_context), "[GPU] Expected ocl_handle to be same size as cl_context");
    cl_context context = reinterpret_cast<cl_context>(ocl_ctx);
    cl_int error;
    ze_handle ze_context = nullptr;
    error = clGetContextInfo(context, CL_L0_CONTEXT_HANDLE, sizeof(ze_handle), &ze_context, nullptr);
    OPENVINO_ASSERT(error == CL_SUCCESS, "[GPU] clGetContextInfo error code: ", std::to_string(error));
    OPENVINO_ASSERT(ze_context != nullptr, "[GPU] Received nullptr when converting OCL context");
    return ze_context;
}

ze_handle ocl_ze_converter::convert_ocl_queue_to_ze(ocl_handle ocl_queue) {
    static_assert(sizeof(ocl_handle) == sizeof(cl_command_queue), "[GPU] Expected ocl_handle to be same size as cl_command_queue");
    cl_command_queue queue = reinterpret_cast<cl_command_queue>(ocl_queue);
    cl_int error;
    ze_handle ze_cmd_list;
    error = clGetCommandQueueInfo(queue, CL_L0_IMMEDIATE_CMD_LIST_HANDLE, sizeof(ze_handle), &ze_cmd_list, nullptr);
    OPENVINO_ASSERT(error == CL_SUCCESS, "[GPU] clGetCommandQueueInfo error code: ", std::to_string(error));
    OPENVINO_ASSERT(ze_cmd_list != nullptr, "[GPU] Received nullptr when converting OCL command queue");
    return ze_cmd_list;
}

ze_handle ocl_ze_converter::convert_ocl_buffer_to_ze(ocl_handle ocl_buffer) {
    static_assert(sizeof(ocl_handle) == sizeof(cl_mem), "[GPU] Expected ocl_handle to be same size as cl_mem");
    cl_mem mem = reinterpret_cast<cl_mem>(ocl_buffer);
    cl_int error;
    ze_handle ze_mem;
    error = clGetMemObjectInfo(mem, CL_L0_MEM_OBJ_HANDLE, sizeof(ze_handle), &ze_mem, nullptr);
    OPENVINO_ASSERT(error == CL_SUCCESS, "[GPU] clGetMemObjectInfo error code: ", std::to_string(error));
    OPENVINO_ASSERT(ze_mem != nullptr, "[GPU] Received nullptr when converting OCL buffer");
    return ze_mem;
}

ze_handle ocl_ze_converter::convert_ocl_device_to_ze(ocl_handle ocl_device) {
    static_assert(sizeof(ocl_handle) == sizeof(cl_device_id), "[GPU] Expected ocl_handle to be same size as cl_mem");
    cl_device_id device = reinterpret_cast<cl_device_id>(ocl_device);
    ze_handle ze_device;
    cl_int error = clGetDeviceInfo(device, CL_L0_DEVICE_HANDLE, sizeof(ze_handle), &ze_device, nullptr);
    OPENVINO_ASSERT(error == CL_SUCCESS, "[GPU] clGetDeviceInfo error code: ", std::to_string(error));
    OPENVINO_ASSERT(ze_device != nullptr, "[GPU] Received nullptr when converting OCL device");
    return ze_device;
}

std::vector<ze_handle> ocl_ze_converter::get_ze_devices_from_ocl_context(ocl_handle ocl_ctx) {
    cl::Context ctx = cl::Context(static_cast<cl_context>(ocl_ctx), true);
    auto all_devices = ctx.getInfo<CL_CONTEXT_DEVICES>();

    std::vector<ze_handle> supported_devices;
    for (size_t i = 0; i < all_devices.size(); i++) {
        auto& device = all_devices[i];
        if (!ocl::does_device_match_config(device.get()))
            continue;

        supported_devices.emplace_back(convert_ocl_device_to_ze(device.get()));
    }
    return supported_devices;
}

ocl_handle ocl_ze_converter::convert_ze_context_to_ocl(ze_handle ze_ctx, ocl_handle ocl_device) {
    cl_context_properties properties[] = {CL_L0_CONTEXT_HANDLE, reinterpret_cast<cl_context_properties>(ze_ctx), 0};
    constexpr cl_uint num_devices = 1;
    cl_int error;
    auto converted_context = clCreateContext(properties, num_devices, reinterpret_cast<cl_device_id*>(&ocl_device), nullptr, nullptr, &error);
    OPENVINO_ASSERT(error == CL_SUCCESS, "[GPU] clCreateContext error code: ", std::to_string(error));
    OPENVINO_ASSERT(converted_context != nullptr, "[GPU] Received nullptr when converting ZE context");
    return converted_context;
}

ocl_handle ocl_ze_converter::convert_ze_cmd_list_to_ocl(ocl_handle ocl_ctx, ocl_handle ocl_device, ze_handle ze_cmd_list) {
    cl_mem_properties properties[] = {CL_L0_IMMEDIATE_CMD_LIST_HANDLE, reinterpret_cast<cl_properties>(ze_cmd_list), 0};
    cl_int error;
    auto converted_queue = clCreateCommandQueueWithProperties(reinterpret_cast<cl_context>(ocl_ctx), reinterpret_cast<cl_device_id>(ocl_device), properties, &error);
    OPENVINO_ASSERT(error == CL_SUCCESS, "[GPU] clCreateCommandQueueWithProperties error code: ", std::to_string(error));
    OPENVINO_ASSERT(converted_queue != nullptr, "[GPU] Received nullptr when converting ZE command list");
    return converted_queue;
}

ocl_handle ocl_ze_converter::convert_ze_buffer_to_ocl(ocl_handle ocl_ctx, ze_handle ze_buffer) {
    cl_mem_properties properties[] = {CL_L0_MEM_OBJ_HANDLE, reinterpret_cast<cl_mem_properties>(ze_buffer), 0};
    cl_int error;
    // Size of the buffer should not be relevant for the conversion
    constexpr size_t size = 0;
    cl_mem converted_mem = clCreateBufferWithProperties(reinterpret_cast<cl_context>(ocl_ctx), properties, 0, size, nullptr, &error);
    OPENVINO_ASSERT(error == CL_SUCCESS, "[GPU] clCreateBufferWithProperties error code: ", std::to_string(error));
    OPENVINO_ASSERT(converted_mem != nullptr, "[GPU] Received nullptr when converting ZE buffer");
    return converted_mem;
}

}  // namespace cldnn

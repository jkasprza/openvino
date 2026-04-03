// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/ocl_ze_converter.hpp"
#include "ocl_common.hpp"

namespace cldnn {
namespace {

static constexpr cl_context_info CL_L0_CONTEXT_HANDLE = 0x10021;
static constexpr cl_context_info CL_L0_IMMEDIATE_CMD_LIST_HANDLE = 0x10022;
//static constexpr cl_context_info CL_L0_EVENT_HANDLE = 0x10023;
static constexpr cl_context_info CL_L0_MEM_OBJ_HANDLE = 0x10024;

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

}  // namespace cldnn

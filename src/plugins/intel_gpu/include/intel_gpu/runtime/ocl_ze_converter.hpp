// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "layout.hpp"
#include <vector>

namespace cldnn {

// Define handles here to remain runtime agnostic
using ocl_handle = void*;
using ze_handle = void*;

class ocl_ze_converter {
public:
    static ze_handle get_ze_context_from_cl_context(ocl_handle ocl_ctx);
    static ze_handle get_ze_cmd_list_from_cl_queue(ocl_handle ocl_queue);
    static ze_handle get_ze_mem_from_cl_mem(ocl_handle ocl_mem);
    static ze_handle get_ze_device_from_cl_device(ocl_handle ocl_device);
    static ocl_handle create_cl_context_from_ze_context(ocl_handle ocl_device, ze_handle ze_ctx);
    static ocl_handle create_cl_queue_from_ze_cmd_list(ocl_handle ocl_ctx, ocl_handle ocl_device, ze_handle ze_cmd_list);
    static ocl_handle create_cl_buffer_from_ze_usm(ocl_handle ocl_context, ze_handle ze_buffer);
    static ocl_handle create_cl_image_from_ze_image(ocl_handle ocl_context, ze_handle ze_image, const layout &layout);
};

}  // namespace cldnn

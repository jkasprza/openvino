// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

namespace cldnn {

// Define handles here to remain runtime agnostic
using ocl_handle = void*;
using ze_handle = void*;

class ocl_ze_converter {
public:
    static ze_handle convert_ocl_context_to_ze(ocl_handle ocl_ctx);
    static ze_handle convert_ocl_queue_to_ze(ocl_handle ocl_queue);
    static ze_handle convert_ocl_buffer_to_ze(ocl_handle ocl_buffer);
    static ze_handle convert_ocl_device_to_ze(ocl_handle ocl_device);
    static std::vector<ze_handle> get_ze_devices_from_ocl_context(ocl_handle ocl_ctx);
    static ocl_handle convert_ze_context_to_ocl(ze_handle ze_ctx, ocl_handle ocl_device);
    static ocl_handle convert_ze_cmd_list_to_ocl(ocl_handle ocl_ctx, ocl_handle ocl_device, ze_handle ze_cmd_list);
    static ocl_handle convert_ze_buffer_to_ocl(ocl_handle ocl_context, ze_handle ze_buffer);
};

}  // namespace cldnn

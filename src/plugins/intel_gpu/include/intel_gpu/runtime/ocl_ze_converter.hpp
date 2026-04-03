// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace cldnn {

// Define handles here to remain runtime agnostic
typedef void* ocl_handle;
typedef void* ze_handle;

class ocl_ze_converter {
public:
    static ze_handle convert_ocl_context_to_ze(ocl_handle ocl_ctx);
    static ze_handle convert_ocl_queue_to_ze(ocl_handle ocl_queue);
    static ze_handle convert_ocl_buffer_to_ze(ocl_handle ocl_buffer);
    
};

}  // namespace cldnn

// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdpa_spirv.hpp"
#include "kernels/sdpa_sycl_kernel.spv.inc"

#include <string>

namespace ov::intel_gpu::spirv { 
    std::string SDPAGenerator1::get_spirv() const {
        return std::string(reinterpret_cast<char const*>(sdpa_sycl_kernel_spv), sdpa_sycl_kernel_spv_len);
    }

    std::string SDPAGenerator2::get_spirv() const {
        return std::string(reinterpret_cast<char const*>(sdpa_sycl_kernel_spv), sdpa_sycl_kernel_spv_len);
    }
}
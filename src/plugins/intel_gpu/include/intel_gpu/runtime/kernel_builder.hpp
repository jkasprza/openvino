// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel.hpp"
#include "kernel_args.hpp"

#include <memory>
#include <string>
#include <vector>

namespace cldnn {

/// @brief Defines possible kernel formats
enum class KernelFormat {
    SOURCE,         ///< source code format
    INTERMEDIATE,   ///< Intermediate language format
    NATIVE_BIN,     ///< device native binary format
};

inline KernelFormat get_kernel_format(const kernel_language &lang) {
    switch (lang) {
        case kernel_language::OCLC:
        case kernel_language::CM:
        case kernel_language::OCLC_V2: {
            return KernelFormat::SOURCE;
        }
        case kernel_language::SPIRV: {
            return KernelFormat::INTERMEDIATE;
        }
        default:
            OPENVINO_THROW("Unexpected kernel language");
    }
}

/// @brief Interface for building the GPU kernels. Implementations must be thread-safe to support case where multiple threads use single builder.
class kernel_builder {
public:
    virtual ~kernel_builder() = default;
    virtual void build_kernels(const void *src, size_t src_bytes, KernelFormat src_format, const std::string &options, std::vector<kernel::ptr> &out) const = 0;
};

}  // namespace cldnn

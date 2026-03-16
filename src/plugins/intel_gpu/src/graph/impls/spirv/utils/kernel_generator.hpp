// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "common_utils/jitter.hpp"
#include "common_utils/kernel_generator_base.hpp"

namespace ov::intel_gpu::spirv {

class KernelGenerator : public KernelGeneratorBase {
public:
    // kernel name is the id used to find kernel in database while suffix is an optional identifier of specific stage
    // for multi-stage implementations which makes kernel names more clear in any kind of profiles
    explicit KernelGenerator(std::string_view entry_point) : m_entry_point(entry_point) {}
    virtual ~KernelGenerator() = default;

    // Code generator is not supposed to be copied/moved as it's mainly used once to produce KernelData
    // or to query DispatchDataFunc during importing compiled blob. After that generators can be removed
    KernelGenerator(const KernelGenerator&) = delete;
    KernelGenerator(KernelGenerator&&) = delete;
    KernelGenerator& operator=(const KernelGenerator&) = delete;
    KernelGenerator& operator=(KernelGenerator&&) = delete;

    [[nodiscard]] KernelData get_kernel_data(const RuntimeParams& params) const override;

protected:
    // Defines mapping between kernel argument and primitive_inst's memory buffers
    // Count of elements in the vector must match count of kernel arguments
    [[nodiscard]] virtual Arguments get_arguments_desc(const RuntimeParams& params) const;

    [[nodiscard]] virtual std::string get_spirv() const = 0;

private:
    std::string m_entry_point;
};

}  // namespace ov::intel_gpu::spirv

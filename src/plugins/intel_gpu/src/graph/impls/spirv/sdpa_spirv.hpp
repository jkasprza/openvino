// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "utils/kernel_generator.hpp"

namespace ov::intel_gpu::spirv { 
struct SDPAGenerator1 : public KernelGenerator {
    SDPAGenerator1() : KernelGenerator("PLACEHOLDER") {}
    //[[nodiscard]] virtual Arguments get_arguments_desc(const RuntimeParams& params) const override;
    [[nodiscard]] virtual std::string get_spirv() const override;
};
struct SDPAGenerator2 : public KernelGenerator {
    SDPAGenerator2() : KernelGenerator("PLACEHOLDER") {}
    //[[nodiscard]] virtual Arguments get_arguments_desc(const RuntimeParams& params) const override;
    [[nodiscard]] virtual std::string get_spirv() const override;
};
}
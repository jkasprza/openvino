// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_generator.hpp"

#include <cctype>

#include "intel_gpu/runtime/kernel_args.hpp"

namespace ov::intel_gpu::spirv {

KernelData KernelGenerator::get_kernel_data(const RuntimeParams& params) const {
    KernelData kd;
    kd.code = std::make_shared<KernelString>();
    kd.code->language = KernelLanguage::SPIRV;
    kd.code->entry_point = m_entry_point;
    kd.code->jit = {};
    kd.code->undefs = "";
    kd.code->options = "";
    kd.code->batch_compilation = false;
    kd.code->has_microkernels = false;
    kd.code->str = get_spirv();

    kd.params.arguments = get_arguments_desc(params);
    kd.params.layerID = params.desc->id;
    kd.update_dispatch_data_func = get_dispatch_data_func();
    kd.need_args_update = true;
    kd.need_dispatch_data_update = true;

    return kd;
}

Arguments KernelGenerator::get_arguments_desc(const RuntimeParams& params) const {
    Arguments args;

    if (params.is_dynamic()) {
        args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
    }

    for (uint32_t i = 0; i < params.input_layouts.size(); i++) {
        args.push_back({ArgumentDescriptor::Types::INPUT, i});
    }

    for (uint32_t i = 0; i < params.output_layouts.size(); i++) {
        args.push_back({ArgumentDescriptor::Types::OUTPUT, i});
    }

    return args;
}

}  // namespace ov::intel_gpu::spirv

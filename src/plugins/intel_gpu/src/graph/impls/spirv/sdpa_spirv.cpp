// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdpa_spirv.hpp"
#include "utils/primitive_spirv_base.hpp"
#include "kernels/sdpa_sycl_kernel.spv.inc"

#include <string>

namespace ov::intel_gpu::spirv { 
std::string SDPASplitGenerator::get_spirv() const {
    return std::string(reinterpret_cast<char const*>(sdpa_sycl_kernel_spv), sdpa_sycl_kernel_spv_len);
}

Arguments SDPASplitGenerator::get_arguments_desc(const RuntimeParams& params) const {
    Arguments args;
    auto query_arg = argument_desc{ArgumentDescriptor::Types::INPUT, 0};
    auto key_arg = argument_desc{ArgumentDescriptor::Types::INPUT, 1};
    auto value_arg = argument_desc{ArgumentDescriptor::Types::INPUT, 2};
    auto mask_arg = argument_desc{ArgumentDescriptor::Types::INPUT, 3};
    auto partial_out_arg = argument_desc{ArgumentDescriptor::Types::INTERNAL_BUFFER, 0};
    auto partial_m_arg = argument_desc{ArgumentDescriptor::Types::INTERNAL_BUFFER, 1};
    auto partial_l_arg = argument_desc{ArgumentDescriptor::Types::INTERNAL_BUFFER, 2};
    auto scale_arg = argument_desc{ArgumentDescriptor::Types::SCALAR, 0};
    auto mask_mode_arg = argument_desc{ArgumentDescriptor::Types::SCALAR, 1};
    return {query_arg, key_arg, value_arg, mask_arg, partial_out_arg, partial_m_arg, partial_l_arg, scale_arg, mask_mode_arg};
}

DispatchDataFunc SDPASplitGenerator::get_dispatch_data_func() const {
    return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            auto& wgs = kd.params.workGroups;
            size_t H = 32;
            size_t WG = 128;
            size_t SPLITS = 4;
            wgs.global = {H * WG, SPLITS, 1};
            wgs.local = {WG, 1, 1};

            auto& scalars = kd.params.scalars;
            scalars.clear();
            scalar_desc scale_desc;
            scale_desc.t = scalar_desc::Types::FLOAT32;
            scale_desc.v.f32 = 1.0f; // HARDCODED FOR TESTING
            scalar_desc mask_mode_desc;
            mask_mode_desc.t = scalar_desc::Types::INT32;
            mask_mode_desc.v.s32 = 0; // HARDCODED FOR TESTING
            scalars.push_back(scale_desc);
            scalars.push_back(mask_mode_desc);
    }};
}

std::string SDPAReduceGenerator::get_spirv() const {
    return std::string(reinterpret_cast<char const*>(sdpa_sycl_kernel_spv), sdpa_sycl_kernel_spv_len);
}
Arguments SDPAReduceGenerator::get_arguments_desc(const RuntimeParams& params) const {
    Arguments args;
    auto partial_out_arg = argument_desc{ArgumentDescriptor::Types::INTERNAL_BUFFER, 0};
    auto partial_m_arg = argument_desc{ArgumentDescriptor::Types::INTERNAL_BUFFER, 1};
    auto partial_l_arg = argument_desc{ArgumentDescriptor::Types::INTERNAL_BUFFER, 2};
    auto out_arg = argument_desc{ArgumentDescriptor::Types::OUTPUT, 0};
    return {partial_out_arg, partial_m_arg, partial_l_arg, out_arg};
}
DispatchDataFunc SDPAReduceGenerator::get_dispatch_data_func() const {

    return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            auto& wgs = kd.params.workGroups;
            size_t H = 32;
            size_t WG = 128;
            wgs.global = {H * WG, 1, 1};
            wgs.local = {WG, 1, 1};
    }};
}

class SDPAImpl : public PrimitiveImplSPIRV {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::spirv::SDPAImpl)
    Stage::Ptr sdpa_split = make_stage<SDPASplitGenerator>();
    Stage::Ptr sdpa_reduce = make_stage<SDPAReduceGenerator>();

SDPAImpl() : PrimitiveImplSPIRV(SDPAImplManager::get_type_info_static()) {}
SDPAImpl(const program_node& node, const RuntimeParams& params) : SDPAImpl() {
        add_stage(sdpa_split, params);
        add_stage(sdpa_reduce, params);
}
[[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<SDPAImpl>(this);
}

[[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const RuntimeParams& params) const override {
        // Hardcode buffer sizes as the kernel supports only one shape
        auto partial_out = BufferDescriptor{32 * 4 * 128, ov::element::f32};
        auto partial_m = BufferDescriptor{32 * 4, ov::element::f32};
        auto partial_l = BufferDescriptor{32 * 4, ov::element::f32};
        std::vector<BufferDescriptor> buffers = {partial_out, partial_m, partial_l};
        return buffers;
}
};

std::unique_ptr<primitive_impl> SDPAImplManager::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<scaled_dot_product_attention>());
    return std::make_unique<SDPAImpl>(node, params);
}
}

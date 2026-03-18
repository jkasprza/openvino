// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdpa_kf.hpp"
#include "utils/primitive_spirv_base.hpp"
#include "kernels/sdpa_kf.spv.inc"

#include <string>
#include <memory>

// Set up constants as this kernel support only single shape
constexpr size_t H = 32;
constexpr size_t WG = 128;
constexpr size_t SPLITS = 4;
constexpr int64_t D = 128;

namespace ov::intel_gpu::spirv::kf { 
std::string SDPASplitGenerator::get_spirv() const {
    return std::string(reinterpret_cast<char const*>(sdpa_sycl_kernel_spv), sdpa_sycl_kernel_spv_len);
}

Arguments SDPASplitGenerator::get_arguments_desc(const RuntimeParams& params) const {
    auto desc = std::dynamic_pointer_cast<const scaled_dot_product_attention>(params.desc);
    OPENVINO_ASSERT(desc, "Unexpected descriptor type for SDPA primitive");
    Arguments args;
    auto query_arg = argument_desc{ArgumentDescriptor::Types::INPUT, 0};
    auto key_arg = argument_desc{ArgumentDescriptor::Types::INPUT, 1};
    auto value_arg = argument_desc{ArgumentDescriptor::Types::INPUT, 2};
    auto mask_arg = argument_desc{ArgumentDescriptor::Types::INPUT, 3};
    if (!desc->has_attn_mask_input) {
        // Set to nullptr by using scalar argument
        mask_arg = argument_desc{ArgumentDescriptor::Types::SCALAR, 0};
    }
    auto partial_out_arg = argument_desc{ArgumentDescriptor::Types::INTERNAL_BUFFER, 0};
    auto partial_m_arg = argument_desc{ArgumentDescriptor::Types::INTERNAL_BUFFER, 1};
    auto partial_l_arg = argument_desc{ArgumentDescriptor::Types::INTERNAL_BUFFER, 2};
    auto scale_arg = argument_desc{ArgumentDescriptor::Types::SCALAR, 1};
    auto mask_mode_arg = argument_desc{ArgumentDescriptor::Types::SCALAR, 2};
    auto smem_out_arg = argument_desc{ArgumentDescriptor::Types::LOCAL_MEMORY_SIZE, 0};
    auto smem_m_arg = argument_desc{ArgumentDescriptor::Types::LOCAL_MEMORY_SIZE, 1};
    auto smem_l_arg = argument_desc{ArgumentDescriptor::Types::LOCAL_MEMORY_SIZE, 2};
    return {query_arg, key_arg, value_arg, mask_arg, partial_out_arg, partial_m_arg, partial_l_arg, scale_arg, mask_mode_arg, smem_out_arg, smem_m_arg, smem_l_arg};
}

DispatchDataFunc SDPASplitGenerator::get_dispatch_data_func() const {
    return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            auto desc = std::dynamic_pointer_cast<const scaled_dot_product_attention>(params.desc);
            OPENVINO_ASSERT(desc, "Unexpected descriptor type for SDPA primitive");
            auto& wgs = kd.params.workGroups;
            wgs.global = {SPLITS, H * WG, 1};
            wgs.local = {1, WG, 1};

            auto& scalars = kd.params.scalars;
            scalars.clear();
            scalar_desc nullptr_desc;
            nullptr_desc.t = scalar_desc::Types::UINT64;
            nullptr_desc.v.u64 = 0;
            scalar_desc scale_desc;
            scale_desc.t = scalar_desc::Types::FLOAT32;
            scale_desc.v.f32 = 1.0f;
            if (desc->scale_val.has_value()) {
                scale_desc.v.f32 = desc->scale_val.value();
            }
            scalar_desc mask_mode_desc;
            mask_mode_desc.t = scalar_desc::Types::INT32;
            mask_mode_desc.v.s32 = 0;
            if (desc->has_attn_mask_input) {
                auto mask_rank = params.input_layouts[3].get_partial_shape().rank().get_length();
                if (mask_rank == 4) {
                    mask_mode_desc.v.s32 = 2;
                } else if (mask_rank == 1) {
                    mask_mode_desc.v.s32 = 1;
                } else {
                    OPENVINO_THROW("Unexpected attention mask layout");
                }
            }
            scalars.push_back(nullptr_desc);
            scalars.push_back(scale_desc);
            scalars.push_back(mask_mode_desc);

            auto& local_mem = kd.params.local_memory_args;
            local_mem.clear();
            local_mem.push_back(8 * 128 * sizeof(float));
            local_mem.push_back(8 * sizeof(float));
            local_mem.push_back(8 * sizeof(float));
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
    auto smem_w = argument_desc{ArgumentDescriptor::Types::LOCAL_MEMORY_SIZE, 0};
    return {partial_out_arg, partial_m_arg, partial_l_arg, out_arg, smem_w};
}
DispatchDataFunc SDPAReduceGenerator::get_dispatch_data_func() const {

    return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
            auto& wgs = kd.params.workGroups;
            wgs.global = {H * WG, 1, 1};
            wgs.local = {WG, 1, 1};

            auto& local_mem = kd.params.local_memory_args;
            local_mem.clear();
            local_mem.push_back((SPLITS + 1) * sizeof(float));
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
        auto partial_out = BufferDescriptor{H * SPLITS * D, ov::element::f32};
        auto partial_m = BufferDescriptor{H * SPLITS, ov::element::f32};
        auto partial_l = BufferDescriptor{H * SPLITS, ov::element::f32};
        std::vector<BufferDescriptor> buffers = {partial_out, partial_m, partial_l};
        return buffers;
}
};

std::unique_ptr<primitive_impl> SDPAImplManager::create_impl(const program_node& node, const RuntimeParams& params) const {
    assert(node.is_type<scaled_dot_product_attention>());
    return std::make_unique<SDPAImpl>(node, params);
}
}

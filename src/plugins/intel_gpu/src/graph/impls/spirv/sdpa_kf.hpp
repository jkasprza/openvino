// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "program_node.h"
#include "utils/kernel_generator.hpp"
#include "registry/implementation_manager.hpp"
#include "scaled_dot_product_attention_inst.h"

using namespace cldnn;

namespace ov::intel_gpu::spirv::kf { 
struct SDPASplitGenerator : public KernelGenerator {
    SDPASplitGenerator() : KernelGenerator("_ZTSZZ36run_sdpa_b1h32_lq1_lkv5120_d128_syclRN4sycl3_V15queueEPKNS0_6detail9half_impl4halfES7_S7_S7_PS5_fiENKUlRNS0_7handlerEE_clESA_E25SDPA_Partial_B1H32_SPLIT4") {}
    [[nodiscard]] std::string get_spirv() const override;
    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;

};
struct SDPAReduceGenerator : public KernelGenerator {
    SDPAReduceGenerator() : KernelGenerator("_ZTSZZ36run_sdpa_b1h32_lq1_lkv5120_d128_syclRN4sycl3_V15queueEPKNS0_6detail9half_impl4halfES7_S7_S7_PS5_fiENKUlRNS0_7handlerEE0_clESA_E30SDPA_ReduceSplits_B1H32_SPLIT4") {}
    [[nodiscard]] std::string get_spirv() const override;
    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override;
    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override;
};

struct SDPAImplManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("spirv::sdpa_kf")
    explicit SDPAImplManager(shape_types shape_type, ValidateFunc vf = nullptr) 
        : ImplementationManager(impl_types::spirv, shape_type, std::move(vf)) {}
    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;
    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        const auto desc = node.as<scaled_dot_product_attention>().get_primitive();

        const auto& config = node.get_program().get_config();
        if (!config.get_use_spirv()) {
            return false;
        }

        if (desc->is_causal || desc->has_sink_input || desc->indirect_axis != -1 || desc->is_kv_compressed) {
            return false;
        }

        const auto has_supported_transpose = [](const std::vector<int64_t>& order) {
            static const std::vector<int64_t> identity = {0, 1, 2, 3};
            return order.empty() || order == identity;
        };
        if (!has_supported_transpose(desc->input_q_transpose_order) ||
            !has_supported_transpose(desc->input_k_transpose_order) ||
            !has_supported_transpose(desc->input_v_transpose_order) ||
            !has_supported_transpose(desc->output_transpose_order)) {
            return false;
        }

        const auto& q_layout = node.get_input_layout(ScaledDotProductAttentionInputIdx::QUERY);
        const auto& k_layout = node.get_input_layout(ScaledDotProductAttentionInputIdx::KEY);
        const auto& v_layout = node.get_input_layout(ScaledDotProductAttentionInputIdx::VALUE);
        const auto& out_layout = node.get_output_layout(0);

        if (!everyone_is(format::bfyx, q_layout.format, k_layout.format, v_layout.format, out_layout.format)) {
            return false;
        }

        if (!everyone_is(ov::element::f16, q_layout.data_type, k_layout.data_type, v_layout.data_type, out_layout.data_type)) {
            return false;
        }

        const auto& q_shape = q_layout.get_partial_shape();
        const auto& k_shape = k_layout.get_partial_shape();
        const auto& v_shape = v_layout.get_partial_shape();
        const auto& out_shape = out_layout.get_partial_shape();
        if (q_shape.rank().is_dynamic() || k_shape.rank().is_dynamic() || v_shape.rank().is_dynamic() || out_shape.rank().is_dynamic()) {
            return false;
        }

        if (q_shape.rank().get_length() != 4 || k_shape.rank().get_length() != 4 ||
            v_shape.rank().get_length() != 4 || out_shape.rank().get_length() != 4) {
            return false;
        }

        for (size_t i = 0; i < 4; i++) {
            if (q_shape[i].is_dynamic() || k_shape[i].is_dynamic() || v_shape[i].is_dynamic() || out_shape[i].is_dynamic()) {
                return false;
            }
        }

        constexpr int64_t B = 1;
        constexpr int64_t H = 32;
        constexpr int64_t Lq = 1;
        constexpr int64_t D = 128;
        constexpr int64_t Lkv = 5120;

        if (q_shape[0].get_length() != B || q_shape[1].get_length() != H || q_shape[2].get_length() != Lq || q_shape[3].get_length() != D) {
            return false;
        }

        if (k_shape[0].get_length() != B || k_shape[1].get_length() != H || k_shape[2].get_length() != Lkv || k_shape[3].get_length() != D) {
            return false;
        }

        if (v_shape[0].get_length() != B || v_shape[1].get_length() != H || v_shape[2].get_length() != Lkv || v_shape[3].get_length() != D) {
            return false;
        }

        if (out_shape[0].get_length() != B || out_shape[1].get_length() != H || out_shape[2].get_length() != Lq || out_shape[3].get_length() != D) {
            return false;
        }

        if (desc->has_attn_mask_input) {
            const auto& mask_layout = node.get_input_layout(ScaledDotProductAttentionInputIdx::ATTN_MASK);
            const auto& mask_shape = mask_layout.get_partial_shape();

            if (mask_layout.data_type != ov::element::f16 || mask_shape.rank().is_dynamic()) {
                return false;
            }

            const auto mask_rank = mask_shape.rank().get_length();
            if (mask_rank == 1) {
                if (mask_shape[0].is_dynamic() || mask_shape[0].get_length() != Lkv) {
                    return false;
                }
            } else if (mask_rank == 4) {
                if (mask_shape[0].is_dynamic() || mask_shape[1].is_dynamic() ||
                    mask_shape[2].is_dynamic() || mask_shape[3].is_dynamic()) {
                    return false;
                }
                if (mask_shape[0].get_length() != B || mask_shape[1].get_length() != 1 ||
                    mask_shape[2].get_length() != Lq || mask_shape[3].get_length() != Lkv) {
                    return false;
                }
            } else {
                return false;
            }
        }

        return true;
    }
};
}

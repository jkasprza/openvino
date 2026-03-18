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

namespace ov::intel_gpu::spirv { 
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
/*
torch::Tensor forward(
    const torch::Tensor& queries,
    const torch::Tensor& keys,
    const torch::Tensor& values,
    const torch::Tensor& mask,
    float scale) {

  TORCH_CHECK(queries.device().type() == c10::DeviceType::XPU, "queries must be on an XPU device");
  TORCH_CHECK(keys.device().type() == c10::DeviceType::XPU, "keys must be on an XPU device");
  TORCH_CHECK(values.device().type() == c10::DeviceType::XPU, "values must be on an XPU device");

  TORCH_CHECK(queries.scalar_type() == torch::kFloat16, "queries must be float16");
  TORCH_CHECK(keys.scalar_type() == torch::kFloat16, "keys must be float16");
  TORCH_CHECK(values.scalar_type() == torch::kFloat16, "values must be float16");
  TORCH_CHECK(!mask.defined() || mask.scalar_type() == torch::kFloat16, "mask must be float16 when provided");

  TORCH_CHECK(queries.is_contiguous(), "queries must be contiguous");
  TORCH_CHECK(keys.is_contiguous(), "keys must be contiguous");
  TORCH_CHECK(values.is_contiguous(), "values must be contiguous");
  TORCH_CHECK(!mask.defined() || mask.is_contiguous(), "mask must be contiguous when provided");

  TORCH_CHECK(queries.dim() == 4, "queries must have shape [B,H,Lq,D]");
  TORCH_CHECK(keys.dim() == 4, "keys must have shape [B,H,Lkv,D]");
  TORCH_CHECK(values.dim() == 4, "values must have shape [B,H,Lkv,D]");

  const int64_t B   = queries.size(0);
  const int64_t H   = queries.size(1);
  const int64_t Lq  = queries.size(2);
  const int64_t D   = queries.size(3);
  const int64_t Lkv = keys.size(2);

  TORCH_CHECK(keys.size(0) == B && values.size(0) == B, "batch mismatch");
  TORCH_CHECK(keys.size(1) == H && values.size(1) == H, "num_heads mismatch");
  TORCH_CHECK(keys.size(2) == Lkv && values.size(2) == Lkv, "key/value sequence length mismatch");
  TORCH_CHECK(keys.size(3) == D && values.size(3) == D, "head dimension mismatch");

  TORCH_CHECK(B == 1, "This optimized kernel targets batch=1");
  TORCH_CHECK(H == 32, "This optimized kernel targets num_heads=32");
  TORCH_CHECK(Lq == 1, "This optimized kernel targets seq_len_q=1");
  TORCH_CHECK(D == 128, "This optimized kernel targets head_dim=128");
  TORCH_CHECK(Lkv == 5120, "This optimized kernel targets seq_len_kv=5120");

  int mask_mode = 0;
  const bool use_mask = (mask.defined() && mask.numel() > 0);
  if (use_mask) {
    if (mask.numel() == Lkv) {
      mask_mode = 1;
    } else if (mask.numel() == (B * 1 * Lq * Lkv)) {
      mask_mode = 2;
    } else {
      TORCH_CHECK(false, "mask must be [Lkv] or [B,1,Lq,Lkv] for this kernel");
    }
  }

  torch::Tensor output = torch::empty_like(queries);

  const sycl::half* q_ptr = reinterpret_cast<const sycl::half*>(queries.data_ptr<c10::Half>());
  const sycl::half* k_ptr = reinterpret_cast<const sycl::half*>(keys.data_ptr<c10::Half>());
  const sycl::half* v_ptr = reinterpret_cast<const sycl::half*>(values.data_ptr<c10::Half>());
  const sycl::half* m_ptr = use_mask ? reinterpret_cast<const sycl::half*>(mask.data_ptr<c10::Half>()) : nullptr;
  sycl::half* o_ptr       = reinterpret_cast<sycl::half*>(output.data_ptr<c10::Half>());

  sycl::queue& q = c10::xpu::getCurrentXPUStream().queue();

  // Choose SPLITS for Lunar Lake / Arc 140V:
  // SPLITS=4 increases WGs from 32->128, usually improving latency hiding without too much reduction overhead.
  constexpr int SPLITS = 4;
  constexpr size_t WG = 128;

  // Temporary partial buffers (float) on XPU
  auto fopts = torch::TensorOptions().device(queries.device()).dtype(torch::kFloat32);
  torch::Tensor partial_out = torch::empty({H, SPLITS, D}, fopts); // [32,4,128]
  torch::Tensor partial_m   = torch::empty({H, SPLITS}, fopts);    // [32,4]
  torch::Tensor partial_l   = torch::empty({H, SPLITS}, fopts);    // [32,4]

  float* partial_out_ptr = partial_out.data_ptr<float>();
  float* partial_m_ptr   = partial_m.data_ptr<float>();
  float* partial_l_ptr   = partial_l.data_ptr<float>();

  // Kernel 1 launch: 2D, groups = [H, SPLITS], local = [WG,1]
  sycl::nd_range<2> launch1(
      sycl::range<2>((size_t)H * WG, (size_t)SPLITS),
      sycl::range<2>(WG, 1));

  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> smem_out(sycl::range<1>(8 * 128), cgh);
    sycl::local_accessor<float, 1> smem_m(sycl::range<1>(8), cgh);
    sycl::local_accessor<float, 1> smem_l(sycl::range<1>(8), cgh);

    SDPA_Partial_B1H32_Lq1_Lkv5120_D128_WG128_SG16<SPLITS> functor{
        q_ptr, k_ptr, v_ptr, m_ptr,
        partial_out_ptr, partial_m_ptr, partial_l_ptr,
        scale, mask_mode,
        smem_out, smem_m, smem_l
    };

    cgh.parallel_for<class SDPA_Partial_B1H32_SPLIT4>(launch1, functor);
  });

  // Kernel 2 launch: 1D, one WG per head
  // Need local memory of SPLITS+1 floats (we store l_total at [SPLITS]).
  sycl::nd_range<1> launch2(
      sycl::range<1>((size_t)H * WG),
      sycl::range<1>(WG));

  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> smem_w(sycl::range<1>(SPLITS + 1), cgh);

    SDPA_ReduceSplits_B1H32_Lq1_D128_WG128_SG16<SPLITS> functor{
        partial_out_ptr,
        partial_m_ptr,
        partial_l_ptr,
        o_ptr,
        smem_w
    };

    cgh.parallel_for<class SDPA_ReduceSplits_B1H32_SPLIT4>(launch2, functor);
  });

  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward,
        "Scaled dot product attention (SYCL 2020, fixed shapes B1H32Lq1Lkv5120D128, KV-split online softmax + stable reduction, WG=128 SG=16)");
}*/

struct SDPAImplManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("spirv::sdpa")
    explicit SDPAImplManager(shape_types shape_type, ValidateFunc vf = nullptr) 
        : ImplementationManager(impl_types::spirv, shape_type, std::move(vf)) {}
    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;
    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        const auto desc = node.as<scaled_dot_product_attention>().get_primitive();

        const auto& config = node.get_program().get_config();
        if (!config.get_use_spirv()) {
            return false;
        }

        // Not sure about those but better assume they are not supported
        if (desc->is_causal || desc->has_sink_input || desc->indirect_axis != -1 || desc->is_kv_compressed) {
            return false;
        }

        const auto has_supported_transpose = [](const std::vector<int64_t>& order) {
            static const std::vector<int64_t> identity = {0, 1, 2, 3};
            return order.empty() || order == identity;
        };
        // Not sure if transpose orders need to be checked
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

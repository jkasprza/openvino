// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/kernel.hpp"
#include "openvino/core/except.hpp"
#include "ze_common.hpp"
#include "ze_kernel_holder.hpp"

#include <memory>

namespace cldnn {
namespace ze {

class ze_kernel : public kernel {
public:
    /// @brief Create L0 kernel object for every entry point in the module.
    /// Ignore special "Intel_Symbol_Table_Void_Program" entry point.
    /// @param module Input module to create kernels from
    /// @param out Vector for appending resulting kernels 
    static void create_kernels_from_module(std::shared_ptr<ze_module_holder> module, std::vector<kernel::ptr> &out);

    ze_kernel(std::shared_ptr<ze_kernel_holder> kernel, const std::string& kernel_id);

    ze_kernel_handle_t get_kernel_handle() const;
    ze_module_handle_t get_module_handle() const;

    std::string get_id() const override;
    std::shared_ptr<kernel> clone(bool reuse_kernel_handle = false) const override;
    void set_arguments(const kernel_arguments_desc& args_desc, const kernel_arguments_data& args) override;
    std::vector<uint8_t> get_binary() const override;
    std::string get_build_log() const override;

private:
    std::shared_ptr<ze_kernel_holder> m_kernel;
    std::string m_kernel_id;
};

}  // namespace ze
}  // namespace cldnn

// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/command_list.hpp"
#include "ze_base_event_factory.hpp"
#include "ze_stream.hpp"
#include "ze_common.hpp"
#include "ze_engine.hpp"
#include "ze_event.hpp"

namespace cldnn {
namespace ze {

class ze_command_list : public command_list {
public:

    ze_command_list(ze_stream &stream);
    ~ze_command_list();

    void close_impl() override;
    void reset_impl() override;

    event::ptr append_kernel_launch(kernel& k,
        const kernel_arguments_desc& args_desc,
        const kernel_arguments_data& args,
        const std::vector<event::ptr>& events,
        bool needs_out_event) override;
    ze_command_list_handle_t get_handle() const { return m_command_list; }

private:

    void reset();

    ze_stream &m_stream;
    ze_command_list_handle_t m_command_list = nullptr;
    uint64_t m_last_barrier = 0;
};

}  // namespace ze
}  // namespace cldnn

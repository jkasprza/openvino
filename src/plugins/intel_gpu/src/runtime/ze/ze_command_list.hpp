// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/command_list.hpp"
#include "ze_base_event_factory.hpp"
#include "ze_common.hpp"
#include "ze_engine.hpp"
#include "ze_event.hpp"

namespace cldnn {
namespace ze {

class ze_command_list : public command_list {
public:

    ze_command_list(const ze_engine& engine, std::shared_ptr<ze_base_event_factory> ev_factory, QueueTypes queue_type);
    ~ze_command_list();

    void close_impl() override;
    void reset_impl() override;

    void append_kernel_launch(kernel& k, const kernel_arguments_desc& args_desc, const kernel_arguments_data& args, const std::vector<event::ptr>& events, event::ptr out_event) override;
    ze_command_list_handle_t get_handle() const { return m_command_list; }

private:

    void reset();

    const ze_engine& m_engine;
    ze_command_list_handle_t m_command_list = nullptr;
    std::shared_ptr<ze_base_event_factory> m_event_factory;
};

}  // namespace ze
}  // namespace cldnn

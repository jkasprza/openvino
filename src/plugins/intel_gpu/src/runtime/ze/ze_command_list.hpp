// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/command_list.hpp"
#include "ze/ze_base_event_factory.hpp"
#include "ze_common.hpp"
#include "ze_engine.hpp"
#include "ze_event.hpp"

namespace cldnn {



namespace ze {
class ze_command_list : public command_list {
public:

    ze_command_list(const ze_engine& engine);
    ~ze_command_list();

    void start() override;
    void close() override;

    event::ptr add_kernel(kernel& k, const kernel_arguments_desc& args_desc, const kernel_arguments_data& args, std::vector<event::ptr> const& deps) override;
    ze_command_list_handle_t get_handle() const { return m_command_list; }

    event::ptr get_output_event() const override;

private:

    void reset();
    uint64_t get_command_id();

    const ze_engine& m_engine;
    ze_command_list_handle_t m_command_list = nullptr;
    event::ptr m_output_event = nullptr;
    std::shared_ptr<ze_base_event_factory> m_event_factory;
    // mutable std::atomic<uint64_t> m_queue_counter{0};
    // std::atomic<uint64_t> m_last_barrier{0};
    // std::shared_ptr<ze_event> m_last_barrier_ev = nullptr;
    // ze_events_pool m_pool;
};

}  // namespace ze
}  // namespace cldnn
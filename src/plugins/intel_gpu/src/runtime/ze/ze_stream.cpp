// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_stream.hpp"
#include "intel_gpu/runtime/memory_caps.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/properties.hpp"

#include "ze_counter_based_event_factory.hpp"
#include "ze_event_factory.hpp"
#include "ze_events.hpp"
#include "ze_empty_event.hpp"
#include "ze_command_list.hpp"

#include "ze_event.hpp"
#include "ze_kernel.hpp"
#include "ze_memory.hpp"
#include "ze_common.hpp"

#include <ze_api.h>
#include <ze_intel_gpu.h>
#include <ze_stypes.h>

#include <cassert>
#include <string>
#include <vector>
#include <memory>

#ifdef ENABLE_ONEDNN_FOR_GPU
#include <oneapi/dnnl/dnnl_l0.hpp>
#endif

namespace cldnn {
namespace ze {

ze_stream::ze_stream(const ze_engine &engine, const ExecutionConfig& config)
    : stream(config.get_queue_type(), stream::get_expected_sync_method(config))
    , _engine(engine) {
    const auto &info = engine.get_device_info();

    ze_command_queue_desc_t command_queue_desc = {};
    command_queue_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    command_queue_desc.pNext = nullptr;
    command_queue_desc.index = 0;
    command_queue_desc.ordinal = info.compute_queue_group_ordinal;
    command_queue_desc.flags = m_queue_type == QueueTypes::out_of_order ? 0 : ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
    command_queue_desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    command_queue_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;

    zex_intel_queue_copy_operations_offload_hint_exp_desc_t cp_offload_desc = {};
    cp_offload_desc.stype = ZEX_INTEL_STRUCTURE_TYPE_QUEUE_COPY_OPERATIONS_OFFLOAD_HINT_EXP_PROPERTIES;
    cp_offload_desc.copyOffloadEnabled = true;
    cp_offload_desc.pNext = nullptr;
    bool use_cp_offload = info.supports_cp_offload;
    if (use_cp_offload) {
        command_queue_desc.pNext = &cp_offload_desc;
    }

    OV_ZE_EXPECT(zeCommandListCreateImmediate(_engine.get_context(), _engine.get_device(), &command_queue_desc, &m_command_list));
    bool use_counter_based_events = m_queue_type == QueueTypes::in_order && info.supports_counter_based_events;
    if (use_counter_based_events) {
        m_ev_factory = std::make_shared<ze_counter_based_event_factory>(engine, config.get_enable_profiling());
    } else {
        m_ev_factory = std::make_shared<ze_event_factory>(engine, config.get_enable_profiling());
    }
    GPU_DEBUG_INFO << "[GPU] Created L0 stream ("
        << "use_cp_offload=" << use_cp_offload
        << ", use_counter_based_events=" << use_counter_based_events
        << ")" << std::endl;
}

ze_stream::~ze_stream() {
#ifdef ENABLE_ONEDNN_FOR_GPU
    // Destroy OneDNN stream before destroying command list
    _onednn_stream.reset();
#endif
    if (m_command_list != nullptr)
        zeCommandListDestroy(m_command_list);
}

void ze_stream::set_arguments(kernel& kernel, const kernel_arguments_desc& args_desc, const kernel_arguments_data& args) {
    auto& ze_kernel = downcast<ze::ze_kernel>(kernel);
    ze_kernel.set_arguments(args_desc, args);
}

event::ptr ze_stream::enqueue_kernel(kernel& kernel,
                                     const kernel_arguments_desc& args_desc,
                                     const kernel_arguments_data& args_data,
                                     std::vector<event::ptr> const& deps,
                                     bool is_output) {
    auto& ze_kernel = downcast<ze::ze_kernel>(kernel);
    const std::vector<event::ptr> empty_events;

    const std::vector<event::ptr> *dep_events = &empty_events;
    if (m_sync_method == SyncMethods::events) {
        dep_events = &deps;
    } else if (m_sync_method == SyncMethods::barriers) {
        sync_events(deps, is_output);
    }
    bool set_output_event = m_sync_method == SyncMethods::events || is_output;
    auto ev = set_output_event ? create_base_event() : std::make_shared<ze_empty_event>(++m_queue_counter);
    ze_kernel.launch(m_command_list, args_desc, args_data, *dep_events, ev);

    return ev;
}

void ze_stream::enqueue_barrier() {
    OV_ZE_EXPECT(zeCommandListAppendBarrier(m_command_list, nullptr, 0, nullptr));
}

event::ptr ze_stream::enqueue_marker(std::vector<ze_event::ptr> const& deps, bool is_output) {
    if (deps.empty()) {
        auto ev = create_base_event();
        OV_ZE_EXPECT(zeCommandListAppendBarrier(m_command_list, std::dynamic_pointer_cast<ze_base_event>(ev)->get_handle(), 0, nullptr));
        return ev;
    }

    if (m_sync_method  == SyncMethods::events) {
        std::vector<ze_event_handle_t> dep_events;
        for (auto& dep : deps) {
            if (auto ze_base_ev = std::dynamic_pointer_cast<ze_base_event>(dep)) {
                if (ze_base_ev->get_handle() != nullptr)
                    dep_events.push_back(ze_base_ev->get_handle());
            }
        }
        if (dep_events.empty())
            return create_user_event(true);

        auto ev = create_base_event();
        OV_ZE_EXPECT(zeCommandListAppendBarrier(m_command_list,
                                            std::dynamic_pointer_cast<ze_base_event>(ev)->get_handle(),
                                            static_cast<uint32_t>(dep_events.size()),
                                            &dep_events.front()));
        return ev;
    } else if (m_sync_method == SyncMethods::barriers) {
        sync_events(deps, is_output);
        assert(m_last_barrier_ev != nullptr);
        return m_last_barrier_ev;
    } else {
        return create_user_event(true);
    }
}

ze_event::ptr ze_stream::group_events(std::vector<ze_event::ptr> const& deps) {
    return std::make_shared<ze_events>(deps, _engine);
}

void ze_stream::wait() {
    finish();
}

event::ptr ze_stream::create_user_event(bool set) {
    auto ev = m_ev_factory->create_event(++m_queue_counter);
    if (set)
        ev->set();

    return ev;
}

event::ptr ze_stream::create_base_event() {
    return m_ev_factory->create_event(++m_queue_counter);
}

std::unique_ptr<surfaces_lock> ze_stream::create_surfaces_lock(const std::vector<memory::ptr> &mem) const {
    // Level Zero egnine currently does not support surfaces lock
    return nullptr;
}

void ze_stream::flush() const {
    // Immediate Command List submits commands immediately - no flush impl
}

void ze_stream::finish() const {
    OV_ZE_EXPECT(zeCommandListHostSynchronize(m_command_list, endless_wait));
}

void ze_stream::wait_for_events(const std::vector<event::ptr>& events) {
    bool needs_sync = false;
    for (auto& ev : events) {
        auto* ze_base_ev = dynamic_cast<ze_base_event*>(ev.get());
        if (ze_base_ev->get_handle() != nullptr) {
            ze_base_ev->wait();
        } else {
            needs_sync = true;
        }
        // Block thread and wait for event signal
        ev->wait();
    }

    if (needs_sync) {
        finish();
    }
}

void ze_stream::sync_events(std::vector<event::ptr> const& deps, bool is_output) {
    bool needs_barrier = false;
    for (auto& dep : deps) {
        auto* ze_base_ev = dynamic_cast<ze_base_event*>(dep.get());
        assert(ze_base_ev != nullptr);
        if (ze_base_ev->get_queue_stamp() > m_last_barrier) {
            needs_barrier = true;
        }
    }

    if (needs_barrier) {
        if (is_output) {
            m_last_barrier_ev = std::dynamic_pointer_cast<ze_event>(create_base_event());
            m_last_barrier_ev->set_queue_stamp(m_queue_counter.load());
            OV_ZE_EXPECT(zeCommandListAppendBarrier(m_command_list, m_last_barrier_ev->get_handle(), 0, nullptr));
        } else {
            OV_ZE_EXPECT(zeCommandListAppendBarrier(m_command_list, nullptr, 0, nullptr));
        }
        m_last_barrier = ++m_queue_counter;
    }

    if (!m_last_barrier_ev) {
        m_last_barrier_ev = std::dynamic_pointer_cast<ze_event>(create_user_event(true));
        m_last_barrier_ev->set_queue_stamp(m_queue_counter.load());
    }
}

#ifdef ENABLE_ONEDNN_FOR_GPU
dnnl::stream& ze_stream::get_onednn_stream() {
    OPENVINO_ASSERT(m_queue_type == QueueTypes::in_order, "[GPU] Can't create onednn stream handle as onednn doesn't support out-of-order queue");
    OPENVINO_ASSERT(_engine.get_device_info().vendor_id == INTEL_VENDOR_ID, "[GPU] Can't create onednn stream handle as for non-Intel devices");
    if (!_onednn_stream) {
        _onednn_stream = std::make_shared<dnnl::stream>(dnnl::l0_interop::make_stream(_engine.get_onednn_engine(), m_command_list, m_ev_factory->is_profiling_enabled()));
    }

    return *_onednn_stream;
}
#endif

command_list::ptr ze_stream::create_cmd_list() {
    return std::make_shared<ze_command_list>(*this);
}

event::ptr ze_stream::enqueue_cmd_list(const command_list& cmd_list, bool need_output_event = false) {
    OPENVINO_ASSERT(_engine.get_device_info().supports_immediate_cmd_list_append, "[GPU] Command list enqueue is not supported");
    auto ze_list = dynamic_cast<const ze_command_list*>(&cmd_list);
    OPENVINO_ASSERT(ze_list != nullptr, "[GPU] Unexpected command list type");
    uint32_t cmd_list_count = 1;
    ze_command_list_handle_t handle = ze_list->get_handle();
    bool set_output_event = m_sync_method == SyncMethods::events || need_output_event;
    auto ev = set_output_event ? create_base_event() : std::make_shared<ze_empty_event>(++m_queue_counter);
    auto ev_handle = std::dynamic_pointer_cast<ze_base_event>(ev)->get_handle();
    OV_ZE_EXPECT(zeCommandListImmediateAppendCommandListsExp(
        m_command_list, cmd_list_count, &handle, ev_handle, 0, nullptr));
    return ev;
}

}  // namespace ze
}  // namespace cldnn

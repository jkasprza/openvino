// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/event.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "ze_common.hpp"
#include "ze_resource.hpp"
#include "ze_engine.hpp"
#include "ze_event.hpp"
#include "ze_base_event_factory.hpp"

#include <queue>
#include <unordered_map>

namespace cldnn {
namespace ze {

enum class ze_stream_execution_mode {
    immediate = 0,
    deferred = 1
};

class ze_stream : public stream {
public:
    ze_command_list_handle_t get_queue() const {
        if (is_immediate()) {
            OPENVINO_THROW("[GPU] Immediate execution disabled for this experiment");
            return m_imm_cmd_list.handle();
        }
        if (m_cmd_lists.empty()) {
            add_new_cmd_list();
        }
        return m_cmd_lists.front().cmd_list.handle(); 
    }

    ze_command_list_handle_t get_cp_queue() const {
        return m_imm_cmd_list.handle();
    }

    bool is_immediate() const { return mode == ze_stream_execution_mode::immediate; }
    const ze_engine& get_engine() const { return _engine; }

    ze_stream(const ze_engine& engine, const ExecutionConfig& config);
    ze_stream(const ze_engine& engine, const ExecutionConfig& config, ze_command_list_resource cmd_list);
    ze_stream(ze_stream&& other)
        : stream(other.m_queue_type, other.m_sync_method)
        , _engine(other._engine)
        , m_cmd_lists(std::move(other.m_cmd_lists))
        , m_busy_cmd_lists(std::move(other.m_busy_cmd_lists))
        , m_reuse_cmd_list(std::move(other.m_reuse_cmd_list))
        , m_imm_cmd_list(std::move(other.m_imm_cmd_list))
        , m_queue_counter(other.m_queue_counter.load())
        , m_last_barrier(other.m_last_barrier.load())
        , m_last_barrier_ev(other.m_last_barrier_ev)
        , m_ev_factory(std::move(other.m_ev_factory))
        , m_user_ev_factory(std::move(other.m_user_ev_factory)) {
        }

    ~ze_stream();

    virtual bool can_resubmit() override {
        return m_reuse_cmd_list.has_value();
    }
    virtual void resubmit(const std::vector<event::ptr>& events) override {
        OPENVINO_ASSERT(m_reuse_cmd_list.has_value(), "[GPU] Attempt to resubmit stream without reusable command list");
        static ze_command_list_resource sync_cmd_list;
        if (sync_cmd_list.is_empty()) {
            const auto &info = _engine.get_device_info();
            auto ctx_handle = _engine.get_context().handle();
            auto device_handle = _engine.get_device().handle();

            ze_command_list_handle_t cmd_list_handle = nullptr;
            ze_command_list_flags_t flags = m_queue_type == QueueTypes::out_of_order ? 0 : ZE_COMMAND_LIST_FLAG_IN_ORDER;
            ze_command_list_desc_t command_list_desc = {ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, nullptr, info.compute_queue_group_ordinal, flags};
            OV_ZE_EXPECT(ze::zeCommandListCreate(ctx_handle, device_handle, &command_list_desc, &cmd_list_handle));
            sync_cmd_list = ze_command_list_resource(cmd_list_handle);
        }
        // submit commands in flight
        submit_cmd_list();
        // prepare and submit sync cmd list
        std::vector<ze_event_handle_t> dep_events;
        for (auto& dep : events) {
            if (auto ze_base_ev = std::dynamic_pointer_cast<ze_base_event>(dep)) {
                if (ze_base_ev->get_handle() != nullptr)
                    dep_events.push_back(ze_base_ev->get_handle());
            }
        }
        if (dep_events.size() > 0) {
            OV_ZE_EXPECT(ze::zeCommandListReset(sync_cmd_list.handle()));
            OV_ZE_EXPECT(ze::zeCommandListAppendBarrier(sync_cmd_list.handle(), nullptr, dep_events.size(), dep_events.data()));
            OV_ZE_EXPECT(ze::zeCommandListClose(sync_cmd_list.handle()));
            auto cmd_list_handle = sync_cmd_list.handle();
            OV_ZE_EXPECT(ze::zeCommandQueueExecuteCommandLists(m_cmd_queue.handle(), 1, &cmd_list_handle, nullptr));
        }
        auto cmd_list = m_reuse_cmd_list.value();
        auto cmd_list_handle = cmd_list.cmd_list.handle();
        OV_ZE_EXPECT(ze::zeCommandQueueExecuteCommandLists(m_cmd_queue.handle(), 1, &cmd_list_handle, cmd_list.fence.handle()));
        m_busy_cmd_lists.push(cmd_list);
        m_reuse_cmd_list = std::nullopt;
    }

    void flush() const override;
    void flush_cmd_list() const {
        submit_cmd_list();
    }
    void finish() const override;
    void wait() override;

    void set_arguments(kernel& kernel, const kernel_arguments_desc& args_desc, const kernel_arguments_data& args) override;
    event::ptr enqueue_kernel(kernel& kernel,
                              const kernel_arguments_desc& args_desc,
                              const kernel_arguments_data& args,
                              std::vector<event::ptr> const& deps,
                              bool is_output = false) override;
    event::ptr enqueue_marker(std::vector<event::ptr> const& deps, bool is_output) override;
    event::ptr group_events(std::vector<event::ptr> const& deps) override;
    void wait_for_events(const std::vector<event::ptr>& events) override;
    void enqueue_barrier() override;
    event::ptr create_user_event(bool set) override;
    event::ptr create_base_event() override;
    std::unique_ptr<surfaces_lock> create_surfaces_lock(const std::vector<memory::ptr> &mem) const override;
    ze_context_resource get_context() const;

#ifdef ENABLE_ONEDNN_FOR_GPU
    dnnl::stream& get_onednn_stream() override;
#endif

private:
    void sync_events(std::vector<event::ptr> const& deps, bool is_output = false);
    void add_new_cmd_list() const;
    void submit_cmd_list() const;
    void finish_busy_cmd_lists() const;
    struct command_list {
        ze_command_list_resource cmd_list;
        std::shared_ptr<std::unordered_map<std::string, uint64_t>> cmd_ids;
        ze_fence_resource fence;
#ifdef ENABLE_ONEDNN_FOR_GPU
        std::shared_ptr<dnnl::stream> onednn_stream;
#endif
        ~command_list() {
#ifdef ENABLE_ONEDNN_FOR_GPU
            onednn_stream.reset();
#endif
        }
    };

    const ze_engine& _engine;
    mutable std::queue<command_list> m_cmd_lists;
    mutable std::queue<command_list> m_busy_cmd_lists;
    mutable std::optional<command_list> m_reuse_cmd_list;
    ze_command_list_resource m_imm_cmd_list;
    ze_command_queue_resource m_cmd_queue;
    mutable std::atomic<uint64_t> m_queue_counter{0};
    std::atomic<uint64_t> m_last_barrier{0};
    std::shared_ptr<ze_event> m_last_barrier_ev = nullptr;
    std::shared_ptr<ze_base_event_factory> m_ev_factory;
    std::shared_ptr<ze_base_event_factory> m_user_ev_factory;
    ze_stream_execution_mode mode = ze_stream_execution_mode::deferred;

#ifdef ENABLE_ONEDNN_FOR_GPU
    std::shared_ptr<dnnl::stream> _imm_onednn_stream = nullptr;
#endif
};

}  // namespace ze
}  // namespace cldnn

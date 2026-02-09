// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/graph/network_slice.hpp"
#include "primitive_inst.h"

namespace cldnn {

std::vector<NetworkSlice> NetworkSlice::build_slices(engine& engine, stream::ptr stream, ExecutionOrder& exec_order, size_t max_slice_size) {
    OPENVINO_ASSERT(max_slice_size > 0, "[GPU] Maximum network slice size must be greater than 0");
    std::vector<NetworkSlice> slices;
    auto start = exec_order.begin();
    auto end = start;
    size_t slice_len = 0;
    size_t combined_slices_len = 0;
    while (end != exec_order.end()) {
        auto inst = end->get();
        bool supports_cmd_list = inst->supports_cmd_list();
        bool finish_slice = !supports_cmd_list || (slice_len >= max_slice_size);
        if (finish_slice) {
            if (slice_len > 0) {
                slices.push_back(NetworkSlice(engine, stream, start, end));
                combined_slices_len += slice_len;
            }
            start = end;
            slice_len = 0;
            if (!supports_cmd_list) {
                std::advance(start, 1);
                slice_len -= 1;
            }
        }
        std::advance(end, 1);
        slice_len += 1;
    }
    if (slice_len > 0) {
        slices.push_back(NetworkSlice(engine, stream, start, end));
        combined_slices_len += slice_len;
    }
    GPU_DEBUG_INFO << "Network slices created: " << slices.size()
        << "; Primitives added to network slices: " << combined_slices_len
        << "; Total number of primitives: " << exec_order.size() << std::endl;
    return slices;
}
event::ptr NetworkSlice::run(const std::vector<event::ptr>& dep_events) {
    // Create new cmd list for every run
    m_cmd_list = m_stream->create_cmd_list();
    auto iter = m_start;
    while(iter != m_end) {
        auto inst = iter->get();
        inst->reset_events();
        inst->prepare_primitive();
        inst->add_to_cmd_list(m_cmd_list);
        std::advance(iter, 1);
    }
    m_cmd_list->close();
    //if (!m_cmd_list) {
    //    build_cmd_list();
    //} else {
    //    update_cmd_list();
    //}
    auto ev = m_stream->enqueue_cmd_list(*m_cmd_list, dep_events);
    iter = m_start;
    while(iter != m_end) {
        auto inst = iter->get();
        inst->reset_flags();
        std::advance(iter, 1);
    }
    return ev;
}

void NetworkSlice::build_cmd_list() {
    m_cmd_list = m_stream->create_cmd_list();
    auto iter = m_start;
    while(iter != m_end) {
        auto inst = iter->get();
        inst->add_to_cmd_list(m_cmd_list);
        std::advance(iter, 1);
    }
    m_cmd_list->close();
}

void NetworkSlice::update_cmd_list() {
    auto iter = m_start;
    while(iter != m_end) {
        //auto inst = iter->get();
        bool needs_update = true; //TODO:
        bool can_mutate = false; //TODO:
        if (needs_update) {
            if (!can_mutate) {
                build_cmd_list();
                break;
            }
            //TODO: add mutation
        }
        std::advance(iter, 1);
    }
}

}  // namespace cldnn
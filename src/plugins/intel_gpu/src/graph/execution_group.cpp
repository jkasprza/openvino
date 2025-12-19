// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/graph/execution_group.hpp"
#include "primitive_inst.h"

namespace cldnn {
event::ptr ExecutionGroup::run(const std::vector<event::ptr>& dep_events) {
    mutate_or_rebuild();
    return execute(dep_events);
}

bool ExecutionGroup::prepare_primitives() {
    bool requires_rebuild = false;
    for (size_t i = m_interval.start; i < m_interval.end; i++) {
        bool impl_updated = false;
        m_exec_order[i]->prepare_primitive({}, &impl_updated);

        requires_rebuild |= impl_updated;
    }

    return requires_rebuild;
}

void ExecutionGroup::build_list() {
    m_list = m_stream->create_command_list();
    m_list->start();
    for (size_t i = m_interval.start; i < m_interval.end; i++) {
        m_exec_order[i]->add_to_command_list(m_list.get());
    }
    m_list->close();
}


bool ExecutionGroup::requires_update() {
    return false;
}

void ExecutionGroup::mutate() {
    for (size_t i = m_interval.start; i < m_interval.end; i++) {
        m_exec_order[i]->update_command(m_list.get());
    }
}

void ExecutionGroup::mutate_or_rebuild() {
    auto requires_build = prepare_primitives();
    bool can_mutate = m_list != nullptr && m_list->is_mutable();

    if (requires_build || !can_mutate) {
        build_list();
    } else {
        mutate();
    }
}

event::ptr ExecutionGroup::execute(const std::vector<event::ptr>& dep_events) {
    m_stream->enqueue_barrier();
    auto ev = m_stream->enqueue_command_list(*m_list);
    return m_stream->enqueue_marker({ev});
}

}  // namespace cldnn
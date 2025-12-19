// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/event.hpp"
#include "intel_gpu/runtime/stream.hpp"

#include <vector>
#include <memory>

namespace cldnn {

class primitive_inst;

using ExecutionOrder = std::vector<std::shared_ptr<primitive_inst>>;

// [start, end) interval of nodes from execution order
struct ExecutionInterval {
    size_t start;
    size_t end;
};

struct ExecutionGroup {
    ExecutionInterval m_interval;

    const ExecutionOrder& m_exec_order;
    engine& m_engine;
    stream::ptr m_stream;


    ExecutionGroup(const ExecutionOrder& exec_order, engine& engine, stream::ptr stream, ExecutionInterval interval)
        : m_interval(interval)
        , m_exec_order(exec_order)
        , m_engine(engine)
        , m_stream(stream) {}


    event::ptr run(const std::vector<event::ptr>& dep_events);

private:
    std::shared_ptr<command_list> m_list = nullptr;

    bool prepare_primitives();
    void build_list();
    bool requires_update();
    void mutate_or_rebuild();
    void mutate();
    event::ptr execute(const std::vector<event::ptr>& dep_events);
};

}  // namespace cldnn
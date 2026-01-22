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

using ExecutionOrder = std::list<std::shared_ptr<primitive_inst>>;
using PrimitiveIterator = ExecutionOrder::iterator;

/// @brief Range of primitives from the execution order that can be submitted efficiently by using command list
struct NetworkSlice {
    /// @brief Create slices based on the execution order. Slices are ordered according to exec_order and do not include unsupported primitives
    static std::vector<NetworkSlice> build_slices(engine& engine, stream::ptr stream, ExecutionOrder& exec_order, size_t max_slice_size = SIZE_MAX);
    NetworkSlice(engine& engine, stream::ptr stream, PrimitiveIterator start, PrimitiveIterator end)
        : m_engine(engine)
        , m_stream(stream)
        , m_start(start)
        , m_end(end) {
            // It is assumed that all primitives in the range support command list submission
            OPENVINO_ASSERT(std::distance(m_start, m_end) > 0, "[GPU] NetworkSlice must contain at least 1 primitive");
        }
    /// @brief Get iterator of the primitive that starts the slice
    PrimitiveIterator get_start() { return m_start; }
    /// @brief Get iterator of the primitive that marks the end of the slice.
    /// This primitive is excluded from the slice.
    PrimitiveIterator get_end() { return m_end; }
    /// @brief Get length of the slice
    size_t get_length() { return static_cast<size_t>(std::distance(m_start, m_end)); }

    /// @brief Prepare slice and run it
    event::ptr run(const std::vector<event::ptr>& dep_events);

private:
    engine& m_engine;
    stream::ptr m_stream;
    const PrimitiveIterator m_start;
    const PrimitiveIterator m_end;
    std::shared_ptr<command_list> m_list = nullptr;

    void prepare_primitives();
    bool requires_update();
    void update_cmd_list();
    void build_cmd_list();
};

}  // namespace cldnn

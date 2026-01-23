// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "event.hpp"
#include "kernel.hpp"
#include "kernel_args.hpp"
#include "execution_config.hpp"

#include <memory>
#include <vector>


namespace cldnn {

/// @brief Collection of commands that can be executed
class command_list {
public:
    using ptr = std::shared_ptr<command_list>;

    command_list()
        : m_is_closed(false) {}
    virtual ~command_list() = default;

    bool is_closed() const {
        return m_is_closed;
    }

    void close() {
        OPENVINO_ASSERT(!m_is_closed, "[GPU] Unable to close command list that is already closed");
        close_impl();
        m_is_closed = true;
    }

    void reset() {
        reset_impl();
        m_is_closed = false;
    }

    /// @brief Append kernel launch command to the command list
    /// @param k Kernel to be appended
    /// @param args_desc Kernel argument descriptors
    /// @param args Kernel argument data
    /// @param event Optional dependencies
    /// @param out_event Optional output event to be signaled when kernel finishes
    virtual void append_kernel_launch(kernel& k, const kernel_arguments_desc& args_desc, const kernel_arguments_data& args, const std::vector<event::ptr>& events = {}, event::ptr out_event = nullptr) = 0;
protected:
    virtual void close_impl() = 0;
    virtual void reset_impl() = 0;
    bool m_is_closed;
};

}  // namespace cldnn
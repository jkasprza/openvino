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
class kernel;

/// @brief Collection of commands that can be executed
class command_list {
public:
    using ptr = std::shared_ptr<command_list>;

    command_list() = default;
    virtual ~command_list() = default;

    /// @brief Closes command list
    /// @param out_event Optional event to be signaled when command list finishes execution
    virtual void close(event::ptr out_event = nullptr) = 0;

    /// @brief Check if command list is closed and can be executed
    /// @return True if command list is closed, false otherwise
    virtual bool is_closed() const = 0;

    /// @brief Resets command list to the empty state
    virtual void reset() = 0;

    /// @brief Append kernel launch command to the command list
    /// @param k Kernel to be appended
    /// @param args_desc Kernel argument descriptors
    /// @param args Kernel argument data
    /// @param event Optional dependencies
    /// @param out_event Optional output event to be signaled when kernel finishes
    virtual void append_kernel_launch(kernel& k, const kernel_arguments_desc& args_desc, const kernel_arguments_data& args, const std::vector<event::ptr>& events = {}, event::ptr out_event = nullptr) = 0;

    virtual void execute(stream str) = 0;
protected:
    bool m_is_closed = false;
};

}  // namespace cldnn

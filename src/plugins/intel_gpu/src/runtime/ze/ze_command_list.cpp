// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "ze_command_list.hpp"
#include <memory>
#include "intel_gpu/runtime/utils.hpp"
#include "ze/ze_engine.hpp"
#include "ze/ze_kernel.hpp"
#include "ze/ze_memory.hpp"
#include "ze_empty_event.hpp"
#include "ze_api.h"

namespace cldnn {
namespace ze {

ze_command_list::ze_command_list(ze_stream &stream)
    : m_stream(stream) {
    const auto& engine = stream.get_engine();
    const auto &info = engine.get_device_info();
    ze_command_list_desc_t command_list_desc;
    command_list_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    command_list_desc.pNext = nullptr;
    command_list_desc.commandQueueGroupOrdinal = info.compute_queue_group_ordinal;
    command_list_desc.flags = (m_stream.get_queue_type() == QueueTypes::in_order) ? ZE_COMMAND_LIST_FLAG_IN_ORDER : 0;

    // TODO: Add support for mutable command lists
    /*ze_mutable_command_list_exp_desc_t mcl_desc;
    mcl_desc.stype = ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_LIST_EXP_DESC;
    mcl_desc.pNext = nullptr;
    mcl_desc.flags = 0;
    if (info.supports_mutable_command_list) {
        command_list_desc.pNext = &mcl_desc;
    }*/

    OV_ZE_EXPECT(zeCommandListCreate(engine.get_context(), engine.get_device(), &command_list_desc, &m_command_list));
    OV_ZE_EXPECT(zeCommandListAppendBarrier(m_command_list, nullptr, 0, nullptr));
}

event::ptr ze_command_list::append_kernel_launch(kernel& k,
        const kernel_arguments_desc& args_desc,
        const kernel_arguments_data& args,
        const std::vector<event::ptr>& events,
        bool needs_out_event) {
    auto& ze_kern = downcast<ze_kernel>(k);
    auto sync_method = m_stream.get_sync_method();
    bool set_out_event = needs_out_event || sync_method == SyncMethods::events;
    auto out_event = set_out_event ? m_stream.create_base_event() : m_stream.create_empty_event();
    ze_kern.launch(m_command_list, args_desc, args, events, out_event);
    return out_event;
}

void ze_command_list::close_impl() {
    OV_ZE_EXPECT(zeCommandListClose(m_command_list));
}

void ze_command_list::reset_impl() {
    OV_ZE_EXPECT(zeCommandListReset(m_command_list));
}

ze_command_list::~ze_command_list() {
    OV_ZE_WARN(zeCommandListDestroy(m_command_list));
}

//event::ptr ze_command_list::get_output_event() const {
//    assert(m_output_event != nullptr);
//    return m_output_event;
//}

}  // namespace ze
}  // namespace cldnn
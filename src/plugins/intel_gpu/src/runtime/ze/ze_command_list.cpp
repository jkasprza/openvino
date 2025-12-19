// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "ze_command_list.hpp"
#include <memory>
#include "intel_gpu/runtime/utils.hpp"
#include "ze/ze_engine.hpp"
#include "ze/ze_kernel.hpp"
#include "ze/ze_memory.hpp"
#include "ze_api.h"

namespace cldnn {
namespace ze {

ze_command_list::ze_command_list(const ze_engine& engine)
    : m_engine(engine)
    , m_pool(engine.create_events_pool(2, false)) {
    const auto &info = m_engine.get_device_info();
    ze_command_list_desc_t command_list_desc;
    command_list_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    command_list_desc.pNext = nullptr;
    command_list_desc.commandQueueGroupOrdinal = info.compute_queue_group_ordinal;
    command_list_desc.flags = ZE_COMMAND_LIST_FLAG_IN_ORDER;
    
    // TODO: Add support for mutable command lists
    /*ze_mutable_command_list_exp_desc_t mcl_desc;
    mcl_desc.stype = ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_LIST_EXP_DESC;
    mcl_desc.pNext = nullptr;
    mcl_desc.flags = 0;
    if (info.supports_mutable_command_list) {
        command_list_desc.pNext = &mcl_desc;
    }*/

    OV_ZE_EXPECT(zeCommandListCreate(m_engine.get_context(), m_engine.get_device(), &command_list_desc, &m_command_list));
    OV_ZE_EXPECT(zeCommandListAppendBarrier(m_command_list, nullptr, 0, nullptr));
}

void ze_command_list::add_kernel(kernel& k, const kernel_arguments_desc& args_desc, const kernel_arguments_data& args) {
    auto& casted = downcast<ze_kernel>(k);
    auto& ze_handle = casted.get_handle();

    auto global = to_group_count(args_desc.workGroups.global);
    auto local = to_group_count(args_desc.workGroups.local);
    ze_group_count_t ze_args = { global.groupCountX / local.groupCountX, global.groupCountY / local.groupCountY, global.groupCountZ / local.groupCountZ };

    set_arguments_impl(ze_handle, args_desc.arguments, args);
    OV_ZE_EXPECT(zeCommandListAppendBarrier(m_command_list, nullptr, 0, nullptr));
    OV_ZE_EXPECT(zeKernelSetGroupSize(ze_handle, local.groupCountX, local.groupCountY, local.groupCountZ));
    OV_ZE_EXPECT(zeCommandListAppendLaunchKernel(m_command_list, ze_handle, &ze_args, nullptr, 0, nullptr));
}

void ze_command_list::close() {
    m_output_event = m_pool->create_event();
    OV_ZE_EXPECT(zeCommandListAppendBarrier(m_command_list, std::dynamic_pointer_cast<ze_event>(m_output_event)->get(), 0, nullptr));
    OV_ZE_EXPECT(zeCommandListClose(m_command_list));
}

void ze_command_list::reset() {
    OV_ZE_EXPECT(zeCommandListReset(m_command_list));
}

ze_command_list::~ze_command_list() {
    reset();
}

uint64_t ze_command_list::get_command_id() {
    if (is_mutable()) {
        ze_mutable_command_exp_flags_t flags =
            ZE_MUTABLE_COMMAND_EXP_FLAG_KERNEL_ARGUMENTS |
            ZE_MUTABLE_COMMAND_EXP_FLAG_GROUP_COUNT |
            ZE_MUTABLE_COMMAND_EXP_FLAG_GROUP_SIZE;

        ze_mutable_command_id_exp_desc_t cmd_id_desc = { ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_ID_EXP_DESC, nullptr, flags };
        uint64_t cmd_id = 0;
        OV_ZE_EXPECT(zeCommandListGetNextCommandIdExp(m_command_list, &cmd_id_desc, &cmd_id));
        return cmd_id;
    } else {
        thread_local uint64_t cmd_id = 0;
        cmd_id++;

        return cmd_id;
    }
}

event::ptr ze_command_list::get_output_event() const {
    assert(m_output_event != nullptr);
    return m_output_event;
}

}  // namespace ze
}  // namespace cldnn
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_counter_based_event_factory.hpp"
#include "ze_common.hpp"
#include "ze_resource.hpp"
#include "ze_counter_based_event.hpp"

#include "compute_runtime/zex_event.h"

#include <mutex>

namespace cldnn::ze {

ze_counter_based_event_factory::ze_counter_based_event_factory(const ze_engine &ze_engine, bool enable_profiling)
    : ze_base_event_factory(ze_engine, enable_profiling) {}

std::shared_ptr<ze_base_event> ze_counter_based_event_factory::create_event(uint64_t queue_stamp) {
    std::lock_guard<std::mutex> lock(_mutex);
    const auto &engine = get_engine();

    ze_event_handle_t event;
    auto desc = ze_event_counter_based_desc_t{
        ZE_STRUCTURE_TYPE_EVENT_COUNTER_BASED_DESC, //stype
        nullptr, //pNext
        ZE_EVENT_COUNTER_BASED_FLAG_HOST_VISIBLE
            | ZE_EVENT_COUNTER_BASED_FLAG_IMMEDIATE
            | ZE_EVENT_COUNTER_BASED_FLAG_NON_IMMEDIATE, // flags
        ZE_EVENT_SCOPE_FLAG_HOST, // signal
        ZE_EVENT_SCOPE_FLAG_DEVICE // wait
    };
    if (is_profiling_enabled()) {
        desc.flags |= ZE_EVENT_COUNTER_BASED_FLAG_DEVICE_TIMESTAMP;
    }
    OV_ZE_EXPECT(ze::zeEventCounterBasedCreate(engine.get_context().handle(), engine.get_device().handle(), &desc, &event));
    auto ev_holder = ze_event_resource(event);
    return std::make_shared<ze_counter_based_event>(queue_stamp, *this, ev_holder);
}
}  // namespace cldnn::ze

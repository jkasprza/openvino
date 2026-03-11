// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/device_query.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "ze_event_factory.hpp"
#include "ze_base_event.hpp"

#include <gtest/gtest.h>

TEST(test_ze_event_factory, CanCreateEvent) {
    // GIVEN ze_event_factory
    // WHEN create_event is called
    // THEN event is created and has correct queue stamp

    auto engine = cldnn::engine::create(cldnn::device_query::get_default_engine_type(), cldnn::device_query::get_default_runtime_type());
    OPENVINO_ASSERT(engine->type() == cldnn::engine_types::ze, "Expected L0 backend");
    auto& ze_engine = cldnn::downcast<cldnn::ze::ze_engine>(*engine);
    cldnn::ze::ze_event_factory factory(ze_engine, false);
    // Errors up to this point indicate a failure in preconditions
    auto event = factory.create_event(123);
    auto& ze_event = cldnn::downcast<cldnn::ze::ze_base_event>(*event);
    EXPECT_EQ(ze_event.get_queue_stamp(), 123);
}
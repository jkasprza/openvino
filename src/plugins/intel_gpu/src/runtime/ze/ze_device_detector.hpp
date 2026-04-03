// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/device.hpp"

#include <string>
#include <vector>
#include <map>

namespace cldnn {
namespace ze {

class ze_device_detector {
public:
    ze_device_detector() = default;

    std::map<std::string, device::ptr> get_available_devices(void* user_context,
                                                             void* user_device,
                                                             int ctx_device_id,
                                                             int target_tile_id,
                                                             bool initialize_devices = false) const;
};

}  // namespace ze
}  // namespace cldnn

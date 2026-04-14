// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_device_detector.hpp"
#include "ze_device.hpp"
#include "ze_common.hpp"
#include <ze_api.h>
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/ocl_ze_converter.hpp"
#include "openvino/core/except.hpp"

#include <vector>

namespace cldnn {
namespace ze {

namespace {
std::vector<device::ptr> create_device_list() {
    bool initialize_devices = false;
    std::vector<device::ptr> ret;

    uint32_t driver_count = 0;
    OV_ZE_EXPECT(zeDriverGet(&driver_count, nullptr));

    std::vector<ze_driver_handle_t> all_drivers(driver_count);
    OV_ZE_EXPECT(zeDriverGet(&driver_count, &all_drivers[0]));

    for (uint32_t i = 0; i < driver_count; ++i) {
        uint32_t device_count = 0;
        OV_ZE_EXPECT(zeDeviceGet(all_drivers[i], &device_count, nullptr));

        std::vector<ze_device_handle_t> all_devices(device_count);
        OV_ZE_EXPECT(zeDeviceGet(all_drivers[i], &device_count, &all_devices[0]));

        for (uint32_t d = 0; d < device_count; ++d) {
            try {
                ze_device_properties_t device_properties{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
                OV_ZE_EXPECT(zeDeviceGetProperties(all_devices[d], &device_properties));

                if (ZE_DEVICE_TYPE_GPU == device_properties.type) {
                    ret.emplace_back(std::make_shared<ze_device>(all_drivers[i], all_devices[d], initialize_devices));
                }
            } catch (std::exception& ex) {
                GPU_DEBUG_LOG << "Devices query/creation failed for driver " << i << ex.what() << std::endl;
                GPU_DEBUG_LOG << "Platform is skipped" << std::endl;
                continue;
            }
        }
    }

    return ret;
}

std::vector<device::ptr> create_device_list_from_user_context(void* user_context, int ctx_device_id) {
    // Currently there is no way to obtain device lists from Level Zero context
    // Context is created for a specific driver and unless context was created with zeContextCreateEx then all driver devices are visible in the context
    // Work around is to try and create memory with provied context for each device and only consider successfull devices
    ze_context_handle_t ze_context = reinterpret_cast<ze_context_handle_t>(user_context);
    OPENVINO_ASSERT(zeContextGetStatus(ze_context) == ZE_RESULT_SUCCESS, "[GPU] User provided context is not in valid state");
    OPENVINO_ASSERT(ctx_device_id >= 0, "[GPU] Expected user provided device id to be non-negative. But got: ", ctx_device_id);

    auto check_mem_alloc = [](ze_context_handle_t context, ze_device_handle_t device) {
        constexpr size_t alloc_size = 64 * sizeof(float);
        void *memory = nullptr;
        ze_device_mem_alloc_desc_t device_desc{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC , nullptr, 0, 0};
        ze_result_t res = zeMemAllocDevice(context, &device_desc, alloc_size, sizeof(float), device, &memory);
        if (res == ZE_RESULT_SUCCESS) {
            OV_ZE_WARN(zeMemFree(context, memory));
            return true;
        }
        return false;
    };

    auto all_devices = create_device_list();
    std::vector<device::ptr> context_devices;
    int devices_to_skip = ctx_device_id;
    for (auto& device : all_devices) {
        ze_device_handle_t ze_device_handle = std::dynamic_pointer_cast<ze_device>(device)->get_device();
        
        if (check_mem_alloc(ze_context, ze_device_handle) && devices_to_skip-- == 0) {
            context_devices.push_back(device);
            break; // We expect only one device
        }
    }
    OPENVINO_ASSERT(!context_devices.empty(), "[GPU] No devices found for the provided user context");
    return context_devices;
}

std::vector<device::ptr> create_device_list_from_user_device(void* user_device) {
    OPENVINO_NOT_IMPLEMENTED;
}

std::vector<ze_device_handle_t> get_sub_devices(ze_device_handle_t root_device) {
    uint32_t n_subdevices = 0;
    OV_ZE_EXPECT(zeDeviceGetSubDevices(root_device, &n_subdevices, nullptr));
    if (n_subdevices == 0)
        return {};

    std::vector<ze_device_handle_t> subdevices(n_subdevices);

    OV_ZE_EXPECT(zeDeviceGetSubDevices(root_device, &n_subdevices, subdevices.data()));

    return subdevices;
}
} // namespace

std::map<std::string, device::ptr> ze_device_detector::get_available_devices(runtime_types context_type,
                                                                             void* user_context,
                                                                             void* user_device,
                                                                             int ctx_device_id,
                                                                             int target_tile_id,
                                                                             bool initialize_devices) const {
    std::vector<device::ptr> devices_list;
    // We must call this function before any other Level Zero API
    OV_ZE_EXPECT(zeInit(ZE_INIT_FLAG_GPU_ONLY));
    if (user_context != nullptr) {
        if (context_type == runtime_types::ze) {
            devices_list = create_device_list_from_user_context(user_context, ctx_device_id);
        } else if (context_type == runtime_types::ocl) {
            auto handles = ocl_ze_converter::get_ze_devices_from_ocl_context(user_context);
            OPENVINO_ASSERT(handles.size() > (size_t)ctx_device_id, "[GPU] Could not find device handle for provided user context");
            auto selected_handle = handles[ctx_device_id];
            auto full_list = create_device_list();
            auto selected_device = std::find_if(full_list.begin(), full_list.end(), [=](cldnn::device::ptr device){
                auto &zero_device = downcast<const ze_device>(*device);
                return zero_device.get_device() == selected_handle;
            });
            OPENVINO_ASSERT(selected_device != devices_list.end(), "[GPU] No devices found for the provided user context");
            devices_list = {*selected_device};
        }
    } else if (user_device != nullptr) {
        devices_list = create_device_list_from_user_device(user_device);
    } else {
        devices_list = create_device_list();
    }

    devices_list = sort_devices(devices_list);

    std::map<std::string, device::ptr> ret;
    uint32_t idx = 0;
    for (auto& dptr : devices_list) {
        auto map_id = std::to_string(idx++);
        ret[map_id] = dptr;

        auto root_device = std::dynamic_pointer_cast<ze_device>(dptr);
        OPENVINO_ASSERT(root_device != nullptr, "[GPU] Invalid device type created in ze_device_detector");

        auto sub_devices = get_sub_devices(root_device->get_device());
        uint32_t sub_idx = 0;
        for (auto& sub_device : sub_devices) {
            if (target_tile_id != -1 && static_cast<int>(sub_idx) != target_tile_id) {
                sub_idx++;
                continue;
            }
            auto sub_device_ptr = std::make_shared<ze_device>(root_device->get_driver(), sub_device, initialize_devices);
            ret[map_id + "." + std::to_string(sub_idx++)] = sub_device_ptr;
        }
    }

    return ret;
}
}  // namespace ze
}  // namespace cldnn

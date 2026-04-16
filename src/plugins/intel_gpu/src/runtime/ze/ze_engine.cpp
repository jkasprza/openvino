// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_engine.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "openvino/core/except.hpp"
#include "ze_kernel_builder.hpp"
#include "ze_api.h"
#include "ze_engine_factory.hpp"
#include "ze_common.hpp"
#include "ze_memory.hpp"
#include "ze_stream.hpp"
#include "ze_device.hpp"
#include "ze_kernel.hpp"
#include "ze_module_holder.hpp"
#include "ze_kernel_holder.hpp"
#include <exception>
#include <vector>
#include <memory>
#include <stdexcept>

// For OCL interop
#include "ocl/ocl_device.hpp"
#include "intel_gpu/runtime/ocl_ze_converter.hpp"

#ifdef ENABLE_ONEDNN_FOR_GPU
#include <oneapi/dnnl/dnnl_ze.hpp>
#endif
namespace cldnn {
namespace ze {

namespace {

ze_device::ptr ze_device_from_ocl_device(const ocl::ocl_device::ptr& ocl_dev) {
    ze_device_handle_t device = reinterpret_cast<ze_device_handle_t>(
        ocl_ze_converter::get_ze_device_from_cl_device(ocl_dev->get_handle(runtime_resources::OCL_DEVICE)));
    ze_context_handle_t context = reinterpret_cast<ze_context_handle_t>(
        ocl_ze_converter::get_ze_context_from_cl_context(ocl_dev->get_handle(runtime_resources::OCL_CONTEXT)));
    // Currently there is no way to convert platform to driver directly
    OV_ZE_EXPECT(zeInit(ZE_INIT_FLAG_GPU_ONLY));
    uint32_t driver_count = 0;
    OV_ZE_EXPECT(zeDriverGet(&driver_count, nullptr));
    std::vector<ze_driver_handle_t> all_drivers(driver_count);
    OV_ZE_EXPECT(zeDriverGet(&driver_count, &all_drivers[0]));
    ze_driver_handle_t matching_driver = nullptr;
    for (auto driver : all_drivers) {
        uint32_t device_count = 0;
        OV_ZE_EXPECT(zeDeviceGet(driver, &device_count, nullptr));
        std::vector<ze_device_handle_t> all_devices(device_count);
        OV_ZE_EXPECT(zeDeviceGet(driver, &device_count, all_devices.data()));
        for (auto dev : all_devices) {
            if (dev == device) {
                matching_driver = driver;
                break;
            }
        }
        if (matching_driver != nullptr) {
            break;
        }
    }
    OPENVINO_ASSERT(matching_driver != nullptr, "[GPU] Unable to find matching Level Zero driver for the given OpenCL device");
    bool initialize = ocl_dev->is_initialized();
    return std::make_shared<ze_device>(matching_driver, device, context, initialize);
}

ocl::ocl_device::ptr ocl_device_from_ze_device(const ze_device::ptr& ze_dev) {
    auto target_ze_device = ze_dev->get_handle(runtime_resources::ZE_DEVICE);
    cl_platform_id matching_platform = nullptr;
    cl_device_id matching_device = nullptr;
    cl_uint num_platforms = 0;
    cl_int error_code = clGetPlatformIDs(0, NULL, &num_platforms);
    OPENVINO_ASSERT(num_platforms != 0, "[GPU] No OpenCL platforms found");
    OPENVINO_ASSERT(error_code == CL_SUCCESS, "[GPU] clGetPlatformIDs error code: ", std::to_string(error_code));
    std::vector<cl_platform_id> platform_ids(num_platforms);
    error_code = clGetPlatformIDs(num_platforms, platform_ids.data(), NULL);
    OPENVINO_ASSERT(error_code == CL_SUCCESS, "[GPU] clGetPlatformIDs error code: ", std::to_string(error_code));
    for (auto& id : platform_ids) {
        cl::Platform platform = cl::Platform(id);
        try {
            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
            for (auto& device : devices) {
                if (ocl_ze_converter::get_ze_device_from_cl_device(device.get()) == target_ze_device) {
                    matching_device = device.get();
                    matching_platform = id;
                    break;
                }
            }
        } catch (...) {
            continue;
        }
        if (matching_device != nullptr) {
            break;
        }
    }
    OPENVINO_ASSERT(matching_device != nullptr, "[GPU] Unable to find matching OpenCL device for the given Level Zero device");
    OPENVINO_ASSERT(matching_platform != nullptr, "[GPU] Unable to find matching OpenCL platform for the given Level Zero device");
    cl_context context = reinterpret_cast<cl_context>(
        ocl_ze_converter::create_cl_context_from_ze_context(matching_device, ze_dev->get_handle(runtime_resources::ZE_CONTEXT)));
    bool initialize = ze_dev->is_initialized();
    return std::make_shared<ocl::ocl_device>(cl::Device(matching_device), cl::Context(context), cl::Platform(matching_platform), initialize);
}
} // namespace

ze_engine::ze_engine(const device::ptr dev, runtime_types runtime_type)
    : engine(dev) {
    OPENVINO_ASSERT(runtime_type == runtime_types::ze, "[GPU] Invalid runtime type specified for ZE engine. Only ZE runtime is supported");

    auto ze_casted = std::dynamic_pointer_cast<ze_device>(dev);
    auto ocl_casted = std::dynamic_pointer_cast<ocl::ocl_device>(dev);
    if (ze_casted == nullptr && ocl_casted != nullptr) {
        ze_casted = ze_device_from_ocl_device(ocl_casted);
    } else if (ocl_casted == nullptr && ze_casted != nullptr) {
        ocl_casted = ocl_device_from_ze_device(ze_casted);
    }
    _device = ze_casted;
    _ocl_device = ocl_casted;
    OPENVINO_ASSERT(_device != nullptr && _ocl_device != nullptr, "[GPU] Expected to initialize both ze_device and ocl_device");

    _service_stream.reset(new ze_stream(*this, ExecutionConfig()));
}

#ifdef ENABLE_ONEDNN_FOR_GPU
void ze_engine::create_onednn_engine(const ExecutionConfig& config) {
    const std::lock_guard<std::mutex> lock(onednn_mutex);
    OPENVINO_ASSERT(_device->get_info().vendor_id == INTEL_VENDOR_ID, "[GPU] OneDNN engine can be used for Intel GPUs only");
    if (!_onednn_engine) {
        auto casted = std::dynamic_pointer_cast<ze_device>(_device);
        _onednn_engine = std::make_shared<dnnl::engine>(dnnl::ze_interop::make_engine(casted->get_driver(), casted->get_device(), casted->get_context()));
    }
}
#endif

const ze_driver_handle_t ze_engine::get_driver() const {
    auto casted = std::dynamic_pointer_cast<ze_device>(_device);
    OPENVINO_ASSERT(casted, "[GPU] Invalid device type for ze_engine");
    return casted->get_driver();
}

const ze_context_handle_t ze_engine::get_context() const {
    auto casted = std::dynamic_pointer_cast<ze_device>(_device);
    OPENVINO_ASSERT(casted, "[GPU] Invalid device type for ze_engine");
    return casted->get_context();
}

const ze_device_handle_t ze_engine::get_device() const {
    auto casted = std::dynamic_pointer_cast<ze_device>(_device);
    OPENVINO_ASSERT(casted, "[GPU] Invalid device type for ze_engine");
    return casted->get_device();
}

allocation_type ze_engine::detect_usm_allocation_type(const void* memory) const {
    return ze::gpu_usm::detect_allocation_type(this, memory);
}

memory::ptr ze_engine::allocate_memory(const layout& layout, allocation_type type, bool reset) {
    OPENVINO_ASSERT(!layout.is_dynamic() || layout.has_upper_bound(), "[GPU] Can't allocate memory for dynamic layout");

    check_allocatable(layout, type);

    try {
        memory::ptr res;
        if (layout.format.is_image_2d()) {
            res = std::make_shared<ze::gpu_image2d>(this, layout);
        } else if (type == allocation_type::cl_mem) {
            res = std::make_shared<ze::ocl_buffer>(this, layout);
        } else if (memory_capabilities::is_usm_type(type)){
            res = std::make_shared<ze::gpu_usm>(this, layout, type);
        } else {
            OPENVINO_THROW("[GPU] Unsupported allocation type: ", type);
        }

        if (reset || res->is_memory_reset_needed(layout)) {
            auto ev = res->fill(get_service_stream());
            if (ev) {
                get_service_stream().wait_for_events({ev});
            }
        }

        return res;
    } catch (const std::exception& e) {
        OPENVINO_THROW("[GPU] Failed to allocate memory: ", e.what());
    }
}

memory::ptr ze_engine::reinterpret_buffer(const memory& memory, const layout& new_layout) {
    OPENVINO_ASSERT(memory.get_engine() == this, "[GPU] trying to reinterpret buffer allocated by a different engine");
    OPENVINO_ASSERT(new_layout.format.is_image() == memory.get_layout().format.is_image(),
                    "[GPU] trying to reinterpret between image and non-image layouts. Current: ",
                    memory.get_layout().format.to_string(), " Target: ", new_layout.format.to_string());

    bool from_memory_pool = memory.from_memory_pool;
    memory::ptr reinterpret_memory = nullptr;
    if (memory_capabilities::is_usm_type(memory.get_allocation_type())) {
        reinterpret_memory = std::make_shared<ze::gpu_usm>(this,
                                     new_layout,
                                     reinterpret_cast<const ze::gpu_usm&>(memory).get_buffer(),
                                     memory.get_allocation_type(),
                                     memory.get_mem_tracker());
    } else if (new_layout.format.is_image_2d()) {
        reinterpret_memory = std::make_shared<ze::gpu_image2d>(this,
                                     new_layout,
                                     reinterpret_cast<const ze::gpu_image2d&>(memory).get_ocl_handle(),
                                     memory.get_mem_tracker());
    } else if (memory.get_allocation_type() == allocation_type::cl_mem) {
        reinterpret_memory = std::make_shared<ze::ocl_buffer>(this,
                                     new_layout,
                                     reinterpret_cast<const ze::ocl_buffer&>(memory).get_ocl_handle(),
                                     memory.get_mem_tracker());
    } else {
        OPENVINO_THROW("[GPU] Unexpected memory type for reinterpret_buffer");
    }
    reinterpret_memory->from_memory_pool = from_memory_pool;
    return reinterpret_memory;
}

memory::ptr ze_engine::reinterpret_handle(const layout& new_layout, shared_mem_params params) {
    if (params.mem_type == shared_mem_type::shared_mem_usm) {
        ze::UsmMemory usm_buffer(get_context(), get_device(), params.mem);
        return std::make_shared<ze::gpu_usm>(this, new_layout, usm_buffer, nullptr);
    } else if (params.mem_type == shared_mem_type::shared_mem_buffer) {
        return std::make_shared<ze::ocl_buffer>(this, new_layout, params.mem, nullptr);
    } else if (params.mem_type == shared_mem_type::shared_mem_image) {
        return std::make_shared<ze::gpu_image2d>(this, new_layout, params.mem, nullptr);
    } else {
        OPENVINO_THROW("[GPU] Unsupported shared memory type: ", params.mem_type);
    }
}

memory_ptr ze_engine::create_subbuffer(const memory& memory, const layout& new_layout, size_t byte_offset) {
    OPENVINO_ASSERT(memory.get_engine() == this, "[GPU] Trying to create a subbuffer from a buffer allocated by a different engine");
    if (new_layout.format.is_image_2d()) {
        OPENVINO_NOT_IMPLEMENTED;
    }
    OPENVINO_ASSERT(memory_capabilities::is_usm_type(memory.get_allocation_type()), "[GPU] Trying to create subbuffer for non usm memory");
    auto& new_buf = reinterpret_cast<const ze::gpu_usm&>(memory);
    auto ptr = new_buf.get_buffer().get();
    auto sub_buffer = ze::UsmMemory(get_context(), get_device(), ptr, byte_offset);
    return std::make_shared<ze::gpu_usm>(this,
                             new_layout,
                             sub_buffer,
                             memory.get_allocation_type(),
                             memory.get_mem_tracker());
}

bool ze_engine::is_the_same_buffer(const memory& mem1, const memory& mem2) {
    if (mem1.get_engine() != this || mem2.get_engine() != this)
        return false;
    if (mem1.get_allocation_type() != mem2.get_allocation_type())
        return false;
    if (&mem1 == &mem2)
        return true;

    return (reinterpret_cast<const ze::gpu_usm&>(mem1).get_buffer().get() == reinterpret_cast<const ze::gpu_usm&>(mem2).get_buffer().get());
}

std::shared_ptr<kernel_builder> ze_engine::create_kernel_builder() const {
    auto casted = std::dynamic_pointer_cast<ze_device>(_device);
    OPENVINO_ASSERT(casted, "[GPU] Invalid device type for ze_engine");
    return std::make_shared<ze_kernel_builder>(*casted);
}

void* ze_engine::get_user_context() const {
    auto& casted = downcast<ze_device>(*_device);
    return static_cast<void*>(casted.get_context());
}

stream::ptr ze_engine::create_stream(const ExecutionConfig& config) const {
    return std::make_shared<ze_stream>(*this, config);
}

stream::ptr ze_engine::create_stream(const ExecutionConfig& config, void* handle) const {
    return std::make_shared<ze_stream>(*this, config, handle);
}

std::shared_ptr<cldnn::engine> ze_engine::create(const device::ptr device, runtime_types runtime_type) {
    return std::make_shared<ze::ze_engine>(device, runtime_type);
}

std::shared_ptr<cldnn::engine> create_ze_engine(const device::ptr device, runtime_types runtime_type) {
    return ze_engine::create(device, runtime_type);
}

}  // namespace ze
}  // namespace cldnn

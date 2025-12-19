// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ze_kernel.hpp"
#include "ze_memory.hpp"

#include <memory>

namespace cldnn {
namespace ze {
namespace {

template<typename T>
ze_result_t set_kernel_arg_scalar(ze_kernel_handle_t& kernel, uint32_t idx, const T& val) {
    GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel << " set scalar " << idx << " (" << ov::element::from<T>().get_type_name() << ")" << val << "\n";
    return zeKernelSetArgumentValue(kernel, idx, sizeof(T), &val);
}

ze_result_t set_kernel_arg_local_memory(ze_kernel_handle_t& kernel, uint32_t idx, size_t size) {
    if (size == 0)
        return ZE_RESULT_ERROR_INVALID_ARGUMENT;

    GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel << " set arg " << idx << " local memory size: " << size << std::endl;
    return zeKernelSetArgumentValue(kernel, idx, size, NULL);
}

ze_result_t set_kernel_arg(ze_kernel_handle_t& kernel, uint32_t idx, cldnn::memory::cptr mem) {
    if (!mem)
        return ZE_RESULT_ERROR_INVALID_ARGUMENT;

    OPENVINO_ASSERT(memory_capabilities::is_usm_type(mem->get_allocation_type()), "Unsupported alloc type");
    const auto& buf = std::dynamic_pointer_cast<const ze::gpu_usm>(mem)->get_buffer();
    auto mem_type = std::dynamic_pointer_cast<const ze::gpu_usm>(mem)->get_allocation_type();
    GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel << " set arg (" << mem_type << ") " << idx
                            << " mem: " << buf.get() << " size: " << mem->size() << std::endl;

    auto ptr = buf.get();
    return zeKernelSetArgumentValue(kernel, idx, sizeof(ptr), &ptr);
}

void set_arguments_impl(ze_kernel_handle_t kernel,
                         const arguments_desc& args,
                         const kernel_arguments_data& data) {
    using args_t = argument_desc::Types;
    using scalar_t = scalar_desc::Types;

    for (uint32_t i = 0; i < static_cast<uint32_t>(args.size()); i++) {
        ze_result_t status = ZE_RESULT_NOT_READY;
        switch (args[i].t) {
            case args_t::INPUT:
                if (args[i].index < data.inputs.size() && data.inputs[args[i].index]) {
                    status = set_kernel_arg(kernel, i, data.inputs[args[i].index]);
                }
                break;
            case args_t::INPUT_OF_FUSED_PRIMITIVE:
                if (args[i].index < data.fused_op_inputs.size() && data.fused_op_inputs[args[i].index]) {
                    status = set_kernel_arg(kernel, i, data.fused_op_inputs[args[i].index]);
                }
                break;
            case args_t::INTERNAL_BUFFER:
                if (args[i].index < data.intermediates.size() && data.intermediates[args[i].index]) {
                    status = set_kernel_arg(kernel, i, data.intermediates[args[i].index]);
                }
                break;
            case args_t::OUTPUT:
                if (args[i].index < data.outputs.size() && data.outputs[args[i].index]) {
                    status = set_kernel_arg(kernel, i, data.outputs[args[i].index]);
                }
                break;
            case args_t::WEIGHTS:
                status = set_kernel_arg(kernel, i, data.weights);
                break;
            case args_t::BIAS:
                status = set_kernel_arg(kernel, i, data.bias);
                break;
            case args_t::WEIGHTS_ZERO_POINTS:
                status = set_kernel_arg(kernel, i, data.weights_zero_points);
                break;
            case args_t::ACTIVATIONS_ZERO_POINTS:
                status = set_kernel_arg(kernel, i, data.activations_zero_points);
                break;
            case args_t::COMPENSATION:
                status = set_kernel_arg(kernel, i, data.compensation);
                break;
            case args_t::SCALE_TABLE:
                status = set_kernel_arg(kernel, i, data.scale_table);
                break;
            case args_t::SLOPE:
                status = set_kernel_arg(kernel, i, data.slope);
                break;
            case args_t::SCALAR:
                if (data.scalars && args[i].index < data.scalars->size()) {
                    const auto& scalar = (*data.scalars)[args[i].index];
                    switch (scalar.t) {
                        case scalar_t::UINT8:
                            status = set_kernel_arg_scalar<uint8_t>(kernel, i, scalar.v.u8);
                            break;
                        case scalar_t::UINT16:
                            status = set_kernel_arg_scalar<uint16_t>(kernel, i, scalar.v.u16);
                            break;
                        case scalar_t::UINT32:
                            status = set_kernel_arg_scalar<uint32_t>(kernel, i, scalar.v.u32);
                            break;
                        case scalar_t::UINT64:
                            status = set_kernel_arg_scalar<uint64_t>(kernel, i, scalar.v.u64);
                            break;
                        case scalar_t::INT8:
                            status = set_kernel_arg_scalar<int8_t>(kernel, i, scalar.v.s8);
                            break;
                        case scalar_t::INT16:
                            status = set_kernel_arg_scalar<int16_t>(kernel, i, scalar.v.s16);
                            break;
                        case scalar_t::INT32:
                            status = set_kernel_arg_scalar<int32_t>(kernel, i, scalar.v.s32);
                            break;
                        case scalar_t::INT64:
                            status = set_kernel_arg_scalar<int64_t>(kernel, i, scalar.v.s64);
                            break;
                        case scalar_t::FLOAT32:
                            status = set_kernel_arg_scalar<float>(kernel, i, scalar.v.f32);
                            break;
                        case scalar_t::FLOAT64:
                            status = set_kernel_arg_scalar<double>(kernel, i, scalar.v.f64);
                            break;
                        default:
                            break;
                    }
                }
                break;
            case args_t::CELL:
                status = set_kernel_arg(kernel, i, data.cell);
                break;
            case args_t::SHAPE_INFO:
                status = set_kernel_arg(kernel, i, data.shape_info);
                break;
            case args_t::LOCAL_MEMORY_SIZE:
                OPENVINO_ASSERT(args[i].index < data.local_memory_args->size() && data.local_memory_args->at(args[i].index),
                                "The allocated local memory is necessary to set kernel arguments.");
                status = set_kernel_arg_local_memory(kernel, i,  data.local_memory_args->at(args[i].index));
                break;
            default:
                break;
        }
        if (status != ZE_RESULT_SUCCESS) {
            throw std::runtime_error("Error set arg " + std::to_string(i) + ", error code: " + std::to_string(status) + "\n");
        }
    }
}

}  // namespace

void ze_kernel::create_kernels_from_module(std::shared_ptr<ze_module_holder> module, std::vector<kernel::ptr> &out) {
    ze_module_handle_t module_handle = module->get_module_handle();
    uint32_t kernel_count = 0;
    OV_ZE_EXPECT(zeModuleGetKernelNames(module_handle, &kernel_count, nullptr));
    std::vector<const char*> kernel_names(kernel_count);
    // Specification does not mention who is responsible for the returned pointers
    // Assume Level Zero owns the pointers and they will remain valid as long as the module resource
    OV_ZE_EXPECT(zeModuleGetKernelNames(module_handle, &kernel_count, kernel_names.data()));
    ze_kernel_flags_t flags = 0;
    ze_kernel_desc_t kernel_desc = {
        ZE_STRUCTURE_TYPE_KERNEL_DESC, nullptr, flags, nullptr};
    for (auto name_cstr : kernel_names) {
        auto name = std::string(name_cstr);
        // L0 returns Intel_Symbol_Table_Void_Program that does not correspond to actual kernel
        if (name == "Intel_Symbol_Table_Void_Program") {
            continue;
        }
        kernel_desc.pKernelName = name_cstr;
        ze_kernel_handle_t kernel_handle;
        OV_ZE_EXPECT(zeKernelCreate(module_handle, &kernel_desc, &kernel_handle));
        auto kernel_holder = std::make_shared<ze_kernel_holder>(kernel_handle, module);
        out.push_back(std::make_shared<ze_kernel>(kernel_holder, name));
    }
}

ze_kernel::ze_kernel(std::shared_ptr<ze_kernel_holder> kernel, const std::string& kernel_id)
    : m_kernel(kernel)
    , m_kernel_id(kernel_id) { }

ze_kernel_handle_t ze_kernel::get_kernel_handle() const { return m_kernel->get_kernel_handle(); }
ze_module_handle_t ze_kernel::get_module_handle() const { return m_kernel->get_module()->get_module_handle(); }
std::string ze_kernel::get_id() const { return m_kernel_id; }

std::shared_ptr<kernel> ze_kernel::clone(bool reuse_kernel_handle = false) const {
    if (reuse_kernel_handle) {
        return std::make_shared<ze_kernel>(m_kernel, m_kernel_id);
    } else {
        ze_kernel_handle_t cloned_handle;
        ze_module_handle_t module_handle = get_module_handle();
        ze_kernel_desc_t descriptor;
        descriptor.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
        descriptor.pNext = nullptr;
        descriptor.flags = 0;
        descriptor.pKernelName = m_kernel_id.c_str();
        OV_ZE_EXPECT(zeKernelCreate(module_handle, &descriptor, &cloned_handle));
        auto kernel_holder = std::make_shared<ze_kernel_holder>(cloned_handle, m_kernel->get_module());
        return std::make_shared<ze_kernel>(kernel_holder, m_kernel_id);
    }
}

void ze_kernel::set_arguments(const kernel_arguments_desc& args_desc, const kernel_arguments_data& args) {
    GPU_DEBUG_TRACE_DETAIL << "Set arguments for primitive: " << args_desc.layerID << " (" << get_id() << " = " << get_kernel_handle() << ")\n";
    set_arguments_impl(get_kernel_handle(), args_desc.arguments, args);
}

std::vector<uint8_t> ze_kernel::get_binary() const {
    size_t binary_size = 0;
    ze_module_handle_t module_handle = get_module_handle();
    OV_ZE_EXPECT(zeModuleGetNativeBinary(module_handle, &binary_size, nullptr));

    std::vector<uint8_t> binary(binary_size);
    OV_ZE_EXPECT(zeModuleGetNativeBinary(module_handle, &binary_size, binary.data()));

    return binary;
}

std::string ze_kernel::get_build_log() const {
    ze_module_build_log_handle_t build_log_handle = m_kernel->get_module()->get_build_log_handle();
    size_t log_size = 0;
    OV_ZE_EXPECT(zeModuleBuildLogGetString(build_log_handle, &log_size, nullptr));

    std::string log(log_size, ' ');
    OV_ZE_EXPECT(zeModuleBuildLogGetString(build_log_handle, &log_size, log.data()));
    return log;
}

} // namespace ze
}  // namespace cldnn

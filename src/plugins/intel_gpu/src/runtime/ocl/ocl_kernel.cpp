// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ocl_kernel.hpp"

#include <vector>
#include <cstdint>
namespace cldnn {
namespace ocl {
namespace {

cl_int set_kernel_arg(ocl_kernel_type& kernel, uint32_t idx, uint32_t size) {
    if (size == 0)
        return CL_INVALID_ARG_VALUE;

    GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel.get() << " set arg " << idx << " local memory size : " << size << std::endl;
    return kernel.setArg(idx, size, NULL);
}

cl_int set_kernel_arg(ocl_kernel_type& kernel, uint32_t idx, cldnn::memory::cptr mem) {
    if (!mem)
        return CL_INVALID_ARG_VALUE;

    if (mem->get_layout().format.is_image_2d()) {
        auto buf = std::dynamic_pointer_cast<const ocl::gpu_image2d>(mem)->get_buffer();
        GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel.get() << " set arg (image) " << idx << " mem: " << buf.get() << " size: " << mem->size() << std::endl;
        return kernel.setArg(idx, buf);
    } else if (memory_capabilities::is_usm_type(mem->get_allocation_type())) {
        auto buf = std::dynamic_pointer_cast<const ocl::gpu_usm>(mem)->get_buffer();
        auto mem_type = std::dynamic_pointer_cast<const ocl::gpu_usm>(mem)->get_allocation_type();
        GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel.get() << " set arg (" << mem_type << ") " << idx
                               << " mem: " << buf.get() << " size: " << mem->size() << std::endl;
        return kernel.setArgUsm(idx, buf);
    } else {
        auto buf = std::dynamic_pointer_cast<const ocl::gpu_buffer>(mem)->get_buffer();
        GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel.get() << " set arg (buffer) " << idx << " mem: " << buf.get() << " size: " << mem->size() << std::endl;
        return kernel.setArg(idx, buf);
    }

    return CL_INVALID_ARG_VALUE;
}

void set_arguments_impl(ocl_kernel_type& kernel,
                        const arguments_desc& args,
                        const kernel_arguments_data& data) {
    using args_t = argument_desc::Types;
    using scalar_t = scalar_desc::Types;
    for (uint32_t i = 0; i < static_cast<uint32_t>(args.size()); i++) {
        cl_int status = CL_INVALID_ARG_VALUE;
        switch (args[i].t) {
            case args_t::INPUT:
                OPENVINO_ASSERT(args[i].index < data.inputs.size() && data.inputs[args[i].index],
                               "The allocated input memory is necessary to set kernel arguments.");
                status = set_kernel_arg(kernel, i, data.inputs[args[i].index]);
                break;
            case args_t::INPUT_OF_FUSED_PRIMITIVE:
                OPENVINO_ASSERT(args[i].index < data.fused_op_inputs.size() && data.fused_op_inputs[args[i].index],
                                "The allocated fused_op_input memory is necessary to set kernel arguments.");
                status = set_kernel_arg(kernel, i, data.fused_op_inputs[args[i].index]);
                break;
            case args_t::INTERNAL_BUFFER:
                OPENVINO_ASSERT(args[i].index < data.intermediates.size() && data.intermediates[args[i].index],
                                "The allocated intermediate memory is necessary to set kernel arguments.");
                status = set_kernel_arg(kernel, i, data.intermediates[args[i].index]);
                break;
            case args_t::OUTPUT:
                OPENVINO_ASSERT(args[i].index < data.outputs.size() && data.outputs[args[i].index],
                                "The allocated output memory is necessary to set kernel arguments.");
                status = set_kernel_arg(kernel, i, data.outputs[args[i].index]);
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
                            status = kernel.setArg(i, scalar.v.u8);
                            GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel.get() << " set scalar " << i << " (u8): " << static_cast<int>(scalar.v.u8) << "\n";
                            break;
                        case scalar_t::UINT16:
                            status = kernel.setArg(i, scalar.v.u16);
                            GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel.get() << " set scalar " << i << " (u16): " << scalar.v.u16 << "\n";
                            break;
                        case scalar_t::UINT32:
                            status = kernel.setArg(i, scalar.v.u32);
                            GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel.get() << " set scalar " << i << " (u32): " << scalar.v.u32 << "\n";
                            break;
                        case scalar_t::UINT64:
                            status = kernel.setArg(i, scalar.v.u64);
                            GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel.get() << " set scalar " << i << " (u64): " << scalar.v.u64 << "\n";
                            break;
                        case scalar_t::INT8:
                            status = kernel.setArg(i, scalar.v.s8);
                            GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel.get() << " set scalar " << i << " (s8): " << static_cast<int>(scalar.v.s8) << "\n";
                            break;
                        case scalar_t::INT16:
                            status = kernel.setArg(i, scalar.v.s16);
                            GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel.get() << " set scalar " << i << " (s16): " << scalar.v.s16 << "\n";
                            break;
                        case scalar_t::INT32:
                            status = kernel.setArg(i, scalar.v.s32);
                            GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel.get() << " set scalar " << i << " (s32): " << scalar.v.s32 << "\n";
                            break;
                        case scalar_t::INT64:
                            status = kernel.setArg(i, scalar.v.s64);
                            GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel.get() << " set scalar " << i << " (s64): " << scalar.v.s64 << "\n";
                            break;
                        case scalar_t::FLOAT32:
                            status = kernel.setArg(i, scalar.v.f32);
                            GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel.get() << " set scalar " << i << " (f32): " << scalar.v.f32 << "\n";
                            break;
                        case scalar_t::FLOAT64:
                            status = kernel.setArg(i, scalar.v.f64);
                            GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel.get() << " set scalar " << i << " (f64): " << scalar.v.f64 << "\n";
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
                status = set_kernel_arg(kernel, i,  data.local_memory_args->at(args[i].index));
                break;
                break;
            default:
                break;
        }

        if (status != CL_SUCCESS) {
            throw std::runtime_error("Error set arg " + std::to_string(i)
                                     + ", kernel: " + kernel.getInfo<CL_KERNEL_FUNCTION_NAME>()
                                     + ", error code: " + std::to_string(status) + "\n");
        }
    }
}

}  // namespace

void ocl_kernel::set_arguments(const kernel_arguments_desc& args_desc, const kernel_arguments_data& args) {
    try {
        GPU_DEBUG_TRACE_DETAIL << "Set arguments for primitive: " << args_desc.layerID << " (" << get_id() << " = " << _compiled_kernel.get() << ")\n";
        set_arguments_impl(_compiled_kernel, args_desc.arguments, args);
    } catch (cl::Error const& err) {
        OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
    }
}

std::vector<uint8_t> ocl_kernel::get_binary() const {
    // Get the corresponding program object for the kernel
    cl_program program;
    cl_int error = clGetKernelInfo(_compiled_kernel.get(), CL_KERNEL_PROGRAM, sizeof(program), &program, nullptr);
    if (error) {
        throw std::runtime_error("Failed to retrieve CL_KERNEL_PROGRAM: " + std::to_string(error));
    }

    // Get the size of the program binary in bytes.
    size_t binary_size = 0;
    error = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(binary_size), &binary_size, nullptr);
    if (error) {
        throw std::runtime_error("Failed to retrieve CL_PROGRAM_BINARY_SIZES: " + std::to_string(error));
    }

    // Binary is not available for the device.
    if (binary_size == 0)
        throw std::runtime_error("get_binary: Binary size is zero");

    // Get program binary.
    std::vector<uint8_t> binary(binary_size);
    uint8_t* binary_buffer = binary.data();
    error = clGetProgramInfo(program, CL_PROGRAM_BINARIES, binary_size, &binary_buffer, nullptr);
    if (error) {
        throw std::runtime_error("Failed to retrieve CL_PROGRAM_BINARIES: " + std::to_string(error));
    }

    return binary;
}

std::string ocl_kernel::get_build_log() const {
    auto program = _compiled_kernel.getInfo<CL_KERNEL_PROGRAM>();
    auto log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>();
    // Assume program was build for only 1 device
    // Return first log
    if (log.size() > 0) {
        return log[0].second;
    }
    OPENVINO_THROW("[GPU] Failed to retrieve kernel build log");
}

}  // namespace ocl
}  // namespace cldnn

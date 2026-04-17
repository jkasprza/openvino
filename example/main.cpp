#include <iostream>
#include <string>
#include <cstdlib>
#include <chrono>
#include <vector>

#include <CL/cl.h>

#include "openvino/openvino.hpp"
#include "openvino/runtime/intel_gpu/ocl/ocl.hpp"

const std::string model_path = "sr-kuaishou-dirtylens.xml";

#define CL_EXPECT_SUCCESS(call) do { \
    cl_int _ret = (call); \
    if (CL_SUCCESS != _ret) { \
        throw std::runtime_error(std::string(#call) + " returned unexpected result " + std::to_string(_ret)); \
    } \
} while(false);

#define EXPECT(condition, msg) do { \
    if (!(condition)) { \
        throw std::runtime_error(std::string(msg)); \
    } \
} while(false);

#define PRINT_BORDER std::cout << std::string(40, '=') << std::endl;

void printInputAndOutputsInfo(std::ostream &stream, const ov::Model& network) {
    stream << "model name: " << network.get_friendly_name() << std::endl;

    const std::vector<ov::Output<const ov::Node>> inputs = network.inputs();
    for (const ov::Output<const ov::Node>& input : inputs) {
        stream << "    inputs" << std::endl;

        const std::string name = input.get_names().empty() ? "NONE" : input.get_any_name();
        stream << "        input name: " << name << std::endl;

        const ov::element::Type type = input.get_element_type();
        stream << "        input type: " << type << std::endl;

        const ov::Shape shape = input.get_shape();
        stream << "        input shape: " << shape << std::endl;
    }

    const std::vector<ov::Output<const ov::Node>> outputs = network.outputs();
    for (const ov::Output<const ov::Node>& output : outputs) {
        stream << "    outputs" << std::endl;

        const std::string name = output.get_names().empty() ? "NONE" : output.get_any_name();
        stream << "        output name: " << name << std::endl;

        const ov::element::Type type = output.get_element_type();
        stream << "        output type: " << type << std::endl;

        const ov::Shape shape = output.get_shape();
        stream << "        output shape: " << shape << std::endl;
    }
}

std::vector<std::vector<float>> generate_input_vectors(const ov::Model& network) {
    std::vector<std::vector<float>> input_vectors;
    const std::vector<ov::Output<const ov::Node>> inputs = network.inputs();
    for (const ov::Output<const ov::Node>& input : inputs) {
        const ov::element::Type type = input.get_element_type();
        if (type != ov::element::f32) {
            throw std::runtime_error("Expected input type is f32");
        }
        size_t size = 1;
        for (auto &e : input.get_shape()) {
            if (e <= 0) {
                throw std::runtime_error("Invalid input shape");
            }
            size *= e;
        }
        std::vector<float> input_data(size);
        for (auto &e : input_data) {
            e = (static_cast<float>(std::rand()) / RAND_MAX) * 2 - 1;
        }
        input_vectors.push_back(std::move(input_data));
    }
    return input_vectors;
}

std::vector<std::vector<float>> create_output_vectors(const ov::Model& network) {
    std::vector<std::vector<float>> output_vectors;
    const std::vector<ov::Output<const ov::Node>> outputs = network.outputs();
    for (const ov::Output<const ov::Node>& output : outputs) {
        const ov::element::Type type = output.get_element_type();
        if (type != ov::element::f32) {
            throw std::runtime_error("Expected output type is f32");
        }
        size_t size = 1;
        for (auto &e : output.get_shape()) {
            if (e <= 0) {
                throw std::runtime_error("Invalid output shape");
            }
            size *= e;
        }
        std::vector<float> output_data(size);
        output_vectors.push_back(std::move(output_data));
    }
    return output_vectors;
}

std::pair<cl_platform_id, cl_device_id> get_first_gpu_platform() {
    cl_uint numPlatforms = 0;

    CL_EXPECT_SUCCESS(clGetPlatformIDs(0, nullptr, &numPlatforms));
    std::vector<cl_platform_id> platform_ids(numPlatforms);
    CL_EXPECT_SUCCESS(clGetPlatformIDs(platform_ids.size(), platform_ids.data(), nullptr));

    for (auto &id : platform_ids) {
        size_t platform_name_size = 0;
        cl_device_id device;
        if (clGetDeviceIDs(id, CL_DEVICE_TYPE_GPU, 1, &device, nullptr) != CL_SUCCESS) {
            continue;
        }
        return std::make_pair(id, device);
    }
    throw std::runtime_error("No GPU platform found");
}

std::string get_device_name(cl_device_id device_id) {
    size_t size = 0;
    CL_EXPECT_SUCCESS(clGetDeviceInfo(device_id, CL_DEVICE_NAME, 0, nullptr, &size));
    std::string dev_name(size + 1, ' ');
    CL_EXPECT_SUCCESS(clGetDeviceInfo(device_id, CL_DEVICE_NAME, size, dev_name.data(), nullptr));
    return dev_name;
}

std::vector<cl_mem> create_cl_buffers(cl_context context, std::vector<std::vector<float>>& vectors) {
    std::vector<cl_mem> buffers;
    for (auto& vec : vectors) {
        cl_int error = CL_INVALID_VALUE;
        cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, vec.size() * sizeof(float), vec.data(), &error);
        CL_EXPECT_SUCCESS(error);
        buffers.push_back(buffer);
    }
    return buffers;
}

void destroy_cl_buffers(std::vector<cl_mem> buffers) {
    for (auto& buffer : buffers) {
        CL_EXPECT_SUCCESS(clReleaseMemObject(buffer));
    }
}

std::vector<std::vector<float>> run_remote_tensor_inference(ov::Core& core, std::shared_ptr<ov::Model> model, std::vector<std::vector<float>>& input_vectors, size_t iters) {
    std::cout << "\nPreparing OpenCL..." << std::endl;
    cl_platform_id platform_id;
    cl_device_id device_id;
    std::tie(platform_id, device_id) = get_first_gpu_platform();
    std::cout << "Using first GPU found: " << get_device_name(device_id) << std::endl;
    cl_int error = CL_INVALID_VALUE;
    cl_context context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &error);
    CL_EXPECT_SUCCESS(error);
    cl_command_queue queue =  clCreateCommandQueue(context, device_id, 0, &error);
    CL_EXPECT_SUCCESS(error);

    std::cout << "\nCreating OpenCL input buffers..." << std::endl;
    std::vector<cl_mem> input_buffers = create_cl_buffers(context, input_vectors);

    std::cout << "\nPreparing remote context and compiling model..." << std::endl;
    auto outputs = model->outputs();
    auto output_vectors = create_output_vectors(*model);
    auto output_buffers = create_cl_buffers(context, output_vectors);
    {
        auto remote_context = ov::intel_gpu::ocl::ClContext(core, queue);
        ov::CompiledModel compiled_model = core.compile_model(model, remote_context);
        auto gpu_context = compiled_model.get_context().as<ov::intel_gpu::ocl::ClContext>();
        EXPECT(gpu_context.get() == context, "OpenVINO Context should match original OpenCL context");

        std::cout << "\nPreparing inference request..." << std::endl;
        auto inf_req = compiled_model.create_infer_request();
        auto inputs = model->inputs();
        std::vector<ov::intel_gpu::ocl::ClBufferTensor> remote_input_tensors;
        for (int i = 0; i < inputs.size(); i++) {
            auto type = inputs[i].get_element_type();
            auto shape = inputs[i].get_shape();
            remote_input_tensors.push_back(gpu_context.create_tensor(type, shape, input_buffers[i]));
            EXPECT(remote_input_tensors[i].get() == input_buffers[i], "Remote tensor buffer should match original OpenCL buffer");
            inf_req.set_input_tensor(i, remote_input_tensors[i]);
        }
        std::vector<ov::intel_gpu::ocl::ClBufferTensor> remote_output_tensors;
        for (int i = 0; i < outputs.size(); i++) {
            auto type = outputs[i].get_element_type();
            auto shape = outputs[i].get_shape();
            remote_output_tensors.push_back(gpu_context.create_tensor(type, shape, output_buffers[i]));
            EXPECT(remote_output_tensors[i].get() == output_buffers[i], "Remote tensor buffer should match original OpenCL buffer");
            inf_req.set_output_tensor(i, remote_output_tensors[i]);
        }

        double total_time = 0;
        for (size_t i = 0; i < iters; i++) {
            std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
            inf_req.start_async();
            inf_req.wait();
            std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            total_time += elapsed_seconds.count();
        }
        std::cout << "Remote tensor inference average time: " << total_time / iters << "s\n";

        for (int i = 0; i < outputs.size(); i++) {
            CL_EXPECT_SUCCESS(clEnqueueReadBuffer(queue, output_buffers[i], CL_TRUE, 0, output_vectors[i].size() * sizeof(float), output_vectors[i].data(), 0, nullptr, nullptr));
        }
        CL_EXPECT_SUCCESS(clFinish(queue));
    }
    // Need to destroy compiled_model before releasing queue
    CL_EXPECT_SUCCESS(clReleaseCommandQueue(queue));
    destroy_cl_buffers(input_buffers);
    destroy_cl_buffers(output_buffers);
    CL_EXPECT_SUCCESS(clReleaseContext(context));
    return output_vectors;
}

std::vector<std::vector<float>> run_regular_inference(ov::Core& core, std::shared_ptr<ov::Model> model, std::vector<std::vector<float>>& input_vectors, size_t iters) {
    std::cout << "\nPreparing remote context and compiling model..." << std::endl;
    // Assume same GPU will be used as in remote tensor inference
    ov::CompiledModel compiled_model = core.compile_model(model, "GPU.0");

    std::cout << "\nPreparing inference request..." << std::endl;
    auto inf_req = compiled_model.create_infer_request();
    auto inputs = model->inputs();
    std::vector<ov::Tensor> input_tensors;
    for (int i = 0; i < inputs.size(); i++) {
        auto type = inputs[i].get_element_type();
        auto shape = inputs[i].get_shape();
        input_tensors.push_back(ov::Tensor(type, shape, input_vectors[i].data()));
        inf_req.set_input_tensor(i, input_tensors[i]);
    }

    double total_time = 0;
    for (int i = 0; i < iters; i++) {
        std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
        inf_req.start_async();
        inf_req.wait();
        std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        total_time += elapsed_seconds.count();
    }
    std::cout << "Regular inference average time: " << total_time / iters << "s\n";

    auto outputs = model->outputs();
    auto output_vectors = create_output_vectors(*model);
    for (int i = 0; i < outputs.size(); i++) {
        auto output_tensor = inf_req.get_output_tensor(i);
        for (int j = 0; j < output_tensor.get_byte_size() / sizeof(float); j++) {
            output_vectors[i][j] = output_tensor.data<float>()[j];
        }
    }
    input_tensors.clear();
    return output_vectors;
}

int main() {
    std::srand(0);
    std::cout << "OpenVINO version: " << ov::get_openvino_version() << std::endl;
    ov::Core core;

    std::cout << "\nLoading model files: " << model_path << std::endl;
    std::shared_ptr<ov::Model> model = core.read_model(model_path);

    std::cout << "\nGenerating input vectors..." << std::endl;
    printInputAndOutputsInfo(std::cout, *model);
    std::vector<std::vector<float>> input_vectors = generate_input_vectors(*model);

    size_t iters = 1000;
    std::cout << "Number of iterations: " << iters << std::endl;

    PRINT_BORDER;
    std::cout << "Running regular inference..." << std::endl;
    auto regular_results = run_regular_inference(core, model, input_vectors, iters);

    PRINT_BORDER;
    std::cout << "Running inference with remote tensors..." << std::endl;
    auto remote_tensor_results = run_remote_tensor_inference(core, model, input_vectors, iters);

    PRINT_BORDER;
    std::cout << "\nComparing results..." << std::endl;
    EXPECT(remote_tensor_results.size() == regular_results.size(), "Number of outputs should be the same");
    bool results_match = true;
    for (int i = 0; i < remote_tensor_results.size(); i++) {
        EXPECT(remote_tensor_results[i].size() == regular_results[i].size(), "Output sizes should be the same");
        float eps = 1e-5f;
        size_t zeros = 0;
        for (int j = 0; j < remote_tensor_results[i].size(); j++) {
            bool correct = std::abs(remote_tensor_results[i][j] - regular_results[i][j]) <= eps;
            correct = correct 
                && !std::isnan(remote_tensor_results[i][j]) 
                && !std::isnan(regular_results[i][j]);
            if (!correct) {
                results_match = false;
                std::cout << "Mismatch at output " << i << " element " << j << ":\n\t"
                << "remote tensor: " << remote_tensor_results[i][j] << " vs regular: " << regular_results[i][j] << std::endl;
            }
            if (abs(remote_tensor_results[i][j]) < eps)
                zeros += 1;
        }
        float ratio = static_cast<float>(zeros) / remote_tensor_results[i].size();
        std::cout << "Output # " << i << ": Out of " << remote_tensor_results[i].size() << " elements, " << ratio * 100 << "% are zero" << std::endl;
    }
    std::cout << "Test: " << (results_match ? "PASS" : "FAIL") << std::endl;
    return results_match ? EXIT_SUCCESS : EXIT_FAILURE;
}

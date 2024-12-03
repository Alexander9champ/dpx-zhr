#include <onnxruntime/core/providers/cuda/cuda_provider_factory.h>
#include <onnxruntime/core/providers/cpu/cpu_provider_factory.h>
#include <onnxruntime/core/providers/provider_api.h>
#include <onnxruntime/core/providers/shared_provider_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Initialize ONNX Runtime environment
    std::shared_ptr<onnxruntime::InferenceSession> session;
    onnxruntime::SessionOptions options;

    // Enable GPU support
    options.AppendExecutionProvider_CUDA(0);  // Using GPU 0

    onnxruntime::Env env(onnxruntime::ORT_LOGGING_LEVEL_WARNING, "ONNXModel");
    session = std::make_shared<onnxruntime::InferenceSession>(env);

    // Load the model
    session->Load("model.onnx", options);

    // Load input image
    cv::Mat img = cv::imread("test_image.jpg");
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::resize(img, img, cv::Size(224, 224));

    // Preprocess image (normalize)
    img.convertTo(img, CV_32F, 1.0 / 255);
    img = (img - 0.485) / 0.229;

    // Convert image to tensor
    std::vector<float> input_data(img.begin<float>(), img.end<float>());
    std::vector<int64_t> input_shape = {1, 3, 224, 224};

    // Create tensor
    std::shared_ptr<onnxruntime::Ort::GetApi().CreateTensor<float>(input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

    // Run inference
    std::vector<onnxruntime::Value> output_tensors;
    session->Run(onnxruntime::RunOptions(), input_tensors.data(), 1, output_tensors.data(), 1);

    // Process output
    auto& output_tensor = output_tensors[0];
    // Post-process and output results...

    return 0;
}

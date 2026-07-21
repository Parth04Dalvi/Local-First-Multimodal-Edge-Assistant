#include "engine.hpp"
#include <algorithm>

LocalInferenceEngine::LocalInferenceEngine(const std::string& model_path) {
    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "EdgeInferenceEngine");
    
    // Enable intra-op threads for low-latency multi-core processing
    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    std::cout << "[*] Loading ONNX Session from: " << model_path << std::endl;
    session = Ort::Session(env, model_path.c_str(), session_options);
    memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
}

std::string LocalInferenceEngine::generate_response(const std::vector<int64_t>& input_ids, int max_tokens) {
    std::vector<int64_t> current_tokens = input_ids;
    
    for (int i = 0; i < max_tokens; ++i) {
        std::vector<int64_t> attention_mask(current_tokens.size(), 1);
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(current_tokens.size())};

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, current_tokens.data(), current_tokens.size(), input_shape.data(), input_shape.size()));
        input_tensors.push_back(Ort::Value::CreateTensor<int64_t>(
            memory_info, attention_mask.data(), attention_mask.size(), input_shape.data(), input_shape.size()));

        // Run ONNX Session forward pass
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_node_names.data(), input_tensors.data(), input_tensors.size(),
            output_node_names.data(), output_node_names.size()
        );

        // Extract logits & greedy token sampling logic
        float* logits = output_tensors[0].GetTensorMutableData<float>();
        size_t vocab_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()[2];
        
        // Find argmax for the last generated position
        size_t last_token_offset = (current_tokens.size() - 1) * vocab_size;
        int64_t next_token = std::distance(
            logits + last_token_offset,
            std::max_element(logits + last_token_offset, logits + last_token_offset + vocab_size)
        );

        current_tokens.push_back(next_token);

        // Break on standard EOS ID (e.g., 50256 or model-specific)
        if (next_token == 50256) break; 
    }

    return "[Generated sequence of " + std::to_string(current_tokens.size() - input_ids.size()) + " tokens]";
}

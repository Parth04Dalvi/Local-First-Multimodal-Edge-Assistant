#ifndef ENGINE_HPP
#define ENGINE_HPP

#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <iostream>

class LocalInferenceEngine {
public:
    LocalInferenceEngine(const std::string& model_path);
    std::string generate_response(const std::vector<int64_t>& input_ids, int max_tokens = 128);

private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session session{nullptr};
    Ort::MemoryInfo memory_info{nullptr};

    std::vector<const char*> input_node_names = {"input_ids", "attention_mask"};
    std::vector<const char*> output_node_names = {"logits"};
};

#endif

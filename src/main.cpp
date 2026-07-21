#include "include/engine.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    std::string model_file = "./onnx_models/phi3/model_int4.onnx";

    if (argc > 1) {
        model_file = argv[1];
    }

    try {
        LocalInferenceEngine engine(model_file);
        
        // Mock tokenized prompt: "Hello, system status check:"
        std::vector<int64_t> dummy_prompt_ids = {15496, 11, 3124, 3747, 2116, 25};
        
        std::cout << "[*] Executing local edge inference pass..." << std::endl;
        std::string result = engine.generate_response(dummy_prompt_ids, 32);
        std::cout << "[+] Result: " << result << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[-] Error executing ONNX Runtime engine: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

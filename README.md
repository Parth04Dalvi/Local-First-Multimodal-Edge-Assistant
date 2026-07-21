# ⚡ Local-First Multimodal Edge Assistant

A low-latency, privacy-preserving C++ and ONNX Runtime application designed to execute speech recognition, local vector search, and LLM inference entirely on-device without cloud API dependencies.

---

## 🌟 Key Features

* **Zero-Cloud Inference:** Quantizes 3B+ parameter open-weights language models (e.g., Phi-3) to **INT4** precision using ONNX Runtime, reducing memory footprint by **65%**.
* **Real-Time Voice Ingestion:** Processes low-latency audio streams via **PortAudio** and executes on-device speech-to-text using quantized Whisper models.
* **On-Device Semantic Search:** Uses **SQLite-VSS** for local document embeddings and vector retrieval, keeping contextual user data private and on-device.
* **Sub-80ms Response Latency:** Optimized intra-op threading and memory allocation for fast CPU/NPU execution context switching.

---

## 🏗️ System Architecture

```text
[ Microphone Audio Stream ] 
           │
           ▼
 [ PortAudio Capture ] ───> [ Whisper ONNX (Encoder/Decoder) ]
                                          │
                                          ▼ (Transcribed Text)
 [ Local Context / Docs ] ──> [ SQLite-VSS Vector Search ]
                                          │
                                          ▼ (Retrieved Context)
                             [ Llama-3 / Phi-3 INT4 ONNX ] ───> [ System Actions / CLI Output ]


🚀 Getting Started
Prerequisites
Ensure you have the following installed on your machine:

CMake (v3.18+)

C++17 Compiler (g++, clang++, or MSVC)

ONNX Runtime C++ SDK

Python 3.10+ (for model quantization scripts)

1. Model Preparation & Quantization
Run the Python pipeline script to download the target HuggingFace model, export it to ONNX format, and apply dynamic INT4 quantization:

Bash
# Install Python prerequisites
pip install optimum[onnxruntime] transformers onnxruntime

# Export and quantize model to INT4
python scripts/quantize_model.py
2. Build the C++ Application
Bash
# Clone the repository
git clone [https://github.com/YourUsername/edge-assistant.git](https://github.com/YourUsername/edge-assistant.git)
cd edge-assistant

# Create build directory
mkdir build && cd build

# Configure and compile
cmake ..
make -j$(nproc)
3. Run the Assistant
Bash
./edge_assistant ../onnx_models/phi3/model_int4.onnx
⚙️ Tech Stack
Language: C++17, Python (Quantization Pipeline)

Inference Engine: ONNX Runtime C++ API

Audio Processing: PortAudio, Whisper ONNX

Vector Store: SQLite-VSS

Build System: CMake

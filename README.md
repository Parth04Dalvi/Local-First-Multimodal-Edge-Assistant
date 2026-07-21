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

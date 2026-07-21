import os
from pathlib import Path
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer
from onnxruntime.quantization import quantize_dynamic, QuantType

MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
OUTPUT_DIR = Path("./onnx_models/phi3")

def export_and_quantize():
    print(f"[*] Downloading and exporting {MODEL_ID} to ONNX...")
    model = ORTModelForCausalLM.from_pretrained(MODEL_ID, export=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    raw_model_path = OUTPUT_DIR / "model.onnx"
    quant_model_path = OUTPUT_DIR / "model_int4.onnx"

    print("[*] Applying INT4 Dynamic Quantization...")
    quantize_dynamic(
        model_input=raw_model_path,
        model_output=quant_model_path,
        weight_type=QuantType.QUInt8 # INT8/INT4 quantization for edge CPUs
    )
    print(f"[+] Model successfully exported to: {quant_model_path}")

if __name__ == "__main__":
    export_and_quantize()

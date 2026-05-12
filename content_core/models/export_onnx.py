# content_core/models/export_onnx.py
"""BGE-base-zh → ONNX + INT8 导出（传统导出器，权重内嵌）"""
import io
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from transformers import AutoTokenizer, AutoModel
from onnxruntime.quantization import quantize_dynamic, QuantType
import torch

MODEL_NAME = "BAAI/bge-base-zh-v1.5"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
FP32_PATH = os.path.join(OUTPUT_DIR, "bge_base_zh_fp32.onnx")
INT8_PATH = os.path.join(OUTPUT_DIR, "bge_base_zh_int8.onnx")


def export():
    print(f"加载模型 {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    dummy_input = tokenizer("测试", return_tensors="pt")

    print("导出 ONNX (FP32, dynamo=False) ...")
    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        FP32_PATH,
        dynamo=False,  # 传统导出器，权重直接嵌入 ONNX 文件
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids":         {0: "batch_size", 1: "sequence_length"},
            "attention_mask":    {0: "batch_size", 1: "sequence_length"},
            "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    fp32_size = os.path.getsize(FP32_PATH) / 1024 / 1024
    print(f"  → {FP32_PATH} ({fp32_size:.1f} MB)")

    print("验证输出一致性 ...")
    _verify(model, tokenizer)

    print("量化 INT8 ...")
    quantize_dynamic(FP32_PATH, INT8_PATH, weight_type=QuantType.QInt8)
    int8_size = os.path.getsize(INT8_PATH) / 1024 / 1024
    print(f"  → {INT8_PATH} ({int8_size:.1f} MB) ({fp32_size/int8_size:.1f}x)")

    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"  删除临时 FP32 ...")
    os.remove(FP32_PATH)
    print("完成！")


def _verify(model, tokenizer):
    """FP32 ONNX vs PyTorch 输出一致性检查"""
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(FP32_PATH, providers=["CPUExecutionProvider"])
    texts = ["介绍一下transformer", "测试"]

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            pt_out = model(**inputs).last_hidden_state.numpy()

        feed = {
            "input_ids": inputs["input_ids"].numpy(),
            "attention_mask": inputs["attention_mask"].numpy(),
        }
        onnx_out = session.run(["last_hidden_state"], feed)[0]

        cos = float(np.dot(onnx_out.ravel(), pt_out.ravel()) / (
            np.linalg.norm(onnx_out) * np.linalg.norm(pt_out)))
        max_diff = float(np.abs(onnx_out - pt_out).max())
        print(f"  text='{text}' → cos={cos:.6f} max_diff={max_diff:.6f} "
              f"{'✓' if cos > 0.9999 else '✗'}")


if __name__ == "__main__":
    export()

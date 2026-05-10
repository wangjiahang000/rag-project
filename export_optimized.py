import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "intent_classifier_output")
OUTPUT_DIR = os.path.join(BASE_DIR, "models", "intent_classifier_optimized")
ONNX_PATH = os.path.join(OUTPUT_DIR, "model.onnx")

os.makedirs(OUTPUT_DIR, exist_ok=True)

id2label = {0: "事实型查询", 1: "定义型查询", 2: "比较型查询",
            3: "操作型查询", 4: "探索型查询", 5: "多跳推理"}

# ============================================================
# 1. 加载原始模型
# ============================================================
print("=" * 50)
print("1/4 加载原始 PyTorch 模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
print("   完成。")

# ============================================================
# 2. 用 optimum 导出 ONNX
# ============================================================
print("2/4 导出 ONNX（FP32）...")
ort_model = ORTModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    export=True,
    provider="CPUExecutionProvider"
)
ort_model.save_pretrained(OUTPUT_DIR)
print("   导出完成。")

# ============================================================
# 3. 手动 INT8 量化
# ============================================================
print("3/4 ONNX Runtime INT8 量化...")
QUANT_PATH = os.path.join(OUTPUT_DIR, "model_quantized.onnx")

quantize_dynamic(
    model_input=ONNX_PATH,
    model_output=QUANT_PATH,
    weight_type=QuantType.QInt8
)

os.remove(ONNX_PATH)
os.rename(QUANT_PATH, ONNX_PATH)
print("   量化完成。")

# ============================================================
# 4. 验证
# ============================================================
print("4/4 验证量化后模型...")
session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
test_input = tokenizer(
    "如何用 LoRA 微调 LLaMA 模型",
    padding="max_length", truncation=True, max_length=128, return_tensors="np"
)
outputs = session.run(None, {
    "input_ids": test_input["input_ids"],
    "attention_mask": test_input["attention_mask"],
    "token_type_ids": test_input["token_type_ids"]
})
logits = outputs[0]
pred = np.argmax(logits, axis=-1)[0]
probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
confidence = probs[0][pred]
print(f"   意图: {id2label[pred]} (置信度: {confidence:.4f})")
print("   验证通过！")

# ============================================================
# 体积对比
# ============================================================
print("=" * 50)
tokenizer.save_pretrained(OUTPUT_DIR)

orig_total = sum(
    os.path.getsize(os.path.join(MODEL_PATH, f))
    for f in os.listdir(MODEL_PATH)
    if os.path.isfile(os.path.join(MODEL_PATH, f))
) / 1024 / 1024

opt_total = sum(
    os.path.getsize(os.path.join(OUTPUT_DIR, f))
    for f in os.listdir(OUTPUT_DIR)
    if os.path.isfile(os.path.join(OUTPUT_DIR, f))
) / 1024 / 1024

print(f"原始 PyTorch 模型文件夹: {orig_total:.1f} MB")
print(f"INT8 量化 ONNX 文件夹: {opt_total:.1f} MB")
print(f"体积缩减: {(1 - opt_total/orig_total)*100:.1f}%")
print(f"\n优化模型已保存到: {OUTPUT_DIR}")
print("第一步完成！")
import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# ============================================================
# 配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "bert-base-chinese")  # 原始 BERT

# 六种意图的关键词语义描述
INTENT_DESCRIPTIONS = {
    "事实型查询": "多少参数 多少层 哪年发表 patch size 多大 hidden size 是多少 多少个版本 输入分辨率多少 参数量是多少 flops 多少 训练用了多少 gpu",
    "定义型查询": "是什么意思 是干嘛的 什么是 怎么理解 解决什么问题 原理是什么 概念是什么 是什么东西 是啥 怎么解释",
    "比较型查询": "有什么区别 哪个好 有什么不同 效果差多少 哪个更主流 哪个更准 有什么不一样 怎么选 对比 比较 区别在哪",
    "操作型查询": "怎么用 怎么配置 怎么训练 怎么部署 怎么实现 怎么微调 怎么加速 怎么量化 什么步骤 什么工具 怎么开",
    "探索型查询": "有没有综述 有没有推荐 有什么新东西 有什么进展 有什么好方法 有什么论文 最新研究 最近工作 找几篇 帮我找 帮我推荐",
    "多跳推理": "怎么一步步 怎么发展过来的 怎么演进的 怎么解决的 从哪篇开始 后来怎么改进 关键节点是什么 发展脉络 技术演进 怎么从 到 再到"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 加载原始 BERT
# ============================================================
print(f"加载原始 BERT: {MODEL_PATH}")
model = SentenceTransformer(MODEL_PATH, device=device)
print(f"句子向量维度: {model.get_sentence_embedding_dimension()}")

print("计算意图描述向量...")
intent_names = list(INTENT_DESCRIPTIONS.keys())
intent_texts = list(INTENT_DESCRIPTIONS.values())

# 对意图描述做均值池化（多个关键词取平均）
intent_embeddings = []
for text in intent_texts:
    # 把关键词拆开分别编码再取平均，让向量覆盖更多表达
    words = text.split()
    word_vecs = model.encode(words, convert_to_tensor=True, show_progress_bar=False)
    mean_vec = word_vecs.mean(dim=0)
    intent_embeddings.append(mean_vec)

intent_embeddings = torch.stack(intent_embeddings)
intent_embeddings = torch.nn.functional.normalize(intent_embeddings, p=2, dim=-1)

print(f"模型就绪，设备: {device}\n")

# ============================================================
# 推理函数（带温度缩放）
# ============================================================
def analyze(query: str, temperature: float = 0.2) -> dict:
    query_vec = model.encode(query, convert_to_tensor=True, show_progress_bar=False)
    query_vec = torch.nn.functional.normalize(query_vec, p=2, dim=-1)

    # 余弦相似度
    similarities = torch.nn.functional.cosine_similarity(
        query_vec.unsqueeze(0),
        intent_embeddings,
        dim=-1
    )

    # 温度缩放：放大差异
    scaled = similarities / temperature
    scores = torch.softmax(scaled, dim=-1).cpu().numpy()

    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    return {
        "intent": intent_names[best_idx],
        "confidence": round(best_score, 4),
        "all_scores": {
            intent_names[i]: round(float(scores[i]), 4)
            for i in range(len(intent_names))
        }
    }

# ============================================================
# 循环输入
# ============================================================
while True:
    text = input("输入查询（输入 q 退出）: ").strip()
    if text.lower() == 'q':
        print("退出。")
        break

    result = analyze(text)

    print(f"  意图: {result['intent']}")
    print(f"  置信度: {result['confidence']:.4f}")
    print(f"  所有相似度分数:")
    for intent, score in result["all_scores"].items():
        bar = "█" * min(int(score * 30), 30)
        print(f"    {intent}: {score:.4f} {bar}")
    print("-" * 50)
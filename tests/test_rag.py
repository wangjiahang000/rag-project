# tests/test_rag.py
import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from content_core.models.sbert_classifier import SBERTClassifier

MODEL_C = "BAAI/bge-base-zh-v1.5"                 # 中文优化(base) - ONNX INT8 量化版

QUERIES_FILE = os.path.join(os.path.dirname(__file__), "test_queries_v2.txt")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "a.txt")

def compute_scores(clf, query):
    query_emb = clf.model.encode(query)
    query_norm = np.linalg.norm(query_emb)
    results = {}
    for intent in clf.INTENT_LABELS:
        anchor_embs = clf.anchor_embs[intent]
        sims = np.dot(anchor_embs, query_emb) / (
            np.linalg.norm(anchor_embs, axis=1) * query_norm + 1e-10
        )
        k = min(3, len(sims))
        top_sims = np.partition(sims, -k)[-k:]
        raw = top_sims.mean()

        neg_embs = clf.negative_embs[intent]
        neg_sims = np.dot(neg_embs, query_emb) / (
            np.linalg.norm(neg_embs, axis=1) * query_norm + 1e-10
        )
        if raw > 0.35:
            penalty = neg_sims.max() * 0.15
            adj = max(0.0, raw - penalty)
        else:
            adj = raw
        results[intent] = (raw, adj)
    return results

def run_batch(queries, file=None):
    _print = lambda s: print(s, file=file)
    for q in queries:
        res = compute_scores(clf_c, q)

        _print(f"\n查询: {q}")
        _print(f"{'意图':12s}  {'校准分数':8s}  {'可视化 (█ 越密集越高)':50s}")
        _print("-" * 75)
        for intent in clf_c.INTENT_LABELS:
            raw_val, adj_val = res[intent]
            bar = "█" * int(adj_val * 30) + "░" * (30 - int(adj_val * 30))
            _print(f"  {intent:12s}  {adj_val:.4f}    {bar}")
        _print("-" * 75)

def load_queries(path):
    queries = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                queries.append(line)
    return queries

clf_c = SBERTClassifier(model_name=MODEL_C, threshold=0.30)

# ── 批量模式：读文件 → 写 a.txt ──
queries = load_queries(QUERIES_FILE)

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write("SBERT 单模型测试 — ONNX INT8 量化版\n")
    f.write(f"模型: {MODEL_C} (ONNX INT8)\n")
    f.write("= 校准后分数 | █ 越密集越高\n")
    f.write("=" * 75 + "\n")
    run_batch(queries, file=f)

print(f"批量测试完成，结果已写入 {OUTPUT_FILE}")

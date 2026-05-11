# core/graph_executor.py

import os
import jieba
import chromadb
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from litellm import completion

load_dotenv()


class ToolRegistry:
    def __init__(self):
        self.chroma = chromadb.Client()
        self.collection = self.chroma.get_or_create_collection("docs")
        self._init_test_data()

    def _init_test_data(self):
        if self.collection.count() > 0:
            return
        docs = [
            "Transformer 是一种基于自注意力机制的神经网络架构，由 Vaswani 等人在 2017 年提出。",
            "BERT 的参数量为 110M（Base）和 340M（Large），采用双向编码器。",
            "GPT 系列使用自回归方式生成文本，参数量从 117M 到 175B 不等。",
            "某某某主要研究知识图谱推理与补全，提出了基于图神经网络的推理框架。",
            "A 的研究方向是知识图谱构建与知识融合，代表作包括知识融合框架 KFusion。",
        ]
        ids = [f"d{i}" for i in range(len(docs))]
        self.collection.add(documents=docs, ids=ids)

    def run(self, op: str, **kwargs):
        if op == "hybrid_search":
            return self._hybrid_search(**kwargs)
        elif op == "rerank":
            return self._rerank(**kwargs)
        elif op == "compare":
            return self._compare(**kwargs)
        elif op == "summarize":
            return self._summarize(**kwargs)
        elif op == "extract":
            return self._extract(**kwargs)
        else:
            raise ValueError(f"未知工具: {op}")

    def _hybrid_search(self, query: str, k: int = 5, vec_weight: float = 0.7, bm25_weight: float = 0.3):
        all_docs = self.collection.get()
        if not all_docs["documents"]:
            return []

        vec_results = self.collection.query(query_texts=[query], n_results=k)
        vec_docs = vec_results["documents"][0]
        vec_distances = vec_results.get("distances", [[]])[0]
        vec_scores = {d: 1 - (dist / 2) for d, dist in zip(vec_docs, vec_distances)} if vec_distances else {}

        tokenized = [jieba.lcut(d) for d in all_docs["documents"]]
        bm25 = BM25Okapi(tokenized)
        bm25_scores = bm25.get_scores(jieba.lcut(query))
        bm25_sorted = sorted(zip(all_docs["documents"], bm25_scores), key=lambda x: x[1], reverse=True)
        bm25_max = bm25_sorted[0][1] if bm25_sorted else 1
        bm25_scores = {d: s / bm25_max for d, s in zip(all_docs["documents"], bm25_scores)}

        final = {}
        for doc in all_docs["documents"]:
            v = vec_scores.get(doc, 0)
            b = bm25_scores.get(doc, 0)
            final[doc] = vec_weight * v + bm25_weight * b

        ranked = sorted(final.items(), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked[:k]]

    def _rerank(self, docs: list, query: str):
        if len(docs) <= 1:
            return docs
        tokenized = [jieba.lcut(d) for d in docs]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(jieba.lcut(query))
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked]

    def _compare(self, docs_a: list, docs_b: list):
        response = completion(
            model="deepseek/deepseek-chat",
            messages=[{
                "role": "user",
                "content": f"对比以下两组文档：\n\nA组：\n{chr(10).join(docs_a)}\n\nB组：\n{chr(10).join(docs_b)}\n\n从核心观点、方法、结论三方面对比异同。"
            }],
            temperature=0,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            api_base=os.getenv("DEEPSEEK_BASE_URL")
        )
        return response.choices[0].message.content

    def _summarize(self, docs: list):
        response = completion(
            model="deepseek/deepseek-chat",
            messages=[{
                "role": "user",
                "content": f"用3-5句话总结以下文档的核心内容：\n\n{chr(10).join(docs)}"
            }],
            temperature=0,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            api_base=os.getenv("DEEPSEEK_BASE_URL")
        )
        return response.choices[0].message.content

    def _extract(self, docs: list, target: str):
        response = completion(
            model="deepseek/deepseek-chat",
            messages=[{
                "role": "user",
                "content": f"从以下文档中提取关于'{target}'的具体信息：\n\n{chr(10).join(docs)}"
            }],
            temperature=0,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            api_base=os.getenv("DEEPSEEK_BASE_URL")
        )
        return response.choices[0].message.content


class GraphExecutor:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def execute(self, graph):
        results = {}
        for node in graph.nodes:
            args = {}
            for k, v in node.args.items():
                if isinstance(v, str) and v.startswith("{{") and v.endswith("}}"):
                    dep_id = v.strip("{}")
                    args[k] = results.get(dep_id, "")
                else:
                    args[k] = v
            results[node.id] = self.registry.run(node.op, **args)
        return results
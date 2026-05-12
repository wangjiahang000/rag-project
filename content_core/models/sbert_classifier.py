# content_core/models/sbert_classifier.py
import os
# 使用 HuggingFace 国内镜像加速下载
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict


# ── ONNX INT8 推理引擎（替换 SentenceTransformer.encode 接口）──
class _ONNXEncoder:
    """用 onnxruntime 替换 SentenceTransformer，提供相同的 .encode() 接口

    内部自己管理 tokenizer + ONNX session，使用 mean pooling + L2 normalize。
    """

    def __init__(self, onnx_path: str, model_dir: str):
        import onnxruntime
        from transformers import AutoTokenizer

        self.session = onnxruntime.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        # 读取模型期望的输入名，只送必要字段
        self._input_names = {i.name for i in self.session.get_inputs()}

    def encode(self, sentences, convert_to_numpy=True, **kwargs):
        if isinstance(sentences, str):
            sentences = [sentences]
        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="np",
            max_length=512,
        )
        feed = {k: v for k, v in inputs.items() if k in self._input_names}
        outputs = self.session.run(["last_hidden_state"], feed)[0]
        # BGE 使用 CLS token pooling (取 [CLS] 向量)
        emb = outputs[:, 0, :]
        # L2 normalize
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        # 单条输入返回 1D，兼容 SentenceTransformer 行为
        return emb[0] if len(sentences) == 1 else emb


class SBERTClassifier:
    """多标签意图分类器（6 类）—— 基于高质量锚点的零样本语义分类

    核心思路：
      - 每个意图预写一组语义集中的锚点句
      - 推理时：query 与每个锚点算余弦相似度，取 top-3 均值作为意图分数
      - 用负例锚点做对比校准，抑制易混淆的误判
      - 每个意图独立打分，互不影响，天然支持多意图输出
    """

    INTENT_LABELS = ["retrieve", "compare", "summarize", "howto", "reason", "extract"]

    # ── 正例锚点（每个意图 12~15 条，语义集中）────
    ANCHORS = {
        "retrieve": [
            "帮我查一下关于这个主题的资料",
            "搜索一下相关的研究文献",
            "找几篇关于这个方向的论文",
            "我想检索一下这方面的信息",
            "查一查最近有什么相关研究",
            "有没有关于这个领域的综述文章",
            "寻找关于这个技术的介绍和资料",
            "帮我找一些参考资料来看看",
            "我想看看这方面有哪些研究成果",
            "搜索某个技术或方法的相关论文",
            "有哪些文献是关于这个方向的",
            "帮我查查有没有人研究过这个问题",
            "查一下某个概念的定义和解释",
            "什么是注意力机制",
            "介绍一下某个技术的原理和应用",
            "介绍一下某个方向的基本概念",
            "介绍一下这个技术是怎么回事",
            "简单介绍一下什么是",
            "帮我介绍一下这个概念",
            "帮我介绍一下",
            "介绍一下这个技术",
            "给我介绍一下这个东西",
            "找一些关于模型训练的资料",
        ],
        "compare": [
            "比较一下A和B有什么区别",
            "对比这两种方法的不同之处",
            "分析一下这两个模型的优缺点",
            "A和B相比哪个更好更优",
            "这两个方案有什么差异和共同点",
            "从多个角度对比一下它们的表现",
            "这两个技术各有什么优劣",
            "横向对比一下它们的性能差异",
            "比较一下准确率方面的差别",
            "A和B各有什么特点和不足",
            "这两种做法的优劣分别是什么",
            "从不同维度对比这两个方案",
            "A和B相比哪个更强",
            "这两个模型哪个表现更好效果更好",
        ],
        "summarize": [
            "总结一下这个领域的研究进展",
            "概括这些文献的核心观点",
            "归纳一下目前的主要研究方向",
            "把这几篇文章的要点提炼出来",
            "概述一下当前的研究现状",
            "整体情况怎么样帮我归纳一下",
            "梳理一下这个方向的发展脉络",
            "汇总一下目前已有的研究成果",
            "用几句话总结这些资料的关键内容",
            "提炼这些文献的共同点和主题方向",
            "整理一下这些论文的核心发现",
        ],
        "howto": [
            "如何实现这个功能",
            "怎么配置和部署这个系统",
            "具体操作步骤是什么",
            "教我怎么搭建这个环境",
            "有没有详细的教程可以参考",
            "做这个需要哪些步骤和工具",
            "怎么开始入门学习这个技术",
            "实际开发中应该怎么做",
            "能给我一个具体的例子演示吗",
            "如果我想自己复现应该怎么操作",
            "实现流程是怎样的有哪些注意事项",
            "从零开始搭建需要哪些准备工作",
            "用代码怎么具体实现这个算法",
            "实际动手做的时候具体的步骤是什么",
            "详细的实现流程和注意事项有哪些",
            "写代码实现一个具体功能应该怎么做",
            "在框架中编写模型的核心代码",
            "编程实现这个算法需要写哪些代码",
        ],
        "reason": [
            "为什么会出现这种现象",
            "导致这个结果的原因是什么",
            "背后的原理和机制是什么",
            "是什么因素导致了这样的结果",
            "怎么解释这个现象背后的逻辑",
            "为什么会这样背后的深层原因是什么",
            "什么因素造成了这种差异",
            "这个结果的深层原因是什么",
            "解释一下为什么这样设计",
            "有什么理论依据支撑这个结论",
            "背后的推导过程和原理是怎样的",
            "解释一下这个现象是怎么产生的",
            "解释一下为什么这样做效果更好",
            "为什么说这种方法更有效有什么依据",
            "这样设计有什么道理和依据",
            "怎么理解这个结论背后的逻辑",
            "解释一下某个技术的核心原理",
            "为什么说这种设计比传统方案更好",
            "产生这个结果的具体原因是什么",
            "解释一下为什么会得出这样的结论",
        ],
        "extract": [
            "从这段话中提取具体的数值信息",
            "从论文里找出实验参数和指标",
            "抽取这篇文章中的关键数据",
            "从文本中获取准确率是多少",
            "找到文中提到的训练数据量有多大",
            "提取具体的技术指标和性能数值",
            "从文档中查一下用了什么数据集",
            "从这些内容中找出具体的参数值",
            "把这个实验的数值信息提取出来",
            "从文档中找某个具体指标是多少",
        ],
    }

    # ── 负例锚点（语义上与目标意图相似但不属于它，用于对比校准）──
    NEGATIVE_ANCHORS = {
        "retrieve": [
            "总结提炼一下这些文献的核心观点",
            "比较一下这两种方法有什么区别",
            "教我怎么一步步实现这个功能",
            "从这段文字中把准确率提取出来",
            "解释一下为什么会得出这个结论",
            "解释一下这个概念的深层含义是什么",
        ],
        "compare": [
            "分别介绍一下A和B各自的原理",
            "帮我查一下A的相关资料",
            "从这个实验中提取性能数据",
            "总结一下A和B的研究现状",
            "教我怎么选择和使用这两种方法",
            "分别了解A方法和B方法各自怎么用",
        ],
        "summarize": [
            "搜索关于这个方向的最新论文",
            "比较这两个模型有什么差异",
            "怎么部署这个模型到生产环境",
            "从文档中提取实验参数",
            "解释一下这个技术的基本原理",
            "帮我介绍一下这方面的最新进展",
        ],
        "howto": [
            "什么是Transformer解释一下原理",
            "对比一下这两个框架的优缺点",
            "为什么这个模型效果更好",
            "从论文里提取训练参数",
            "搜索一下这个功能的相关资料和文档",
            "总结一下搭建环境的几个关键步骤",
        ],
        "reason": [
            "给出这个技术的定义和基本概念",
            "帮我总结一下这篇论文的内容",
            "查一下这个实验用了什么数据集",
            "比较这两篇论文的方法差异",
            "按步骤推导出这个结论",
            "找出数据中导致差异的关键因素",
        ],
        "extract": [
            "总结一下这些文献的核心观点",
            "怎么实现这个功能给我讲讲",
            "哪个模型的效果更好",
            "为什么模型越大效果越好",
            "搜索一下实验用到的数据集和指标",
            "对比不同模型在这个数据集上的结果",
        ],
    }

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-zh-v1.5",
        threshold: float = 0.30,
        use_onnx: bool = False,
    ):
        # 自动检测 ONNX INT8 模型（默认行为，显式 use_onnx=True/False 可覆盖）
        onnx_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "bge_base_zh_int8.onnx"
        )
        _onnx_available = os.path.exists(onnx_path)

        if use_onnx or (not use_onnx and _onnx_available and model_name == "BAAI/bge-base-zh-v1.5"):
            if not os.path.exists(onnx_path):
                raise FileNotFoundError(
                    f"ONNX 模型不存在: {onnx_path}\n"
                    f"请先运行 content_core/models/export_onnx.py 导出"
                )
            model_dir = os.path.dirname(os.path.abspath(__file__))
            self.model = _ONNXEncoder(onnx_path, model_dir)
            self.onnx_mode = True
        else:
            self.model = SentenceTransformer(model_name)
            self.onnx_mode = False

        self.threshold = threshold
        self.anchor_embs = self._encode_anchors()
        self.negative_embs = self._encode_negative()

    # ── 编码 ────────────────────────────────

    def _encode_anchors(self) -> Dict[str, np.ndarray]:
        """预计算所有意图的锚点向量矩阵 {intent: (N, D)}"""
        return {
            intent: self.model.encode(anchors, convert_to_numpy=True)
            for intent, anchors in self.ANCHORS.items()
        }

    def _encode_negative(self) -> Dict[str, np.ndarray]:
        """预计算负例锚点矩阵"""
        return {
            intent: self.model.encode(anchors, convert_to_numpy=True)
            for intent, anchors in self.NEGATIVE_ANCHORS.items()
        }

    # ── 单条分类 ────────────────────────────

    def classify(self, query: str) -> Tuple[List[str], List[float]]:
        """返回：(排序后的标签列表, 对应置信度列表)
        每个意图独立打分，互不影响，天然支持多意图输出。
        """
        query_emb = self.model.encode(query, convert_to_numpy=True)
        query_norm = np.linalg.norm(query_emb)
        if query_norm < 1e-10:
            return [], []

        scores = {}

        for intent in self.INTENT_LABELS:
            # ── 正例匹配：取 top-3 余弦相似度均值 ──
            anchor_embs = self.anchor_embs[intent]
            sims = np.dot(anchor_embs, query_emb) / (
                np.linalg.norm(anchor_embs, axis=1) * query_norm + 1e-10
            )
            k = min(3, len(sims))
            top_sims = np.partition(sims, -k)[-k:]  # top-k（未排序）
            pos_score = top_sims.mean()

            # ── 负例校准：仅正分较高(>0.35)时启用，抑制易混淆意图 ──
            # 弱 query 不做惩罚，避免短句被过度压低
            if pos_score > 0.65:
                neg_embs = self.negative_embs[intent]
                neg_sims = np.dot(neg_embs, query_emb) / (
                    np.linalg.norm(neg_embs, axis=1) * query_norm + 1e-10
                )
                neg_penalty = neg_sims.max() * 0.30
                scores[intent] = max(0.0, pos_score - neg_penalty)
            else:
                scores[intent] = pos_score

        # 过滤高于阈值，按分数降序排列
        sorted_pairs = sorted(
            [(i, s) for i, s in scores.items() if s >= self.threshold],
            key=lambda x: x[1],
            reverse=True,
        )

        if not sorted_pairs:
            return [], []

        return [p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs]

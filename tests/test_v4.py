"""
test_v4.py — 实体抽取完整质量报告

对用户输入做分词 + 实体抽取，输出每个候选词的三维分数：
  - 频次分 Score_freq  (权重 0.3)
  - 凝聚分 Score_pmi   (权重 0.4)
  - 技术相关分 Score_tech (权重 0.3)

运行方式：
  python tests/test_v4.py                       # 内置用例
  python tests/test_v4.py "你的查询文本"          # 自定义输入
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from content_core.tools.process.entity_extractor import get_extractor


def fmt_bar(val: float, width: int = 24) -> str:
    """分数可视化条"""
    filled = int(val * width)
    return "#" * filled + "." * (width - filled)


def show_report(ext, text: str):
    print("\n" + "=" * 72)
    print(f"输入: {text}")
    print("=" * 72)

    # 1. jieba 原始分词
    words = list(ext._pseg.cut(text))
    words_filtered = [(w.word, w.flag) for w in words if w.word.strip()]
    print(f"\n[1] jieba 分词（词性标注）")
    print(f"    {' '.join(f'{w}/{f}' for w, f in words_filtered)}")

    # 2. 实体抽取全结果
    entities = ext.extract(text)
    print(f"\n[2] 实体抽取结果")
    if entities:
        for e in entities:
            print(f"    \"{e['text']}\"  [{e['type']}]  pos={e['start']}:{e['end']}")
    else:
        print("    (无)")

    # 3. 候选词质量报告（对每个 UNKNOWN 类型的候选词）
    print(f"\n[3] 候选词质量分报告")

    # 收集本次提取中出现的 UNKNOWN 候选词
    unknown_words = {}
    for e in entities:
        if e["type"] == "UNKNOWN":
            w = e["text"]
            cnt = ext._candidate_counts.get(w, 1)
            unknown_words[w] = cnt

    # 同时也展示 tech_dict 已覆盖的词
    tech_hits = [e["text"] for e in entities if e["type"] == "TECH"]
    if tech_hits:
        print(f"\n    已覆盖（TECH）:")
        for w in tech_hits:
            # 展示词性和频次
            freq = ext.tech_dict.get(w.lower(), "?")
            print(f"    - \"{w}\"  (词频: {freq})")

    if unknown_words:
        print(f"\n    候选词（UNKNOWN，次数≥{ext._candidate_counts.get(list(unknown_words.keys())[0] if unknown_words else '', 0)}）:")
        print(f"    {'词':<22} {'次数':>4} {'频次分':>7} {'凝聚分':>7} {'技术分':>7} {'综合分':>7}  {'趋势'}")
        print(f"    {'-' * 72}")

        for w, cnt in sorted(unknown_words.items(), key=lambda x: -x[1]):
            score_freq = min(cnt / 10.0, 1.0)
            score_pmi = ext._compute_pmi(w)
            score_tech = ext._compute_tech_similarity(w)
            combined = round(0.3 * score_freq + 0.4 * score_pmi + 0.3 * score_tech, 2)

            bar_freq = fmt_bar(score_freq)
            bar_pmi = fmt_bar(score_pmi)
            bar_tech = fmt_bar(score_tech)
            bar_all = fmt_bar(combined)

            threshold_ok = "[OK]" if combined >= 0.55 else "   "
            admit_note = ""
            if combined >= 0.55:
                # 计算需要达到当前分数的模拟出现次数
                admit_note = "  <- 达到准入线"
            else:
                # 达到 0.55 需要的出现次数（保持其他分不变）
                needed_freq = (0.55 - 0.4 * score_pmi - 0.3 * score_tech) / 0.3
                if needed_freq > 0:
                    needed_count = int(needed_freq * 10) + 1
                    if needed_count <= 30:
                        admit_note = f"  还需 {max(1, needed_count - cnt)} 次"
                    else:
                        admit_note = "  凝聚分或技术分不足"

            print(f"    {w:<20} {cnt:>4}  {score_freq:.3f} {score_pmi:.3f} {score_tech:.3f} {combined:.3f}  {bar_all} {admit_note}")
    else:
        print("    (本次无候选词)")

    # 4. 各维度分数分解说明
    print(f"\n[4] 分数说明")
    print(f"    频次分   = min(出现次数/10, 1.0)                                   权重 0.3")
    print(f"    凝聚分   = min(相邻字符最小PMI / 8.0, 1.0)                          权重 0.4")
    print(f"    技术分   = 与tech_dict最长公共子串比例                               权重 0.3")
    print(f"    综合分   = 0.3*频次 + 0.4*凝聚 + 0.3*技术")
    print(f"    准入线   = 综合分 >= 0.55 且累计出现 >= 3 次")


def main():
    ext = get_extractor()

    if len(sys.argv) > 1:
        # 命令行自定义输入
        show_report(ext, sys.argv[1])
    else:
        # 内置测试用例
        cases = [
            "LoRA和QLoRA有什么区别",
            "对比分析CNN和Transformer的优缺点",
            "你觉得RAG跟微调哪个更靠谱",
            "介绍一下BERT的原理",
            "怎么用PyTorch训练模型",
        ]
        for q in cases:
            show_report(ext, q)

        # 输出 JSON
        output_path = os.path.join(os.path.dirname(__file__), "test_v4_result.json")
        print(f"\n\n详细报告已保存至: {output_path}")

    # 展示 candidates.txt 当前状态
    cand_path = os.path.join(ext.memory_dir, "candidates.txt")
    if os.path.exists(cand_path):
        with open(cand_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) > 1:
            print(f"\n{'=' * 72}")
            print(f"candidates.txt 当前状态 ({len(lines) - 1} 条候选)")
            print(f"{'=' * 72}")
            print(f"    {'词':<20} {'次数':>4}  {'质量分':>6}  {'最近命中'}")
            print(f"    {'-' * 42}")
            for line in lines[1:]:
                parts = line.strip().split("\t")
                if len(parts) >= 4:
                    print(f"    {parts[0]:<20} {parts[1]:>4}  {parts[2]:>6}  {parts[3]}")


if __name__ == "__main__":
    main()

"""
实体抽取引擎 — 基于 jieba + 技术词典 + 质量分闭环

设计：
  - tech_dict.txt：人工维护的技术实体，格式 "词 词频"
  - candidates.txt：候选词发现，质量分达标后写入，供人工审查
  - blacklist.txt：人工维护的停用词，jieba.del_word + 输出过滤
  - 质量分 = 0.3×频次分 + 0.4×凝固度 + 0.3×技术相关性
  - 文件修改自动检测，下次提取前重载
"""

import os
import re
import math
import json
import time
import logging
from typing import List, Tuple, Optional
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)

# ── 权重常量 ──
W_FREQ = 0.3
W_PMI = 0.4
W_TECH = 0.3
ADMIT_THRESHOLD = 0.55
CANDIDATE_MIN_COUNT = 3
SCORE_FREQ_MAX = 10  # 出现 10 次频次分满分
PMI_MAX = 8.0  # PMI 满分阈值
INITIAL_BG_FREQ = "__bg_freq__.json"


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


class EntityExtractor:
    """实体抽取器，含 jieba 配置、候选词发现、质量分计算、自动重载"""

    def __init__(self, memory_dir: str = "memory"):
        self.memory_dir = memory_dir
        _ensure_dir(memory_dir)

        # 运行时状态
        self.tech_dict: dict = {}        # 词 -> 词频
        self.blacklist: set = set()
        self._char_freq: dict = {}       # 单字频次（PMI 背景）
        self._bigram_freq: dict = {}     # 二元频次（PMI 背景）
        self._total_chars = 0
        self._total_bigrams = 0

        # 候选词统计（内存）
        self._candidate_counts: dict = defaultdict(int)
        self._candidate_written: set = set()

        # 文件 mtime 跟踪
        self._file_mtimes: dict = {}

        self._init_jieba()
        self._load_all()
        self._build_bg_freq()

    # ── 初始化 ─────────────────────────────

    def _init_jieba(self):
        """首次加载 jieba + 用户词典"""
        import jieba
        import jieba.posseg as pseg

        dict_path = os.path.join(self.memory_dir, "tech_dict.txt")
        if os.path.exists(dict_path):
            jieba.load_userdict(dict_path)
            logger.info("加载用户词典: %s", dict_path)

        black_path = os.path.join(self.memory_dir, "blacklist.txt")
        if os.path.exists(black_path):
            with open(black_path, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip()
                    if word:
                        jieba.del_word(word)
                        self.blacklist.add(word)
            logger.info("加载黑名单: %d 词", len(self.blacklist))

        self._jieba = jieba
        self._pseg = pseg

    def _load_all(self):
        """加载所有词典文件"""
        dict_path = os.path.join(self.memory_dir, "tech_dict.txt")
        if os.path.exists(dict_path):
            self.tech_dict = {}
            with open(dict_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        freq = int(parts[-1])
                        word = ' '.join(parts[:-1]).lower()
                        # 大小写不敏感：同义词取最大词频
                        if word in self.tech_dict:
                            self.tech_dict[word] = max(self.tech_dict[word], freq)
                        else:
                            self.tech_dict[word] = freq
            self._file_mtimes[dict_path] = os.path.getmtime(dict_path)
            logger.info("技术词典载入: %d 词", len(self.tech_dict))

        cand_path = os.path.join(self.memory_dir, "candidates.txt")
        if os.path.exists(cand_path):
            self._file_mtimes[cand_path] = os.path.getmtime(cand_path)
            # 仅记录 mtime，candidates 内容在 _update_candidates 中写入

        black_path = os.path.join(self.memory_dir, "blacklist.txt")
        if os.path.exists(black_path):
            self.blacklist = set()
            with open(black_path, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip()
                    if word:
                        self.blacklist.add(word)
            self._file_mtimes[black_path] = os.path.getmtime(black_path)

    def _build_bg_freq(self):
        """从 tech_dict 构建静态单字/二元频次表，用于 PMI 计算

        原理：技术词典中的词代表领域核心词汇，其内部字符的共现
        模式可以反映领域内的语言统计特征。虽然粒度粗，但对技术
        实体的内部凝固度估算已经够用。
        """
        self._char_freq = defaultdict(int)
        self._bigram_freq = defaultdict(int)

        for word in self.tech_dict:
            wf = max(1, self.tech_dict[word])
            # 英文字母转小写后按字符处理
            text = word.lower()
            for ch in text:
                self._char_freq[ch] += wf
                self._total_chars += wf
            for i in range(len(text) - 1):
                bigram = text[i:i + 2]
                self._bigram_freq[bigram] += wf
                self._total_bigrams += wf

        # 加平滑，避免 log(0)
        for ch in set("abcdefghijklmnopqrstuvwxyz0123456789中文技术实体神经网络卷积"):
            if ch not in self._char_freq:
                self._char_freq[ch] = 1
                self._total_chars += 1

        logger.info("背景频率表: %d 单字, %d 二元组", len(self._char_freq), len(self._bigram_freq))

    # ── 重载 ───────────────────────────────

    def _check_reload(self):
        """检查文件变更，必要时重新加载 jieba 实例"""
        import jieba
        import jieba.posseg as pseg

        changed = False
        for fname in ["tech_dict.txt", "candidates.txt", "blacklist.txt"]:
            fpath = os.path.join(self.memory_dir, fname)
            if not os.path.exists(fpath):
                continue
            mtime = os.path.getmtime(fpath)
            prev = self._file_mtimes.get(fpath)
            if prev is not None and mtime > prev:
                changed = True
            self._file_mtimes[fpath] = mtime

        if not changed:
            return

        logger.info("词典文件变更，重载中...")
        # 用 importlib.reload 重置 jieba 模块到干净状态，避免手动替换函数的副作用
        import importlib
        import jieba as jieba_mod
        importlib.reload(jieba_mod)
        self._init_jieba()
        self._load_all()
        self._build_bg_freq()
        logger.info("重载完成")

    # ── 提取 ───────────────────────────────

    def extract(self, text: str) -> List[dict]:
        """主抽取方法

        返回:
          [{"text": "LoRA", "start": 0, "end": 4, "type": "TECH"},
           {"text": "从2020到2024年", "start": 5, "end": 14, "type": "TIME",
            "time_relation": "year_range", "start_year": 2020, "end_year": 2024}]
        """
        if not text:
            return []

        self._check_reload()

        words = self._pseg.cut(text)
        entities = self._merge_entities(list(words), text)

        # 追加时间表达式实体
        time_entities = self.extract_time_entities(text)
        entities.extend(time_entities)

        return entities

    def _merge_entities(self, words: list, text: str) -> List[dict]:
        """扫描分词序列，按 技术词典 > 正则 > 未登录 三级匹配"""
        entities = []
        i = 0
        n = len(words)
        covered = [False] * len(text)

        # ── 第一遍：技术词典匹配（字母边界感知，避免子串误吞） ──
        sorted_dict = sorted(self.tech_dict.keys(), key=len, reverse=True)
        for term in sorted_dict:
            pattern = re.compile(
                r'(?<![a-z])' + re.escape(term) + r'(?![a-z])',
                re.IGNORECASE,
            )
            for m in pattern.finditer(text):
                s, e = m.start(), m.end()
                if not any(covered[s:e]):
                    entities.append({
                        "text": text[s:e],
                        "start": s,
                        "end": e,
                        "type": "TECH",
                    })
                    for j in range(s, e):
                        covered[j] = True
                    cand_key = text[s:e].strip()
                    if cand_key in self._candidate_counts:
                        del self._candidate_counts[cand_key]

        # ── 第二遍：正则匹配（版本号、URL 等） ──
        patterns = [
            (r'\b[a-zA-Z]+\d+(?:\.\d+)+\b', 'REGEX'),     # 版本号 v1.2.3
            (r'https?://\S+', 'REGEX'),                    # URL
            (r'[\w.+-]+@[\w.-]+\.\w+', 'REGEX'),          # 邮箱
        ]
        for pat, label in patterns:
            for m in re.finditer(pat, text):
                s, e = m.start(), m.end()
                if not any(covered[s:e]):
                    entities.append({
                        "text": m.group(),
                        "start": s,
                        "end": e,
                        "type": label,
                    })
                    for j in range(s, e):
                        covered[j] = True

        # ── 第三遍：jieba 未覆盖的实体词 ──
        for word_obj in words:
            word = word_obj.word
            flag = word_obj.flag
            pos = text.find(word)
            if pos == -1:
                continue

            # 过滤
            if word in self.blacklist:
                continue
            if len(word.strip()) < 2:
                continue
            if re.match(r'^[\s\d\W_]+$', word):
                continue

            start_pos = pos
            end_pos = pos + len(word)
            if any(covered[start_pos:end_pos]):
                continue

            # 仅保留有意义的词性
            valid_pos = ('n', 'v', 'a', 'eng', 'x', 'nz', 'vn', 'an', 'ns', 'nt', 'nrt', 'nr')
            if not flag.startswith(valid_pos):
                continue

            entity_type = "UNKNOWN"

            # 异步更新候选词统计
            if word not in self.tech_dict:
                self._candidate_counts[word] += 1
                count = self._candidate_counts[word]
                if count >= CANDIDATE_MIN_COUNT and word not in self._candidate_written:
                    self._candidate_written.add(word)
                    self._update_candidates(word, count)

            entities.append({
                "text": word,
                "start": start_pos,
                "end": end_pos,
                "type": entity_type,
            })
            for j in range(start_pos, end_pos):
                covered[j] = True

        return entities

    def extract_entities(self, text: str) -> List[str]:
        """对外简化接口：只返回实体文本列表"""
        return [e["text"] for e in self.extract(text)]

    # ── 时间表达式抽取 ─────────────────────

    def extract_time_entities(self, text: str) -> List[dict]:
        """从文本中抽取时间表达式（年份范围、之前/之后、近N年）

        返回：
          [{"text": "从2020到2024年", "start": 0, "end": 10, "type": "TIME",
            "time_relation": "year_range", "start_year": 2020, "end_year": 2024},
           ...]
        """
        if not text:
            return []

        results = []
        covered = [False] * len(text)

        for label, pat_str in _TIME_PATTERNS:
            pattern = re.compile(pat_str)
            for match in pattern.finditer(text):
                s, e = match.start(), match.end()
                if any(covered[s:e]):
                    continue
                entity = _parse_time_match(label, match, text)
                if entity:
                    results.append(entity)
                    for j in range(s, e):
                        covered[j] = True

        # 补充：裸四位数字年份（如 "2020 之后" 中的 "2020"）
        if not any(covered):
            for match in _YEAR_PATTERN.finditer(text):
                s, e = match.start(), match.end()
                if any(covered[s:e]):
                    continue
                year = int(match.group(1))
                results.append({
                    "text": match.group(),
                    "start": s,
                    "end": e,
                    "type": "TIME",
                    "time_relation": "single_year",
                    "year": year,
                })
                for j in range(s, e):
                    covered[j] = True

        return results

    # ── 候选词管理 ─────────────────────────

    def _update_candidates(self, word: str, count: int):
        """计算质量分，写入 candidates.txt"""
        score = self._compute_quality_score(word, count)

        cand_path = os.path.join(self.memory_dir, "candidates.txt")
        today = datetime.now().strftime("%Y-%m-%d")

        # 读取现有候选
        existing = {}
        if os.path.exists(cand_path):
            with open(cand_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("词\t") or not line.strip():
                        continue
                    parts = line.strip().split("\t")
                    if len(parts) >= 4:
                        existing[parts[0]] = (parts[1], parts[2], parts[3])

        if score >= ADMIT_THRESHOLD:
            existing[word] = (str(count), f"{score:.2f}", today)
            logger.info("候选词准入: %s (count=%d, score=%.2f)", word, count, score)
        else:
            # 分数不够也记录次数，下次累计后再算
            old_count = count
            if word in existing:
                old_count = max(count, int(existing[word][0]))
            existing[word] = (str(old_count), f"{score:.2f}", today)

        # 写回
        with open(cand_path, "w", encoding="utf-8") as f:
            f.write("词\t出现次数\t质量分\t最近命中日期\n")
            for w, (c, s, d) in sorted(existing.items(), key=lambda x: float(x[1][2]) if x[1][2].replace('.','',1).isdigit() else 0, reverse=True):
                f.write(f"{w}\t{c}\t{s}\t{d}\n")

    def _compute_quality_score(self, word: str, count: int) -> float:
        """综合质量分 = 0.3×频次分 + 0.4×凝固度 + 0.3×技术相关性"""
        score_freq = min(count / SCORE_FREQ_MAX, 1.0)
        score_pmi = self._compute_pmi(word)
        score_tech = self._compute_tech_similarity(word)
        combined = W_FREQ * score_freq + W_PMI * score_pmi + W_TECH * score_tech
        return round(combined, 2)

    def _compute_pmi(self, word: str) -> float:
        """PMI 内部凝固度

        对于 len≥2 的词，计算相邻字符对的最小 PMI。
        PMI(w1,w2) = log(P(w1,w2) / (P(w1) * P(w2)))

        背景频次来自 tech_dict 的统计数据。
        """
        text = word.lower()
        if len(text) < 2:
            return 0.0

        pmi_values = []
        for i in range(len(text) - 1):
            ch1, ch2 = text[i], text[i + 1]
            bigram = text[i:i + 2]

            p_ch1 = self._char_freq.get(ch1, 1) / max(self._total_chars, 1)
            p_ch2 = self._char_freq.get(ch2, 1) / max(self._total_chars, 1)
            p_bg = self._bigram_freq.get(bigram, 1) / max(self._total_bigrams, 1)

            if p_ch1 > 0 and p_ch2 > 0 and p_bg > 0:
                pmi = math.log(p_bg / (p_ch1 * p_ch2) + 1e-10)
                pmi_values.append(pmi)

        if not pmi_values:
            return 0.0

        pmi_min = min(pmi_values)
        return min(pmi_min / PMI_MAX, 1.0)

    def _compute_tech_similarity(self, word: str) -> float:
        """技术相关性：与 tech_dict 中最长公共子串的比例"""
        text = word.lower()
        max_lcs = 0
        for term in self.tech_dict:
            t = term.lower()
            lcs = self._longest_common_substring(text, t)
            max_lcs = max(max_lcs, lcs)

        if max_lcs < 2 or len(text) < 2:
            return 0.0
        return max_lcs / len(text)

    @staticmethod
    def _longest_common_substring(a: str, b: str) -> int:
        """最长公共子串长度"""
        m, n = len(a), len(b)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        max_len = 0
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                    max_len = max(max_len, dp[i][j])
        return max_len


# ── 时间表达式抽取 ──────────────────────────

_CN_NUM = {
    '零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4,
    '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
}


def _cn2int(text: str) -> int:
    """中文数字转整数（支持一到十、二十、二十一、一百等）"""
    text = text.strip()
    if not text:
        return 0
    # 单个数字
    if text in _CN_NUM:
        return _CN_NUM[text]
    # "百"级别
    if '百' in text:
        parts = text.split('百')
        hundreds = _CN_NUM.get(parts[0], 1) if parts[0] else 1
        remainder = parts[1] if len(parts) > 1 else ''
        if not remainder:
            return hundreds * 100
        tens = _cn2int(remainder)
        return hundreds * 100 + tens
    # "十"级别：二十、二十一
    if '十' in text:
        parts = text.split('十')
        tens = _CN_NUM.get(parts[0], 1) if parts[0] else 1
        ones = _CN_NUM.get(parts[1], 0) if len(parts) > 1 and parts[1] else 0
        return tens * 10 + ones
    try:
        return int(text)
    except ValueError:
        return 0


# 时间表达式正则（顺序重要：先具体后通用）
_TIME_PATTERNS: List[Tuple[str, str]] = [
    # 从X到Y年 / X到Y年 / X-Y年
    ("TIME_RANGE", r'(?:从)?(\d{4})\s*年?\s*(?:到|至|[-–—])\s*(\d{4})\s*年?'),
    # X年之后/以后
    ("TIME_AFTER", r'(\d{4})\s*年\s*(?:之?[后後])'),
    # X年之前/以前
    ("TIME_BEFORE", r'(\d{4})\s*年\s*(?:之?[前前])'),
    # 最近N年 / 近N年
    ("TIME_RECENT", r'(?:最近|近)\s*(\d+|[一二两三四五六七八九十百]+)\s*年'),
    # 近年来 / 近些年
    ("TIME_RECENT_GENERIC", r'(?:近些?年[来來]?)'),
    # 具体年份 YYYY年（不跟"到/至/-"后面，避免抢 range）
    ("TIME_YEAR", r'(?<!\d)(\d{4})\s*年(?!\s*(?:到|至|[-–—]\s*\d))'),
]

# 单一年份匹配（非中文语境）
_YEAR_PATTERN = re.compile(r'(?<!\d)(\d{4})(?!\s*年)(?!\d)')


def _parse_time_match(label: str, match: re.Match, text: str) -> Optional[dict]:
    """将正则匹配结果解析为结构化时间实体"""
    entity = {
        "text": match.group(),
        "start": match.start(),
        "end": match.end(),
        "type": "TIME",
    }

    if label == "TIME_RANGE":
        entity["time_relation"] = "year_range"
        entity["start_year"] = int(match.group(1))
        entity["end_year"] = int(match.group(2))

    elif label == "TIME_AFTER":
        entity["time_relation"] = "after_year"
        entity["year"] = int(match.group(1))

    elif label == "TIME_BEFORE":
        entity["time_relation"] = "before_year"
        entity["year"] = int(match.group(1))

    elif label == "TIME_RECENT":
        num_str = match.group(1)
        entity["time_relation"] = "recent_years"
        entity["years_back"] = _cn2int(num_str)

    elif label == "TIME_RECENT_GENERIC":
        entity["time_relation"] = "recent_generic"

    elif label == "TIME_YEAR":
        entity["time_relation"] = "single_year"
        entity["year"] = int(match.group(1))

    return entity


# ── 全局单例（延迟初始化） ──

_global_extractor: Optional[EntityExtractor] = None


def get_extractor(memory_dir: str = "memory") -> EntityExtractor:
    global _global_extractor
    if _global_extractor is None:
        _global_extractor = EntityExtractor(memory_dir)
    return _global_extractor


def extract_entities(text: str) -> List[str]:
    """对外接口：从文本中提取实体列表"""
    ext = get_extractor()
    return ext.extract_entities(text)


def extract_time_info(text: str) -> List[dict]:
    """对外接口：从文本中提取时间表达式

    返回结构化时间信息，用于后续构造带时间约束的搜索查询。
    """
    ext = get_extractor()
    return ext.extract_time_entities(text)


def format_time_query(time_entities: List[dict]) -> str:
    """将时间实体列表格式化为搜索查询后缀

    例：
      [{"time_relation": "year_range", "start_year": 2020, "end_year": 2024}]
      → "2020 2024"
    """
    if not time_entities:
        return ""

    tokens = []
    for e in time_entities:
        rel = e.get("time_relation")
        if rel == "year_range":
            start = e.get("start_year", "")
            end = e.get("end_year", "")
            tokens.append(f"{start} {end}")
        elif rel == "after_year":
            tokens.append(str(e.get("year", "")))
        elif rel == "before_year":
            tokens.append(str(e.get("year", "")))
        elif rel == "recent_years":
            tokens.append("最近")
        elif rel == "recent_generic":
            tokens.append("最近")
        elif rel == "single_year":
            tokens.append(str(e.get("year", "")))

    return " ".join(tokens)

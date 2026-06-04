import os
import time
import logging
from litellm import completion
import content_core.config as cfg
from content_core.tools.process.entity_extractor import extract_entities

logger = logging.getLogger(__name__)


def compare(docs_a: list, docs_b: list, query: str = "") -> str:
    """对比两组文档，从核心观点、方法、结论三方面分析异同"""
    if not docs_a or not docs_b:
        return "对比数据不足"

    # 使用 jieba + 技术词典抽取实体作为对比参考
    entities = extract_entities(query) if query else []

    entity_hint = ""
    if entities:
        entity_hint = f"\n\n请特别关注以下实体的对比分析：{' / '.join(entities[:4])}"

    prompt = f"""对比以下两组文档：

A组：
{chr(10).join(docs_a)}

B组：
{chr(10).join(docs_b)}

从核心观点、方法、结论三方面对比异同。{entity_hint}"""

    for attempt in range(1 + cfg.LLM_RETRY_COUNT):
        try:
            response = completion(
                model="deepseek/deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                api_base=os.getenv("DEEPSEEK_BASE_URL"),
                timeout=cfg.LLM_TIMEOUT,
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("LLM 返回了空白内容")
            return content
        except Exception as e:
            logger.warning("compare LLM 第 %d/%d 次失败: %s",
                           attempt + 1, cfg.LLM_RETRY_COUNT + 1, e)
            if attempt < cfg.LLM_RETRY_COUNT:
                time.sleep(cfg.LLM_RETRY_DELAY * (attempt + 1))
    logger.error("compare LLM 全部 %d 次尝试均失败", cfg.LLM_RETRY_COUNT + 1)
    return "[对比生成失败]"

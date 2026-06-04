import os
import time
import logging
from litellm import completion
import content_core.config as cfg

logger = logging.getLogger(__name__)


def reason(docs: list, query: str = "") -> str:
    """用 LLM 分析文档中的原因/原理类问题"""
    if not docs:
        return ""
    prompt = (
        f"用户问题：{query}\n\n"
        f"基于以下文档分析原因：\n\n{chr(10).join(docs)}\n\n"
        f"请从原理、机制、影响因素等方面给出分析。"
    )
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
            logger.warning("reason LLM 第 %d/%d 次失败: %s",
                           attempt + 1, cfg.LLM_RETRY_COUNT + 1, e)
            if attempt < cfg.LLM_RETRY_COUNT:
                time.sleep(cfg.LLM_RETRY_DELAY * (attempt + 1))
    logger.error("reason LLM 全部 %d 次尝试均失败", cfg.LLM_RETRY_COUNT + 1)
    return "[原因分析失败]"

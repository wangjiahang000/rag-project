import os
import time
import logging
from litellm import completion
import content_core.config as cfg

logger = logging.getLogger(__name__)


def summarize(docs: list) -> str:
    """用 LLM 对文档列表进行总结，返回 3-5 句核心内容"""
    if not docs:
        return ""
    for attempt in range(1 + cfg.LLM_RETRY_COUNT):
        try:
            response = completion(
                model="deepseek/deepseek-chat",
                messages=[{
                    "role": "user",
                    "content": f"用3-5句话总结以下文档的核心内容：\n\n{chr(10).join(docs)}"
                }],
                temperature=0,
                api_key=os.getenv("DEEPSEEK_API_KEY"),
                api_base=os.getenv("DEEPSEEK_BASE_URL"),
                timeout=cfg.LLM_TIMEOUT,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning("summarize LLM 第 %d/%d 次失败: %s",
                           attempt + 1, cfg.LLM_RETRY_COUNT + 1, e)
            if attempt < cfg.LLM_RETRY_COUNT:
                time.sleep(cfg.LLM_RETRY_DELAY * (attempt + 1))
    logger.error("summarize LLM 全部 %d 次尝试均失败", cfg.LLM_RETRY_COUNT + 1)
    return "[总结生成失败]"

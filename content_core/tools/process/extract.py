import os
import time
import logging
from litellm import completion
import content_core.config as cfg

logger = logging.getLogger(__name__)


def extract(docs: list, target: str = "") -> str:
    """用 LLM 从文档中提取关于 target 的具体信息"""
    if not docs:
        return ""
    for attempt in range(1 + cfg.LLM_RETRY_COUNT):
        try:
            response = completion(
                model="deepseek/deepseek-chat",
                messages=[{
                    "role": "user",
                    "content": f"从以下文档中提取关于'{target}'的具体信息：\n\n{chr(10).join(docs)}"
                }],
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
            logger.warning("extract LLM 第 %d/%d 次失败: %s",
                           attempt + 1, cfg.LLM_RETRY_COUNT + 1, e)
            if attempt < cfg.LLM_RETRY_COUNT:
                time.sleep(cfg.LLM_RETRY_DELAY * (attempt + 1))
    logger.error("extract LLM 全部 %d 次尝试均失败", cfg.LLM_RETRY_COUNT + 1)
    return "[信息提取失败]"

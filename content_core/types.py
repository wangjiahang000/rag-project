from typing import List, Dict, Optional
from pydantic import BaseModel
from enum import Enum

# ── 时间表达式类型 ──

class TimeRelation(str, Enum):
    """时间关系类型"""
    YEAR_RANGE = "year_range"         # 从X到Y年：2020-2024
    AFTER_YEAR = "after_year"         # X年之后：2020年后
    BEFORE_YEAR = "before_year"       # X年之前：2020年前
    RECENT_YEARS = "recent_years"     # 最近N年：最近5年
    RECENT_GENERIC = "recent_generic" # 近年来/近些年
    SINGLE_YEAR = "single_year"       # 具体某一年：2020年


class TimeExpression(BaseModel):
    """解析后的时间表达式"""
    relation: TimeRelation
    year: Optional[int] = None         # 起始/单一 年份
    end_year: Optional[int] = None     # 结束年份（仅 YEAR_RANGE）
    years_back: Optional[int] = None   # 最近几年（仅 RECENT_YEARS）
    original_text: str = ""            # 原始匹配文本


class TaskNode(BaseModel):
    id: str
    op: str                          # 工具名
    args: Dict = {}                  # 工具参数
    depends_on: List[str] = []       # 依赖的节点id
    fallback: Optional[str] = None   # 失败时备用工具
    retry: int = 1                   # 重试次数

class TaskGraph(BaseModel):
    user_tasks: List[str]            # ["retrieve", "compare"]
    nodes: List[TaskNode]            # 执行节点列表
    entities: list = []  
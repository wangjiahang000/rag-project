from typing import List, Dict, Optional
from pydantic import BaseModel

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
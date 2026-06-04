"""用户画像 API"""

import logging
from fastapi import APIRouter

from backend.session import session_manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/profile/{session_id}")
def get_profile(session_id: str):
    """获取用户画像"""
    session = session_manager.get_or_create(session_id)
    return {
        "session_id": session.id,
        "total_queries": session.total_queries,
        "turns": len(session.history),
        "top_interests": session.top_interests,
        "top_intents": session.top_intents,
    }


@router.delete("/profile/{session_id}")
def delete_profile(session_id: str):
    """清除用户数据"""
    try:
        del session_manager._sessions[session_id]
        return {"status": "ok", "message": "用户数据已清除"}
    except KeyError:
        return {"status": "ok", "message": "会话不存在"}


@router.get("/profile/{session_id}/interests")
def get_interests(session_id: str):
    """获取兴趣标签"""
    session = session_manager.get_or_create(session_id)
    return {"interests": dict(session.top_interests)}


@router.get("/stats")
def get_stats():
    """全局统计数据"""
    return {
        "active_sessions": session_manager.active_count,
        "session_ids": list(session_manager._sessions.keys()),
    }

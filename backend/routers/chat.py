import asyncio
import json
import os
import time
import logging
from asyncio import Semaphore

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from litellm import acompletion

from backend.schemas import QueryRequest, ChatResponse, CitationInfo
from backend.dependencies import (
    get_router,
    get_generator,
    get_executor,
    get_vector_store,
)
from backend.cache import query_cache
from backend.metrics import metrics as m
from backend.session import (
    session_manager,
    Turn,
    resolve_references as do_resolve,
    needs_reference_resolution,
)
from content_core.retrieval.enhanced_search import build_enhanced_context
import content_core.config as cfg

logger = logging.getLogger(__name__)

router = APIRouter()

# ── LLM 限流 ──
_llm_semaphore = Semaphore(5)  # 最多 5 个并发 LLM 调用
_llm_semaphore_sync = __import__("threading").Semaphore(5)  # 同步调用限流

# ── 对话历史模板 ──
_HISTORY_PREFIX = """对话历史：
{history}

"""
_CITATION_PROMPT = """你是一个严谨的学术研究助手。请基于以下文献内容回答问题。

要求：
1. 在答案中标注引用来源，格式为 [1]、[2] 等，对应下方文献列表的编号
2. 如果文献与问题不相关或不足以回答问题，请回答"没有检索到相关文献"
3. 不要编造文献中不存在的信息
4. 用中文回答

文献列表：
{context}

{history_section}请回答问题：{question}"""


def _format_history(turns: list) -> str:
    """格式化对话历史"""
    if not turns:
        return ""
    lines = []
    for t in turns[-6:]:  # 最近 6 轮
        role = "用户" if t.role == "user" else "助手"
        lines.append(f"{role}: {t.content[:200]}")
    return "\n".join(lines)


def _call_llm_sync(prompt: str) -> str:
    """同步 LLM 调用（线程池中运行，带限流）"""
    acquired = _llm_semaphore_sync.acquire(blocking=True, timeout=cfg.LLM_TIMEOUT)
    if not acquired:
        logger.error("LLM 限流等待超时")
        return "没有检索到相关文献"
    try:
        for attempt in range(1 + cfg.LLM_RETRY_COUNT):
            try:
                from litellm import completion
                response = completion(
                    model="deepseek/deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.01,
                    api_key=os.getenv("DEEPSEEK_API_KEY"),
                    api_base=os.getenv("DEEPSEEK_BASE_URL"),
                    timeout=cfg.LLM_TIMEOUT,
                )
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("LLM 返回了空白内容")
                return content
            except Exception as e:
                wait = cfg.LLM_RETRY_DELAY * (2 ** attempt)  # 指数退避
                logger.warning("LLM 第 %d/%d 次失败: %s，%.1fs 后重试",
                               attempt + 1, cfg.LLM_RETRY_COUNT + 1, e, wait)
                if attempt < cfg.LLM_RETRY_COUNT:
                    time.sleep(wait)
        logger.error("LLM 全部 %d 次尝试均失败", cfg.LLM_RETRY_COUNT + 1)
        return "没有检索到相关文献"
    finally:
        _llm_semaphore_sync.release()


def _format_references(citations: list) -> str:
    """格式化学术参考文献列表"""
    if not citations:
        return ""
    lines = []
    for c in citations:
        parts = []
        parts.append(f"[{c['index']}] {c['source']}")
        if c.get("year"):
            parts.append(f"({c['year']})")
        lines.append("  " + " ".join(parts))
    return "\n".join(lines)


# ── /chat（非流式，向后兼容）──

@router.post("/chat", response_model=ChatResponse)
async def chat(
    query: QueryRequest,
    router_=Depends(get_router),
    generator=Depends(get_generator),
    executor=Depends(get_executor),
    vector_store=Depends(get_vector_store),
):
    # 1. 查询缓存
    cached = query_cache.get(query.question)
    if cached:
        m.inc("cache_hit", {"endpoint": "chat"})
        return ChatResponse(**cached)
    m.inc("cache_miss", {"endpoint": "chat"})

    # 2. 会话管理
    session = session_manager.get_or_create(query.session_id)
    history_turns = session.get_recent(6)
    history_text = _format_history(history_turns)

    # 3. 指代消解
    resolved = query.question
    if needs_reference_resolution(query.question) and history_turns:
        resolved = await do_resolve(query.question, history_turns)

    # 4. 意图路由（同步，放 executor 避免阻塞）
    result = await asyncio.to_thread(router_.route, resolved)
    tasks = result["user_tasks"]
    entities = result.get("entities", [])

    # 5. 生成执行图 + DAG 执行（异步）
    graphs = await asyncio.to_thread(generator.generate, tasks, resolved)
    all_plan = []
    for graph in graphs:
        dag_results = await executor.execute(graph)
        last_id = graph.nodes[-1].id
        all_plan.extend(
            {"id": n.id, "op": n.op, "args": n.args} for n in graph.nodes
        )

    # 6. 增强检索 + 结构化上下文（同步）
    context_data = await asyncio.to_thread(
        build_enhanced_context,
        query=resolved,
        vector_store=vector_store,
        tasks=tasks,
        k=cfg.CONTEXT_MAX_DOCS,
    )
    context = context_data["context"]
    citations_raw = context_data["citations"]

    # 7. 判断是否检索到相关文献
    has_results = context != "暂无相关文献"
    if not has_results:
        answer = "没有检索到相关文献"
        citations = []
    else:
        # 8. LLM 生成（同步，放 executor）
        history_section = ""
        if history_text:
            history_section = _HISTORY_PREFIX.format(history=history_text)
        prompt = _CITATION_PROMPT.format(
            context=context,
            history_section=history_section,
            question=query.question,
        )
        llm_answer = await asyncio.to_thread(_call_llm_sync, prompt)

        refs = _format_references(citations_raw)
        if refs:
            answer = llm_answer + "\n\n---\n\n参考文献：\n" + refs
        else:
            answer = llm_answer

        citations = [CitationInfo(**c) for c in citations_raw]

    # 9. 保存会话
    session.add_turn(Turn("user", query.question, tasks=tasks, entities=entities))
    session.add_turn(Turn("assistant", answer, tasks=tasks))

    # 10. 写入缓存
    resp = ChatResponse(
        user_tasks=tasks,
        plan=all_plan,
        answer=answer,
        citations=citations,
        source=result.get("source", ""),
    )
    query_cache.set(query.question, resp.model_dump())

    # 指标
    m.inc("llm_calls", {"endpoint": "chat"})
    m.inc(f"routing_source:{result.get('source', 'unknown')}")

    return resp


# ── /chat/stream（SSE 流式输出）──

@router.post("/chat/stream")
async def chat_stream(
    query: QueryRequest,
    router_=Depends(get_router),
    generator=Depends(get_generator),
    executor=Depends(get_executor),
    vector_store=Depends(get_vector_store),
):
    # 会话管理
    session = session_manager.get_or_create(query.session_id)
    history_turns = session.get_recent(6)
    history_text = _format_history(history_turns)

    # 指代消解
    resolved = query.question
    if needs_reference_resolution(query.question) and history_turns:
        resolved = await do_resolve(query.question, history_turns)

    # 同步流水线（路由 → 图生成 → 执行 → 检索）
    result = await asyncio.to_thread(router_.route, resolved)
    tasks = result["user_tasks"]
    entities = result.get("entities", [])

    graphs = await asyncio.to_thread(generator.generate, tasks, resolved)
    all_plan = []
    for graph in graphs:
        dag_results = await executor.execute(graph)
        last_id = graph.nodes[-1].id
        all_plan.extend(
            {"id": n.id, "op": n.op, "args": n.args} for n in graph.nodes
        )

    context_data = await asyncio.to_thread(
        build_enhanced_context,
        query=resolved,
        vector_store=vector_store,
        tasks=tasks,
        k=cfg.CONTEXT_MAX_DOCS,
    )
    context = context_data["context"]
    citations_raw = context_data["citations"]
    has_results = context != "暂无相关文献"

    if not has_results:
        answer = "没有检索到相关文献"
        session.add_turn(Turn("user", query.question, tasks=tasks, entities=entities))
        session.add_turn(Turn("assistant", answer))
        return ChatResponse(
            user_tasks=tasks, plan=all_plan, answer=answer,
            citations=[], source=result.get("source", ""),
        )

    history_section = ""
    if history_text:
        history_section = _HISTORY_PREFIX.format(history=history_text)
    prompt = _CITATION_PROMPT.format(
        context=context,
        history_section=history_section,
        question=query.question,
    )

    async def event_stream():
        full_answer = ""
        try:
            async with _llm_semaphore:
                response = await acompletion(
                    model="deepseek/deepseek-chat",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.01,
                    api_key=os.getenv("DEEPSEEK_API_KEY"),
                    api_base=os.getenv("DEEPSEEK_BASE_URL"),
                    timeout=cfg.LLM_TIMEOUT,
                    stream=True,
                )

                async for chunk in response:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        full_answer += delta
                        yield f"data: {json.dumps({'type': 'token', 'data': delta}, ensure_ascii=False)}\n\n"

        except asyncio.TimeoutError:
            logger.error("流式 LLM 超时")
            yield f"data: {json.dumps({'type': 'error', 'data': '生成超时'}, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.error("流式 LLM 失败: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'data': '生成回答时出错'}, ensure_ascii=False)}\n\n"

        # 追加参考文献
        refs = _format_references(citations_raw)
        if refs:
            ref_block = "\n\n---\n\n参考文献：\n" + refs
            full_answer += ref_block
            yield f"data: {json.dumps({'type': 'references', 'data': ref_block}, ensure_ascii=False)}\n\n"

        # 元数据
        meta = {
            "type": "done",
            "tasks": tasks,
            "source": result.get("source", ""),
            "plan": all_plan,
            "citations": citations_raw,
        }
        yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

        # 指标
        m.inc("llm_calls", {"endpoint": "stream"})
        m.inc(f"routing_source:{result.get('source', 'unknown')}")

        # 保存会话
        session.add_turn(Turn("user", query.question, tasks=tasks, entities=entities))
        session.add_turn(Turn("assistant", full_answer, tasks=tasks))

    return StreamingResponse(event_stream(), media_type="text/event-stream")

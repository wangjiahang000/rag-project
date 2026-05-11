# tests/test_task_router.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from content_core.task_router import TaskRouter

router = TaskRouter()
router.sbert.threshold = 0.52

print("TaskRouter 详细诊断")
print("输入 'q' 退出")
print("=" * 70)

while True:
    query = input("\n请输入句子: ").strip()
    if query.lower() == 'q':
        print("退出")
        break
    if not query:
        continue

    # ── 1. 关键词规则 ──
    rule_tasks = router._rule_match(query)
    print(f"\n📌 关键词命中: {rule_tasks if rule_tasks else '未命中'}")

    # ── 2. SBERT 六个意图概率 ──
    query_emb = router.sbert.model.encode(query)
    print(f"\n📊 SBERT 六个意图相似度:")
    for intent in router.sbert.INTENT_LABELS:
        anchor_emb = router.sbert.anchor_embs[intent]
        sim = np.dot(query_emb, anchor_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(anchor_emb) + 1e-10
        )
        bar = "█" * int(sim * 40) + "░" * (40 - int(sim * 40))
        print(f"  {intent:12s}  {sim:.4f}  {bar}")

    # ── 3. 闲聊检测 ──
    is_chat = router._is_chitchat(query)
    print(f"\n💬 闲聊检测: {'是' if is_chat else '否'}")

    # ── 4. 最终结果 ──
    r = router.route(query)
    print(f"\n✅ 最终意图: {r['user_tasks']}")
    print(f"📍 来源: {r['source']}")
    print(f"📂 资源: {r['resource_hint']}")
    print(f"🔗 复杂度: {r['complexity']}")
    print(f"👤 实体: {r['entities']}")
    print("-" * 70)